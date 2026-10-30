"""Closed-loop evaluation of a GR00T N1.7 policy server on the ColosseumV2 dual-arm Panda tasks.

Runs in the colosseum_v2 conda env and talks to a policy server started from the N1.7 venv:
    CUDA_VISIBLE_DEVICES=<g> .venv/bin/python gr00t/eval/run_gr00t_server.py --model-path <ckpt> \
        --embodiment-tag NEW_EMBODIMENT --use-sim-policy-wrapper --port 6400
    CUDA_VISIBLE_DEVICES=<render g> python multiview/colosseum/eval_colosseum.py --port 6400 --n-episodes 50 \
        --out <dir>
Observations are mapped exactly as convert_colosseum.py stores the training data (camera keys, state layout,
first phrasing of the instruction); the camera set comes from the server's modality config. Each returned
16-step chunk is executed for --n-action-steps steps; gripper commands are binarised to -1 / 1. An episode
counts as a success once the env reports success; it stops then or after 1.5x the longest demo of its task.

--replay replays the demo actions of the training episodes instead of querying a server (checks the env,
the observation/action layout and the success bookkeeping without a model).
"""

import argparse
import io
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers the ColosseumV2 envs)
import numpy as np
import yaml

SRC = Path("/nfs/data/sohyun/data/colosseum-v2/trajectory__cv2-full__pd_joint_pos__100")
LANG_YAML = Path.home() / "project/ColosseumV2/mani_skill/envs/tasks/tabletop/colosseum_v2/language_randomizations.yaml"
TASKS = ["DualArmLiftPot-v1", "DualArmLiftTray-v1", "DualArmPenCap-v1", "DualArmPickBottle-v1", "DualArmPickCube-v1",
         "DualArmPourPot-v1", "DualArmPushBox-v1", "DualArmStack3Cube-v1", "DualArmStackCube-v1", "DualArmThreading-v1"]
CAMERAS = {"external1": "external1_camera", "external2": "external2_camera",
           "left_hand": "panda1_hand_camera", "right_hand": "panda2_hand_camera"}


def state_from_qpos(q: np.ndarray) -> dict:
    """(18,) interleaved ManiSkill qpos -> the four state parts of convert_colosseum.py."""
    return {"left_arm": q[0:14:2], "left_gripper": q[14:16], "right_arm": q[1:14:2], "right_gripper": q[16:18]}


def to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


class Client:
    """Minimal ZMQ client for gr00t/policy/server_client.PolicyServer (same msgpack + numpy wire format)."""

    def __init__(self, host, port, timeout_ms=60000):
        import msgpack
        import msgpack_numpy as mnp
        import zmq
        self.msgpack, self.mnp = msgpack, mnp
        self.sock = zmq.Context().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def _decode(self, obj):
        if isinstance(obj, dict) and (obj.get("__ndarray_class__") or obj.get(b"__ndarray_class__")):
            return np.load(io.BytesIO(obj.get("as_npy", obj.get(b"as_npy"))), allow_pickle=False)
        if isinstance(obj, dict) and ("__ModalityConfig__" in obj or "__ModalityConfig_class__" in obj):
            payload = obj["as_json"]
            return json.loads(payload) if isinstance(payload, (str, bytes)) else payload  # plain dict
        return self.mnp.decode(obj)

    def call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.sock.send(self.msgpack.packb(req, default=self.mnp.encode))
        resp = self.msgpack.unpackb(self.sock.recv(), object_hook=self._decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(resp["error"])
        return resp


def make_env(env_id):
    return gym.make(env_id, obs_mode="rgb", control_mode="pd_joint_pos", render_mode="rgb_array",
                    reward_mode="none", _env_id=env_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6400)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--n-episodes", type=int, default=50, help="per task")
    ap.add_argument("--seed-start", type=int, default=10000, help="eval seeds (training demos use seeds < 1000)")
    ap.add_argument("--n-action-steps", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video", action="store_true", help="save external1 videos of the first 3 episodes per task")
    ap.add_argument("--replay", action="store_true", help="replay training demo actions (no server)")
    args = ap.parse_args()

    meta = json.load(open(f"{SRC}.json"))
    lang = yaml.safe_load(open(LANG_YAML))
    max_len = meta["max_episode_lengths"]
    args.out.mkdir(parents=True, exist_ok=True)

    if args.replay:
        import h5py
        h5 = h5py.File(f"{SRC}.h5", "r")
        demos = {}
        for e in meta["episodes"]:
            demos.setdefault(e["env_id"], []).append(e)
        video_keys = ["external1"]
    else:
        client = Client(args.host, args.port)
        mc = client.call("get_modality_config")
        video_keys = list(mc["video"]["modality_keys"] if isinstance(mc["video"], dict) else mc["video"].modality_keys)
        print("policy cameras:", video_keys, flush=True)

    results = {}
    for env_id in args.tasks:
        env = make_env(env_id)
        instr = lang[env_id][0]
        limit = int(1.5 * max_len[env_id])
        succ, lens = [], []
        for k in range(args.n_episodes):
            if args.replay:
                e = demos[env_id][k]
                seed, demo_act = e["episode_seed"], h5[f"traj_{e['episode_id']}"]["actions"][:]
            else:
                seed = args.seed_start + k
                client.call("reset", {"options": None})
            obs, _ = env.reset(seed=seed)
            done, t, chunk, frames = False, 0, [], []
            while not done and t < limit:
                if args.replay:
                    if t >= len(demo_act):
                        break
                    action = demo_act[t]
                else:
                    if not chunk:
                        q = to_np(obs["agent"]["qpos"])[0].astype(np.float32)
                        o = {f"state.{k_}": v[None, None].astype(np.float32) for k_, v in state_from_qpos(q).items()}
                        for key in video_keys:
                            o[f"video.{key}"] = to_np(obs["sensor_data"][CAMERAS[key]]["rgb"]).astype(np.uint8)[:, None]
                        o["task"] = [instr]
                        act, _ = client.call("get_action", {"observation": o, "options": None})
                        a = {k_.replace("action.", ""): np.asarray(v)[0] for k_, v in act.items()}
                        for i in range(args.n_action_steps):
                            chunk.append(np.concatenate([a["left_arm"][i], [1.0 if a["left_gripper"][i][0] >= 0 else -1.0],
                                                         a["right_arm"][i], [1.0 if a["right_gripper"][i][0] >= 0 else -1.0]]))
                    action = chunk.pop(0)
                if args.video and k < 3:
                    frames.append(to_np(obs["sensor_data"]["external1_camera"]["rgb"])[0].astype(np.uint8))
                obs, _, term, trunc, info = env.step(np.asarray(action, dtype=np.float32)[None])
                t += 1
                done = bool(to_np(info["success"]).reshape(-1)[0])
            succ.append(done)
            lens.append(t)
            if frames:
                import imageio
                imageio.mimsave(args.out / f"{env_id}_ep{k}_{'ok' if done else 'fail'}.mp4", frames, fps=20)
            print(f"{env_id} ep {k} seed {seed}: {'success' if done else 'fail'} in {t} steps", flush=True)
        env.close()
        results[env_id] = {"success_rate": float(np.mean(succ)), "n": len(succ), "success": succ, "steps": lens}
        print(f"== {env_id}: {np.mean(succ):.2f} ({sum(succ)}/{len(succ)})", flush=True)
        json.dump(results, open(args.out / "results.json", "w"), indent=1)
    mean = float(np.mean([r["success_rate"] for r in results.values()]))
    results["_mean"] = mean
    json.dump(results, open(args.out / "results.json", "w"), indent=1)
    print(f"mean success over {len(args.tasks)} tasks: {mean:.3f}")


if __name__ == "__main__":
    main()
