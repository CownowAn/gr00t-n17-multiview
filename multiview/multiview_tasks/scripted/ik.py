"""mink-based IK for one GR1 arm, solved on a copy of the simulation model.

Besides the arm, ``extra_joints`` (e.g. waist yaw and pitch) can move to extend reach.

The env's arm controllers take absolute joint positions, so the IK output can be
sent as the action directly. Every joint outside the arm is frozen, which keeps the
other arm, the hands and the scene unchanged in the IK solution.
"""

import mink
import mujoco
import numpy as np

ARM_JOINTS = {
    side: [
        f"robot0_{s}_{j}"
        for j in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow_pitch",
            "wrist_yaw",
            "wrist_roll",
            "wrist_pitch",
        )
    ]
    for side, s in (("right", "r"), ("left", "l"))
}


def _collision_geoms(model: mujoco.MjModel, body_prefix: str) -> list:
    """Colliding geoms of the bodies whose names start with ``body_prefix``."""
    return [
        g
        for g in range(model.ngeom)
        if (model.geom_contype[g] or model.geom_conaffinity[g])
        and model.body(model.geom_bodyid[g]).name.startswith(body_prefix)
    ]


class ArmIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        side: str,
        frame_name: str,
        frame_type: str = "body",
        orientation_cost: float = 0.3,
        posture_cost: float = 1e-3,
        extra_joints: tuple = (),
        extra_posture_cost: float = None,
    ):
        self.model = model
        self.configuration = mink.Configuration(model)
        joint_ids = [model.joint(name).id for name in ARM_JOINTS[side]]
        extra_ids = [model.joint(name).id for name in extra_joints]
        self.qpos_idx = np.array([model.jnt_qposadr[j] for j in joint_ids])
        self.extra_qpos_idx = np.array([model.jnt_qposadr[j] for j in extra_ids], dtype=int)
        active = {int(model.jnt_dofadr[j]) for j in joint_ids + extra_ids}
        frozen = [i for i in range(model.nv) if i not in active]

        self.frame_task = mink.FrameTask(
            frame_name=frame_name,
            frame_type=frame_type,
            position_cost=1.0,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )
        # A higher posture cost on extra_joints makes the IK use them only when the arm
        # alone falls short.
        cost = np.full(model.nv, posture_cost)
        if extra_posture_cost is not None:
            cost[[int(model.jnt_dofadr[j]) for j in extra_ids]] = extra_posture_cost
        self.posture_task = mink.PostureTask(model, cost=cost)
        self.tasks = [self.frame_task, self.posture_task]
        self.constraints = [mink.DofFreezingTask(model, dof_indices=frozen)]
        self.limits = [mink.ConfigurationLimit(model)]

        # Keep the hand off its own wrist link and the torso: a strongly bent wrist
        # otherwise jams the palm against the wrist, and buttons near the body
        # push the thumb into the torso.
        s = side[0]
        hand = _collision_geoms(model, f"gripper0_{side}_")
        body = _collision_geoms(model, f"robot0_{s}_hand_yaw") + _collision_geoms(
            model, "robot0_torso_"
        ) + _collision_geoms(model, "robot0_whole_body")
        self.limits.append(
            mink.CollisionAvoidanceLimit(
                model,
                geom_pairs=[(hand, body)],
                minimum_distance_from_collisions=0.01,
                collision_detection_distance=0.05,
            )
        )

    def _clip_to_limits(self, qpos: np.ndarray) -> np.ndarray:
        # Contacts push some joints (thumb, buttons, cabinet doors) slightly past their
        # range; a frozen joint outside its limit makes the IK QP infeasible.
        q = qpos.copy()
        m = self.model
        for j in range(m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
                a = m.jnt_qposadr[j]
                q[a] = np.clip(q[a], *m.jnt_range[j])
        return q

    def reset(self, qpos: np.ndarray):
        """Start from the current sim state; the posture task pulls toward it."""
        q = self._clip_to_limits(qpos)
        self.configuration.update(q=q)
        self.posture_task.set_target(q)

    def frame_pose(self) -> mink.SE3:
        return self.configuration.get_transform_frame_to_world(
            self.frame_task.frame_name, self.frame_task.frame_type
        )

    def solve(self, target: mink.SE3, iters: int = 5, dt: float = 0.05) -> np.ndarray:
        """Move the IK configuration toward ``target``; return the arm joint positions.

        Positions of ``extra_joints`` are read from ``self.configuration.q[self.extra_qpos_idx]``.
        """
        self.frame_task.set_target(target)
        for _ in range(iters):
            vel = mink.solve_ik(
                self.configuration,
                self.tasks,
                dt,
                solver="daqp",
                damping=1e-3,
                limits=self.limits,
                constraints=self.constraints,
            )
            self.configuration.integrate_inplace(vel, dt)
        return self.configuration.q[self.qpos_idx].copy()
