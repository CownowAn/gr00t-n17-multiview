"""Breakdown of a StopRollingCanTable / StopRollingBallTable eval (eval_trace_n17.sh output dir):
success per roll direction, false starts (a hand moved before the object was pushed), and when the
blocking hand first moved relative to the push.
usage: python rolling_breakdown.py <eval dir> [<eval dir> ...]   (dirs hold episodes_L3/*.jsonl)"""
import glob
import json
import re
import sys

import numpy as np

DIRS = ("front", "diag", "side")


def counted(d):
    """Seeds of the episodes rollout_policy.py counted (the first N to finish)."""
    try:
        for line in open(f"{d}/rollout_L3.log"):
            if line.startswith("results:"):
                m = re.search(r"'episode_seed': \[([^\]]*)\]", line)
                if m:
                    return {int(x) for x in m.group(1).split(",") if x.strip()}
    except OSError:
        pass
    return None


def summarize(d):
    recs = [json.loads(l) for f in glob.glob(f"{d}/episodes_L3/*.jsonl") for l in open(f)]
    by = {r["seed"]: r for r in recs}
    keep = counted(d)
    recs = [by[s] for s in sorted(keep) if s in by] if keep else sorted(by.values(), key=lambda r: r["seed"])[:100]
    rows = []
    for r in recs:
        push, side = r["layout"]["push_step"], r["side"]
        p0 = np.array(r["steps"][0]["palms"][side])
        moved = next((t for t, s in enumerate(r["steps"]) if np.linalg.norm(np.array(s["palms"][side]) - p0) > 30), None)
        rows.append(dict(direction=r["direction"], success=bool(r["success"]), fs=bool(r["steps"][-1]["false_start"]),
                         move_rel=(moved - push) if moved is not None else None))
    return rows


for d in sys.argv[1:]:
    rows = summarize(d)
    by = " ".join(f"{k} {sum(x['success'] for x in rows if x['direction'] == k)}/{sum(1 for x in rows if x['direction'] == k)}"
                  for k in DIRS)
    mv = [x["move_rel"] for x in rows if x["move_rel"] is not None and not x["fs"]]
    print(f"{d.rstrip('/').split('/')[-1]}: n={len(rows)} success {100 * np.mean([x['success'] for x in rows]):.0f}% | {by} | "
          f"false start {sum(x['fs'] for x in rows)} | move after push median {np.median(mv) if mv else float('nan'):.0f} steps")
