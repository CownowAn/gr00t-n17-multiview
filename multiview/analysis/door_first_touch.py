"""Door tasks (HoldDoorTakeBlockV3 / HoldDoorTakeFruitV3): success and the first item touched per episode
(target / wrong / none) from eval_trace_n17.sh output dirs (episodes_L3/*.jsonl).
usage: python door_first_touch.py <eval dir> [<eval dir> ...]"""
import collections, glob, json, re, sys
import numpy as np
for d in sys.argv[1:]:
    st = open(d + "/status.log").read()
    m = re.findall(r"success rate:\s+([0-9.]+)", st)
    K = collections.Counter()
    for f in glob.glob(d + "/episodes_L3/*.jsonl"):
        for l in open(f):
            r = json.loads(l); b0 = r["steps"][0]["blocks_mm"]; first = None
            for x in r["steps"]:
                mv = [k for k in b0 if np.linalg.norm(np.array(x["blocks_mm"][k]) - b0[k]) > 10]
                if mv: first = mv[0]; break
            K["none" if first is None else ("target" if first == r["target"] else "wrong")] += 1
    print(f"{d.split('/')[-1]}: success {100*float(m[-1]) if m else float('nan'):.0f}% first-touch {dict(K)}")
