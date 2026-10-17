"""Recompute WER/CER from saved hypothesis TSVs. Usage: wh_rescore.py <tag>"""
import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json, glob; sys.path.insert(0, f"{BASE}/scripts")
from wh_common import scores, pd
tag = sys.argv[1]; res = {}
for f in sorted(glob.glob(f"{BASE}/results/wh_{tag}_*_hyps.tsv")):
    name = f.split(f"wh_{tag}_")[1].replace("_hyps.tsv", "")
    df = pd.read_csv(f, sep="\t", keep_default_na=False, quoting=3)
    res[name] = scores(df.ref.tolist(), df.hyp.tolist()); print(name, res[name])
json.dump(res, open(f"{BASE}/results/wh_eval_{tag}.json", "w"), indent=1)
