"""WER/CER of a Whisper checkpoint on the Persian test sets. Usage: wh_eval.py <model> <tag> [max_clips_per_set]"""
import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json, time; sys.path.insert(0, f"{BASE}/scripts")
from wh_common import *
mp, tag = sys.argv[1], sys.argv[2]; cap = int(sys.argv[3]) if len(sys.argv) > 3 else None
m, proc = load_whisper(mp); res = {}
for name, df in test_sets().items():
    if cap: df = df.sample(min(cap, len(df)), random_state=0)
    t0 = time.time(); hyp = transcribe(m, proc, df.path.tolist())
    res[name] = {**scores(df.text.tolist(), hyp), "sec": round(time.time() - t0)}
    print(name, res[name], flush=True)
    pd.DataFrame({"ref": df.text.values, "hyp": hyp}).to_csv(f"{BASE}/results/wh_{tag}_{name}_hyps.tsv", sep="\t", index=False)
json.dump(res, open(f"{BASE}/results/wh_eval_{tag}.json", "w"), indent=1)
