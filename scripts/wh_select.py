"""Pick a checkpoint on a larger validation sample (never test). Usage: wh_select.py <model_dir>... """
import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json; sys.path.insert(0, f"{BASE}/scripts")
from wh_common import *
cv, yt, fl = manifests()
VAL = {"cv_val": cv[cv.split == "validation"].sample(2000, random_state=1),
       "yt_heldout_val": yt[held_out(yt) & (yt.split != "test")].sample(1000, random_state=1),
       "filimo_dev": fl[fl.split == "dev"].sample(2000, random_state=1)}
out = {}
for mp in sys.argv[1:]:
    m, proc = load_whisper(mp)
    out[mp] = {k: scores(v.text.tolist(), transcribe(m, proc, v.path.tolist())) for k, v in VAL.items()}
    out[mp]["mean_WER"] = float(np.mean([out[mp][k]["WER"] for k in VAL])); print(mp, json.dumps(out[mp]), flush=True)
    del m; torch.cuda.empty_cache()
json.dump(out, open(f"{BASE}/results/wh_select.json", "w"), indent=1)
