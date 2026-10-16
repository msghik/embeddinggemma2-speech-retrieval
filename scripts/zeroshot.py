import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, time, json, numpy as np
sys.path.insert(0, f"{BASE}/scripts")
from common import *
model_path = sys.argv[1] if len(sys.argv) > 1 else "google/embeddinggemma-2"
tag = sys.argv[2] if len(sys.argv) > 2 else "base"
m = load_model(model_path)
TEMPLATES = {  # (template, prompt_name)
    "word|Classification": ("{}", "Classification"),
    "word|none": ("{}", None),
    "spoken|Classification": ("someone saying the word \"{}\"", "Classification"),
    "word|SearchQuery": ("{}", "SearchQuery"),
    "word|Document": ("{}", "Document"),
}
res = {}
for sp in ["validation", "test"]:
    files, y = split(sp)
    t0 = time.time(); A = embed_audio(m, files); dt = time.time() - t0
    np.save(f"{BASE}/results/{tag}_{sp}_audio.npy", A)
    print(f"{sp}: {len(files)} clips embedded in {dt:.0f}s ({len(files)/dt:.0f} clips/s)", flush=True)
    for k, (tpl, pn) in TEMPLATES.items():
        T = m.encode([tpl.format(w) for w in WORDS], prompt_name=pn, convert_to_numpy=True)
        pred = (A @ T.T).argmax(1)
        res[f"{sp}/{k}"] = float((pred == y).mean())
        print(f"  {k:28s} acc={res[f'{sp}/{k}']:.4f}", flush=True)
json.dump(res, open(f"{BASE}/results/{tag}_zeroshot.json", "w"), indent=1)
