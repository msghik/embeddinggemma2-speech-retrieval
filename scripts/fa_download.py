import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from huggingface_hub import snapshot_download
for r, d, pats in [("MohammadGholizadeh/common_voice_17.0_fa_ttv", "cv", ["data/train-*"]),
                   ("MohammadGholizadeh/youtube-farsi", "youtube", ["data/*"]),
                   ("MohammadGholizadeh/filimo-farsi", "filimo", ["data/*"])]:
    snapshot_download(r, repo_type="dataset", local_dir=f"{BASE}/data/fa/{d}", allow_patterns=pats, max_workers=16)
    print("done", r, flush=True)
