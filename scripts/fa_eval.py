"""Final eval: audio<->text retrieval on CV test, held-out YouTube videos, filimo test sample."""
import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, json, zlib, numpy as np, pandas as pd
sys.path.insert(0, f"{BASE}/scripts")
from common import load_model
from fa_zeroshot import metrics
D = f"{BASE}/data/fa"
mp, tag = sys.argv[1], sys.argv[2]
cv, yt, fl = (pd.read_parquet(f"{D}/manifest_{d}.parquet") for d in ["cv", "youtube", "filimo"])
held = yt.group.map(lambda v: zlib.crc32(v.encode()) % 10 == 0)   # same rule as fa_train.py
SETS = {"cv_test": cv[cv.split == "test"],
        "youtube_heldout_videos": yt[held & (yt.split == "test")],
        "filimo_test_5k": fl[fl.split == "test"].sample(5000, random_state=0)}
train_keys = set(pd.concat([cv[cv.split == "train"], yt[~held], fl[fl.split == "train"]]).key)
m = load_model(mp); res = {}
for name, df in SETS.items():
    keys = sorted(set(df.key)); rep = dict(zip(df.key, df.text)); idx = {k: i for i, k in enumerate(keys)}
    order = np.argsort(df.dur.values); paths = df.path.values
    A = np.zeros((len(df), 768), np.float32)
    for i in range(0, len(df), 64):
        b = order[i:i+64]; A[b] = m.encode(list(paths[b]), batch_size=64, convert_to_numpy=True)
    T = m.encode([rep[k] for k in keys], prompt_name="SearchQuery", batch_size=128, convert_to_numpy=True)
    gold = np.array([idx[k] for k in df.key])
    res[name] = metrics(A, T, gold, np.random.default_rng(0))
    novel = ~df.key.isin(train_keys).values  # clips whose sentence never occurs in the training mix
    res[name]["a2t_R@1_novel_sentences"] = float(((A @ T.T).argmax(1) == gold)[novel].mean()); res[name]["n_novel"] = int(novel.sum())
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)
json.dump(res, open(f"{BASE}/results/fa_eval_{tag}.json", "w"), indent=1)
