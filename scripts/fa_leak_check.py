import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import zlib, pandas as pd, numpy as np
D = f"{BASE}/data/fa"
cv, yt, fl = (pd.read_parquet(f"{D}/manifest_{d}.parquet") for d in ["cv", "youtube", "filimo"])
held = yt.group.map(lambda v: zlib.crc32(v.encode()) % 10 == 0)
train = pd.concat([cv[cv.split == "train"], yt[~held], fl[fl.split == "train"]])
tk = set(train.key)
for name, df in {"cv_validation": cv[cv.split == "validation"], "cv_test": cv[cv.split == "test"],
                 "yt_heldout_test": yt[held & (yt.split == "test")], "filimo_dev": fl[fl.split == "dev"], "filimo_test": fl[fl.split == "test"]}.items():
    inn = df.key.isin(tk)
    long_ = df.key.str.split().str.len() >= 4
    print(f"{name:16s} n={len(df):6d}  sentence also in train: {inn.mean():.3f}   (among >=4-word sentences: {inn[long_].mean():.3f})")
print("cv speakers shared train/test:", len(set(cv[cv.split=='train'].group) & set(cv[cv.split=='test'].group)))
