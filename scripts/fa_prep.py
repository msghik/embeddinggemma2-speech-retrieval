"""Decode Persian ASR parquet shards to 16 kHz mono int16 WAVs + a manifest (path, text, key, dur, group, split, ds)."""
import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, io, glob, hashlib, numpy as np, pyarrow.parquet as pq, soundfile as sf, librosa, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, f"{BASE}/scripts")
from fa_zeroshot import norm, _MAP

D = f"{BASE}/data/fa"; W = f"{D}/wav"
SRC = {  # ds: (shard glob, text col, group col or None)
    "cv":      (f"{D}/cv/data/*.parquet", "sentence", "client_id"),
    "youtube": (f"{D}/youtube/data/*.parquet", "transcription", "video_id"),
    "filimo":  (f"{D}/filimo/data/*.parquet", "transcription", None),
}

_LIGHT = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})
def light(s):  # what the model sees: unify letter variants + whitespace, keep punctuation/ZWNJ
    return " ".join(s.translate(_LIGHT).split())

def do_shard(args):
    ds, f = args
    _, tcol, gcol = SRC[ds]; split = os.path.basename(f).split("-")[0]
    cols = ["audio", tcol] + ([gcol] if gcol else [])
    t = pq.read_table(f, columns=cols).to_pylist(); rows = []
    os.makedirs(f"{W}/{ds}", exist_ok=True)
    for i, r in enumerate(t):
        text = (r[tcol] or "").strip()
        if not norm(text): continue
        try:
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        except Exception:
            continue
        if a.ndim > 1: a = a.mean(1)
        if sr != 16000: a = librosa.resample(a, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
        dur = len(a) / 16000
        if not 0.5 <= dur <= 20: continue
        uid = hashlib.md5(f"{f}:{i}".encode()).hexdigest()[:16]
        p = f"{W}/{ds}/{uid}.wav"; sf.write(p, np.clip(a, -1, 1), 16000, subtype="PCM_16")
        rows.append(dict(path=p, text=light(text), key=norm(text), dur=dur, group=(r[gcol] if gcol else ""), split=split, ds=ds))
    return rows

if __name__ == "__main__":
    dss = sys.argv[1:] or list(SRC)
    for ds in dss:
        jobs = [(ds, f) for f in sorted(glob.glob(SRC[ds][0]))]
        with Pool(64) as p: rows = [r for rs in p.imap_unordered(do_shard, jobs) for r in rs]
        df = pd.DataFrame(rows); df.to_parquet(f"{D}/manifest_{ds}.parquet")
        print(ds, len(df), "clips kept;", df.groupby("split").size().to_dict(), f"{df.dur.sum()/3600:.0f} h", flush=True)
