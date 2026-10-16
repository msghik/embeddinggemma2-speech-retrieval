import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys; sys.path.insert(0, f"{BASE}/scripts")
import numpy as np, glob, pyarrow as pa, pyarrow.parquet as pq, io, soundfile as sf
from common import load_model
from fa_zeroshot import D
t = pa.concat_tables([pq.read_table(f, columns=["audio", "sentence"]) for f in sorted(glob.glob(f"{D}/cv/data/test-*.parquet"))])
texts = [" ".join(s.split()) for s in t["sentence"].to_pylist()]
dur = np.array([sf.info(io.BytesIO(x["bytes"])).duration for x in t["audio"].to_pylist()])
uniq = sorted(set(texts)); idx = {s: i for i, s in enumerate(uniq)}; gold = np.array([idx[s] for s in texts])
A = np.load("results/fa_base_cv_test_audio.npy")
m = load_model(); T = m.encode(uniq, prompt_name="SearchQuery", batch_size=128, convert_to_numpy=True)
pred = (A @ T.T).argmax(1); ok = pred == gold
J = lambda a, b: len(set(a.split()) & set(b.split())) / len(set(a.split()) | set(b.split()))
rng = np.random.default_rng(0); wrong = np.where(~ok)[0]
print(f"word overlap (Jaccard) gold vs wrong top-1: {np.mean([J(uniq[gold[i]], uniq[pred[i]]) for i in wrong]):.3f}  | gold vs random sentence: {np.mean([J(uniq[gold[i]], uniq[rng.integers(len(uniq))]) for i in wrong]):.3f}")
nw = np.array([len(s.split()) for s in texts])
for lo, hi in [(0, 3), (3, 5), (5, 7), (7, 30)]:
    s = (dur >= lo) & (dur < hi); print(f"clip {lo}-{hi}s: n={s.sum():5d} R@1={ok[s].mean():.3f}")
for lo, hi in [(1, 4), (4, 7), (7, 11), (11, 99)]:
    s = (nw >= lo) & (nw < hi); print(f"{lo}-{hi-1} words: n={s.sum():5d} R@1={ok[s].mean():.3f}")
print("examples of misses (gold || predicted):")
for i in rng.choice(wrong, 6, replace=False): print("  ", uniq[gold[i]], " || ", uniq[pred[i]])
