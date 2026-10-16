import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import glob, io, numpy as np, pyarrow as pa, pyarrow.parquet as pq, soundfile as sf
from collections import Counter
D = f"{BASE}/data/fa/filimo/data"
print(pq.ParquetFile(sorted(glob.glob(f"{D}/train-*"))[0]).schema_arrow.names)
tx = {}
for sp in ["train", "dev", "test"]:
    fs = sorted(glob.glob(f"{D}/{sp}-*.parquet"))
    tx[sp] = pa.concat_tables([pq.read_table(f, columns=["transcription"]) for f in fs])["transcription"].to_pylist()
    a = pq.read_table(fs[0], columns=["audio"]).slice(0, 300)["audio"].to_pylist()
    d = [sf.info(io.BytesIO(x["bytes"])).duration for x in a]
    print(sp, len(tx[sp]), "clips, unique:", len(set(tx[sp])), "median words:", np.median([len(s.split()) for s in tx[sp]]),
          "| dur median %.1fs p95 %.1fs max %.1fs sr=%d" % (np.median(d), np.percentile(d, 95), max(d), sf.info(io.BytesIO(a[0]["bytes"])).samplerate))
    print("   ", tx[sp][:3])
print("most common train transcripts:", Counter(tx["train"]).most_common(8))
tr = set(tx["train"]); print("test transcripts also in train: %.3f" % np.mean([s in tr for s in tx["test"]]))
