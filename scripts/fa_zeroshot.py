"""Zero-shot Persian speech<->text retrieval with EmbeddingGemma 2 (Common Voice 17 fa, FLEURS fa)."""
import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, io, glob, json, time, numpy as np, soundfile as sf, librosa, pyarrow.parquet as pq
sys.path.insert(0, f"{BASE}/scripts")
from common import load_model
from concurrent.futures import ThreadPoolExecutor

D = f"{BASE}/data/fa"
SETS = {  # name: (glob pattern, text column)
    "cv/validation": (f"{D}/cv/data/validation-*.parquet", "sentence"),
    "cv/test": (f"{D}/cv/data/test-*.parquet", "sentence"),
    "fleurs/dev": (f"{D}/fleurs/data/dev-*.parquet", "transcription"),
    "fleurs/test": (f"{D}/fleurs/data/test-*.parquet", "transcription"),
}
TEMPLATES = {"none": None, "Classification": "Classification", "Document": "Document", "SearchQuery": "SearchQuery"}

import re, unicodedata
_MAP = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "آ": "ا", "\u200c": "", "\u200f": "", "\u200e": ""})
def norm(s):
    """Key for deciding two transcriptions are the same sentence (letter variants, ZWNJ, diacritics, punctuation)."""
    s = "".join(c for c in unicodedata.normalize("NFKC", s).translate(_MAP) if not unicodedata.category(c).startswith(("M", "P")))
    return " ".join(s.split())

def decode(b):
    a, sr = sf.read(io.BytesIO(b), dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    return librosa.resample(a, orig_sr=sr, target_sr=16000) if sr != 16000 else a

def load_set(pattern, col):
    t = pq.read_table(sorted(glob.glob(pattern)), columns=["audio", col]) if False else \
        __import__("pyarrow").concat_tables([pq.read_table(f, columns=["audio", col]) for f in sorted(glob.glob(pattern))])
    texts = [" ".join(s.split()) for s in t[col].to_pylist()]
    with ThreadPoolExecutor(32) as ex:
        audio = list(ex.map(decode, [x["bytes"] for x in t["audio"].to_pylist()]))
    return audio, texts

def metrics(A, T, gold, rng, pool=1000, trials=5):
    S = A @ T.T
    rank = (S > S[np.arange(len(A)), gold][:, None]).sum(1)  # 0 = correct sentence ranked first
    out = {"a2t_R@1": float((rank == 0).mean()), "a2t_R@5": float((rank < 5).mean()),
           "a2t_MRR": float((1 / (rank + 1)).mean()), "n_clips": len(A), "n_sentences": len(T)}
    # audio->text in a pool of `pool` sentences (gold + random distractors), comparable across datasets
    if len(T) > pool:
        accs = []
        for _ in range(trials):
            cand = rng.choice(len(T), pool, replace=False)
            sub = S[:, cand]; inpool = np.isin(gold, cand)
            g = np.searchsorted(np.sort(cand), gold); order = np.argsort(cand); sub = sub[:, order]
            ok = inpool & (sub.argmax(1) == g)
            accs.append(ok[inpool].mean())
        out[f"a2t_R@1_pool{pool}"] = float(np.mean(accs))
    # text->audio: is the top clip one that reads this sentence?
    top = S.argmax(0)
    out["t2a_R@1"] = float((gold[top] == np.arange(len(T))).mean())
    return out

if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else "google/embeddinggemma-2"
    tag = sys.argv[2] if len(sys.argv) > 2 else "base"
    m = load_model(mp); rng = np.random.default_rng(0); res = {}
    for name, (pat, col) in SETS.items():
        t0 = time.time(); audio, texts = load_set(pat, col)
        rep = {}  # normalized key -> first original spelling (what the model embeds)
        for s in texts: rep.setdefault(norm(s), s)
        keys = sorted(rep); idx = {k: i for i, k in enumerate(keys)}; uniq = [rep[k] for k in keys]
        gold = np.array([idx[norm(s)] for s in texts])
        dur = np.array([len(a) / 16000 for a in audio])
        print(f"{name}: {len(audio)} clips, {len(uniq)} unique sentences, median {np.median(dur):.1f}s (max {dur.max():.0f}s), decoded in {time.time()-t0:.0f}s", flush=True)
        order = np.argsort(dur); A = np.zeros((len(audio), 768), np.float32)  # length-sorted batches = less padding
        for i in range(0, len(order), 64):
            b = order[i:i+64]; A[b] = m.encode([{"audio": audio[j]} for j in b], batch_size=64, convert_to_numpy=True)
        np.save(f"{BASE}/results/fa_{tag}_{name.replace('/', '_')}_audio.npy", A)
        for k, pn in TEMPLATES.items():
            T = m.encode(uniq, prompt_name=pn, batch_size=128, convert_to_numpy=True)
            r = metrics(A, T, gold, rng); res[f"{name}/{k}"] = r
            print(f"  text prompt={k:15s} " + "  ".join(f"{a}={b:.3f}" for a, b in r.items() if isinstance(b, float)), flush=True)
    json.dump(res, open(f"{BASE}/results/fa_{tag}_zeroshot.json", "w"), indent=1)
