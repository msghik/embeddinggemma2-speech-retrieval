import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, numpy as np, soundfile as sf, torch
ROOT = f"{BASE}/data/sc"
WORDS = sorted(d for d in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{d}") and not d.startswith("_"))

def split(name):
    """Official v0.02 splits: test/validation lists; train = everything else."""
    if name in ("test", "validation"):
        files = open(f"{ROOT}/{ {'test': 'testing'}.get(name, name)}_list.txt").read().split()
    else:
        held = set(open(f"{ROOT}/testing_list.txt").read().split()) | set(open(f"{ROOT}/validation_list.txt").read().split())
        files = sorted(f"{w}/{f}" for w in WORDS for f in os.listdir(f"{ROOT}/{w}") if f"{w}/{f}" not in held)
    return files, np.array([WORDS.index(f.split("/")[0]) for f in files])

def load(f):
    a, sr = sf.read(f"{ROOT}/{f}", dtype="float32"); assert sr == 16000
    return a

def load_model(path="google/embeddinggemma-2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, model_kwargs={"torch_dtype": torch.bfloat16}, config_kwargs={"vision_config": None})

def embed_audio(model, files, bs=128):
    out = []
    for i in range(0, len(files), bs):
        out.append(model.encode([{"audio": load(f)} for f in files[i:i+bs]], batch_size=bs, convert_to_numpy=True))
    return np.concatenate(out)
