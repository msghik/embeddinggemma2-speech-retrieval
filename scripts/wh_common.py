"""Shared pieces for the Whisper Persian fine-tune: test sets, normalization, batched transcription, WER/CER."""
import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, zlib, numpy as np, pandas as pd, soundfile as sf, torch, jiwer
sys.path.insert(0, f"{BASE}/scripts")
from fa_zeroshot import norm
from concurrent.futures import ThreadPoolExecutor

D = f"{BASE}/data/fa"
BASE_MODEL = "openai/whisper-large-v3-turbo"

def manifests():
    return tuple(pd.read_parquet(f"{D}/manifest_{d}.parquet") for d in ["cv", "youtube", "filimo"])

def held_out(yt):  # same 10%-of-videos rule as fa_train.py
    return yt.group.map(lambda v: zlib.crc32(v.encode()) % 10 == 0)

def test_sets():
    cv, yt, fl = manifests()
    return {"cv_test": cv[cv.split == "test"],
            "youtube_heldout_videos": yt[held_out(yt) & (yt.split == "test")],
            "filimo_test_5k": fl[fl.split == "test"].sample(5000, random_state=0)}

def train_df():
    cv, yt, fl = manifests()
    return pd.concat([cv[cv.split == "train"], yt[~held_out(yt)], fl[fl.split == "train"]]).sample(frac=1, random_state=0)

def load(p):
    a, _ = sf.read(p, dtype="float32"); return a

@torch.no_grad()
def transcribe(model, processor, paths, bs=64):
    """Greedy decoding, Persian forced; returns hypotheses in input order (length-sorted batches for speed)."""
    out = [None] * len(paths); order = np.argsort([-sf.info(p).frames for p in paths])
    dev, dt = model.device, model.dtype
    with ThreadPoolExecutor(16) as ex:
        for i in range(0, len(order), bs):
            idx = order[i:i+bs]; audio = list(ex.map(load, [paths[j] for j in idx]))
            feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(dev, dt)
            ids = model.generate(feats, language="fa", task="transcribe", max_new_tokens=225, max_length=None)
            for j, t in zip(idx, processor.batch_decode(ids, skip_special_tokens=True)): out[j] = t
    return out

def clean(s):  # training target / scoring text: drop tatweel speaker-turn marks (ـ), collapse spaces
    return " ".join(s.replace("\u0640", " ").split())

def scores(refs, hyps):
    """WER/CER after Persian normalization (letter variants, ZWNJ, diacritics, punctuation removed)."""
    R, H = [clean(norm(r)) for r in refs], [clean(norm(h)) for h in hyps]
    keep = [i for i, r in enumerate(R) if r]
    R, H = [R[i] for i in keep], [H[i] for i in keep]
    # spacing between Persian morphemes varies (میشیم / می شیم); CER without spaces measures content regardless
    return {"WER": jiwer.wer(R, H), "CER": jiwer.cer(R, H),
            "CER_nospace": jiwer.cer([r.replace(" ", "") for r in R], [h.replace(" ", "") or "-" for h in H]), "n": len(R)}

def load_whisper(path=BASE_MODEL):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    m = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    return m, WhisperProcessor.from_pretrained(path)
