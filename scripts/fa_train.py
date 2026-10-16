"""Contrastive fine-tune of EmbeddingGemma 2 on Persian (audio, transcription) pairs.
Usage: fa_train.py [--max_steps N] [--n_filimo K] [--repo user/name] [--out dir]"""
import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, time, argparse, shutil, zlib, numpy as np, pandas as pd, torch
sys.path.insert(0, f"{BASE}/scripts")
from datasets import Dataset
from transformers import TrainerCallback
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from huggingface_hub import HfApi, hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--max_steps", type=int, default=-1); ap.add_argument("--n_filimo", type=int, default=150000)
ap.add_argument("--bs", type=int, default=256); ap.add_argument("--mini", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--save_steps", type=int, default=400)
ap.add_argument("--repo", default=None); ap.add_argument("--out", default=f"{BASE}/fa_run")
args = ap.parse_args()
D = f"{BASE}/data/fa"; PROMPT = "SearchQuery"

# ---------------- data ----------------
cv, yt = pd.read_parquet(f"{D}/manifest_cv.parquet"), pd.read_parquet(f"{D}/manifest_youtube.parquet")
held_vid = yt.group.map(lambda v: zlib.crc32(v.encode()) % 10 == 0)          # 10% of videos fully held out
yt_eval, yt_train = yt[held_vid], yt[~held_vid]
parts = [cv[cv.split == "train"], yt_train]
fl_eval = pd.DataFrame()
if os.path.exists(f"{D}/manifest_filimo.parquet"):
    fl = pd.read_parquet(f"{D}/manifest_filimo.parquet")
    fl_eval = fl[fl.split == "test"]
    ftr = fl[fl.split == "train"]; parts.append(ftr.sample(min(args.n_filimo, len(ftr)), random_state=0))
train_df = pd.concat(parts).sample(frac=1, random_state=0)
rng = np.random.default_rng(0)
def sub(df, n): return df.iloc[rng.choice(len(df), min(n, len(df)), replace=False)]
VAL = {"cv_val": sub(cv[cv.split == "validation"], 1000), "yt_heldout": sub(yt_eval[yt_eval.split != "test"], 1000)}
if len(fl_eval): VAL["filimo_dev"] = sub(fl[fl.split == "dev"], 1000)
print("train mix:", train_df.ds.value_counts().to_dict(), f"{train_df.dur.sum()/3600:.0f} h;",
      "held-out yt videos:", yt_eval.group.nunique(), flush=True)
train = Dataset.from_dict({"anchor": train_df.path.tolist(), "positive": train_df.text.tolist()})

# ---------------- model ----------------
model = SentenceTransformer("google/embeddinggemma-2", config_kwargs={"vision_config": None})  # fp32 master + bf16 autocast

def r_at_1(df):
    keys = sorted(set(df.key)); rep = dict(zip(df.key, df.text)); idx = {k: i for i, k in enumerate(keys)}
    A = model.encode(df.path.tolist(), batch_size=64, convert_to_tensor=True)
    T = model.encode([rep[k] for k in keys], prompt_name=PROMPT, batch_size=128, convert_to_tensor=True)
    gold = torch.tensor([idx[k] for k in df.key], device=A.device)
    return float(((A @ T.T).argmax(1) == gold).float().mean())

def export(dst):
    """save_pretrained bakes the training tokenizer's padding state into tokenizer.json, which breaks audio
    preprocessing on reload; restore the (untrained) base tokenizer/preprocessor files."""
    model.save_pretrained(dst)
    for f in ["tokenizer.json", "preprocessor_config.json"]:  # copyfile: HF cache blobs are read-only, don't copy perms
        if os.path.exists(f"{dst}/{f}"): os.chmod(f"{dst}/{f}", 0o644)
        shutil.copyfile(hf_hub_download("google/embeddinggemma-2", f), f"{dst}/{f}")

class EvalPush(TrainerCallback):
    def __init__(self): self.hist = []; self.best = -1
    def on_save(self, a, state, control, **kw):
        try: self._eval_push(state)
        except Exception as e: print(f"EVALPUSH_ERROR step {state.global_step}: {e!r}", flush=True); model.train()
    def _eval_push(self, state):
        t0 = time.time(); model.eval()
        with torch.no_grad(): r = {k: r_at_1(v) for k, v in VAL.items()}
        model.train(); mean = float(np.mean(list(r.values())))
        self.hist.append({"step": state.global_step, **r, "mean": mean}); print("VAL", self.hist[-1], f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(self.hist, open(f"{args.out}/val_history.json", "w"), indent=1)
        if mean > self.best:
            self.best = mean; export(f"{args.out}/best")
            if args.repo:  # push best-so-far so nothing is lost if the machine goes away
                HfApi().upload_folder(repo_id=args.repo, folder_path=f"{args.out}/best", commit_message=f"step {state.global_step}: val R@1 {r}", run_as_future=True)

if args.repo: HfApi().create_repo(args.repo, private=False, exist_ok=True)
os.makedirs(args.out, exist_ok=True)
targs = SentenceTransformerTrainingArguments(
    output_dir=args.out, num_train_epochs=1, max_steps=args.max_steps,
    per_device_train_batch_size=args.bs, learning_rate=args.lr, warmup_steps=0.05, lr_scheduler_type="cosine",
    bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES, prompts={"positive": model.prompts[PROMPT]},
    save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
    logging_steps=10, dataloader_num_workers=16, dataloader_prefetch_factor=4, report_to="none", seed=42,
)
trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=train,
                                     loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini),
                                     callbacks=[EvalPush()])
ck = [d for d in os.listdir(args.out) if d.startswith("checkpoint-")] if os.path.isdir(args.out) else []
t0 = time.time(); trainer.train(resume_from_checkpoint=bool(ck) or None); print(f"TRAIN_DONE {time.time()-t0:.0f}s", flush=True)
export(f"{args.out}/final")
