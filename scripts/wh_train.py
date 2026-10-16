"""Fine-tune whisper-large-v3-turbo on Persian (audio, transcription) pairs.
Usage: wh_train.py [--max_steps N] [--bs B] [--accum A] [--repo user/name] [--out dir] [--bench]"""
import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, os, json, time, argparse, numpy as np, torch
sys.path.insert(0, f"{BASE}/scripts")
from wh_common import *
from transformers import (WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, TrainerCallback)
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--max_steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--save_steps", type=int, default=500)
ap.add_argument("--repo", default=None); ap.add_argument("--out", default=f"{BASE}/wh_run")
ap.add_argument("--grad_ckpt", action="store_true"); ap.add_argument("--accum", type=int, default=1)
args = ap.parse_args()

proc = WhisperProcessor.from_pretrained(BASE_MODEL)
proc.tokenizer.set_prefix_tokens(language="persian", task="transcribe", predict_timestamps=False)
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, attn_implementation="sdpa")
model.generation_config.language, model.generation_config.task = "fa", "transcribe"
model.generation_config.forced_decoder_ids = None; model.config.forced_decoder_ids = None
if args.grad_ckpt: model.config.use_cache = False

tr = train_df(); print("train mix:", tr.ds.value_counts().to_dict(), f"{tr.dur.sum()/3600:.0f} h;",
                       f"{args.max_steps} steps x {args.bs}x{args.accum} = {args.max_steps*args.bs*args.accum/len(tr):.2f} epochs", flush=True)

class DS(torch.utils.data.Dataset):
    def __init__(self, df): self.p, self.t = df.path.tolist(), [clean(t) for t in df.text]
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return {"path": self.p[i], "text": self.t[i]}

SOT = model.config.decoder_start_token_id
def collate(batch):
    # The feature extractor's CPU matmul goes through oneDNN/ACL, whose libgomp thread pool deadlocks in forked
    # DataLoader workers on aarch64 (Grace). Launch with OMP_NUM_THREADS=1 and keep oneDNN off in workers.
    torch.backends.mkldnn.enabled = False; torch.set_num_threads(1)
    feats = proc.feature_extractor([load(b["path"]) for b in batch], sampling_rate=16000, return_tensors="pt").input_features
    lab = proc.tokenizer([b["text"] for b in batch], max_length=448, truncation=True, padding=True, return_tensors="pt")
    labels = lab.input_ids.masked_fill(lab.attention_mask.ne(1), -100)
    if (labels[:, 0] == SOT).all(): labels = labels[:, 1:]   # the model prepends <|startoftranscript|> itself
    return {"input_features": feats, "labels": labels}

# small validation sets for checkpoint selection (never the test sets)
cv, yt, fl = manifests()
VAL = {"cv_val": cv[cv.split == "validation"].sample(200, random_state=0),
       "yt_heldout_val": yt[held_out(yt) & (yt.split != "test")].sample(200, random_state=0),
       "filimo_dev": fl[fl.split == "dev"].sample(200, random_state=0)}

def export(dst):
    model.save_pretrained(dst); proc.save_pretrained(dst); model.generation_config.save_pretrained(dst)

class EvalPush(TrainerCallback):
    def __init__(self): self.hist, self.best = [], 9e9
    def on_save(self, a, state, control, **kw):
        try: self._run(state)
        except Exception as e: print(f"EVALPUSH_ERROR step {state.global_step}: {e!r}", flush=True)
        finally: model.train()
    def _run(self, state):
        t0 = time.time(); model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = {k: scores(v.text.tolist(), transcribe(model, proc, v.path.tolist()))["WER"] for k, v in VAL.items()}
        mean = float(np.mean(list(r.values()))); self.hist.append({"step": state.global_step, **r, "mean_WER": mean})
        print("VAL", self.hist[-1], f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(self.hist, open(f"{args.out}/val_history.json", "w"), indent=1)
        if mean < self.best:
            self.best = mean; export(f"{args.out}/best")
            if args.repo:
                HfApi().upload_folder(repo_id=args.repo, folder_path=f"{args.out}/best", run_as_future=True,
                                      commit_message=f"step {state.global_step}: val WER {r}")

if args.repo: HfApi().create_repo(args.repo, private=False, exist_ok=True)
os.makedirs(args.out, exist_ok=True)
targs = Seq2SeqTrainingArguments(
    output_dir=args.out, max_steps=args.max_steps, per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
    learning_rate=args.lr, warmup_steps=min(500, args.max_steps // 10), lr_scheduler_type="linear", weight_decay=0.0,
    bf16=True, gradient_checkpointing=args.grad_ckpt, save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
    logging_steps=25, dataloader_num_workers=16, dataloader_prefetch_factor=4, remove_unused_columns=False,
    report_to="none", seed=42,
)
trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=DS(tr), data_collator=collate, callbacks=[EvalPush()])
ck = [d for d in os.listdir(args.out) if d.startswith("checkpoint-")]
t0 = time.time(); trainer.train(resume_from_checkpoint=bool(ck) or None); print(f"TRAIN_DONE {time.time()-t0:.0f}s", flush=True)
export(f"{args.out}/final")
