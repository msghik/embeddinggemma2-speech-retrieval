import os as _os; BASE = _os.environ.get("EG2_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys, torch
sys.path.insert(0, f"{BASE}/scripts")
from common import *
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

TPL = 'someone saying the word "{}"'
# fp32 master weights + bf16 autocast (pure-bf16 AdamW at lr 2e-5 loses most updates)
model = SentenceTransformer("google/embeddinggemma-2", config_kwargs={"vision_config": None})
files, y = split("train")
train = Dataset.from_dict({"anchor": [f"{ROOT}/{f}" for f in files], "positive": [TPL.format(WORDS[c]) for c in y]})
vf, vy = split("validation")
rng = np.random.default_rng(0); idx = rng.choice(len(vf), 2000, replace=False)
val = Dataset.from_dict({"anchor": [f"{ROOT}/{vf[i]}" for i in idx], "positive": [TPL.format(WORDS[vy[i]]) for i in idx]})

args = SentenceTransformerTrainingArguments(
    output_dir=f"{BASE}/ckpt",
    num_train_epochs=1, per_device_train_batch_size=32, per_device_eval_batch_size=32,
    learning_rate=2e-5, warmup_steps=0.1, bf16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    prompts={"positive": model.prompts["Classification"]},  # text gets the task prefix; audio gets none
    eval_strategy="steps", eval_steps=500, save_strategy="no", logging_steps=100,
    dataloader_num_workers=8, report_to="none", seed=42,
)
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                                     loss=MultipleNegativesRankingLoss(model))
trainer.train()
out = f"{BASE}/eg2-speechcommands-ft"
model.save_pretrained(out)
# Training leaves the tokenizer in padding mode and save_pretrained bakes that into tokenizer.json;
# the processor then forwards max_length=None to the audio feature extractor and audio encoding fails.
# The tokenizer isn't trained, so restore the base copy.
import shutil
from huggingface_hub import hf_hub_download
shutil.copyfile(hf_hub_download("google/embeddinggemma-2", "tokenizer.json"), f"{out}/tokenizer.json")
shutil.copyfile(hf_hub_download("google/embeddinggemma-2", "preprocessor_config.json"), f"{out}/preprocessor_config.json")
