# EmbeddingGemma 2 for speech retrieval

Experiments that use [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), a multimodal embedding model, to match **speech to text**: embed a clip and a set of candidate texts, then pick the closest text.

This is retrieval and matching, **not transcription**. The model ranks texts you provide and can't produce text it hasn't been given.

Three experiments:

1. **English keyword spotting** on Google Speech Commands v0.02 (35 words).
2. **Persian speech ↔ sentence retrieval**, fine-tuned on ~534 h of Persian speech. The model is published at
   **[MohammadGholizadeh/embeddinggemma-2-persian-speech](https://huggingface.co/MohammadGholizadeh/embeddinggemma-2-persian-speech)**.
3. **Persian transcription** with Whisper, since an embedding model can't transcribe: `whisper-large-v3-turbo` fine-tuned on the same data and evaluated on the same test sets. Published at
   **[MohammadGholizadeh/whisper-large-v3-turbo-persian](https://huggingface.co/MohammadGholizadeh/whisper-large-v3-turbo-persian)**.

## Results

### 1. Speech Commands v0.02 (official test split, 11,005 clips, speakers disjoint from train)

| Method | Test accuracy |
|---|---|
| Zero-shot, `Classification` prompt + bare word | 83.7% |
| Zero-shot, `someone saying the word "X"` (chosen on validation) | 93.0% |
| Base model, mean audio embedding per word, 10 clips/word | 94.1% |
| Base model, mean audio embedding per word, all train clips | 95.7% |
| **Fine-tuned (1 epoch contrastive, audio → label text)** | **98.4%** |
| Fine-tuned, mean audio embedding, 1 clip/word | 98.1% |

Zero-shot errors are mostly words that sound alike or share sounds (forward→four, learn→left, no→nine, tree→three, on→one). Fine-tuning mostly fixes them.

### 2. Persian (audio → text R@1 over all distinct sentences in each test set)

| Test set | Candidate sentences | Base | **Fine-tuned** |
|---|---|---|---|
| Common Voice 17 fa test (speakers unseen in training) | 10,457 | 64.0% | **98.5%** |
| YouTube, 82 videos fully held out | 1,326 | 93.9% | **99.5%** |
| Filimo test (5k sample, ~2 s movie dialogue) | 4,856 | 34.4% | **90.7%** |
| FLEURS fa test | 324 | 99.2% | 100% |

On clips whose sentence never appears in training the numbers are essentially the same (98.5 / 99.5 / 92.8%). After the Persian fine-tune, English Speech Commands zero-shot moves only from 93.0% to 91.1%. Full numbers are in `results/*.json` and `results/*.log`.

Evaluation choices:
- **YouTube:** the dataset's official splits share all videos between train and test, so 10% of videos (`crc32(video_id) % 10 == 0`) are held out entirely instead.
- **Sentences:** deduplicated after Persian normalization (ي/ی, ك/ک, ZWNJ, diacritics, punctuation), so spelling variants of the same sentence aren't counted as wrong answers.
- **Prompts:** text prompts are chosen on validation/dev, never on test.

### 3. Persian transcription (Whisper large-v3-turbo, greedy decoding, Persian forced)

WER and CER are computed after the same Persian normalization, with tatweel removed. "CER (no spaces)" also ignores spacing, because Persian writes many words with or without a space (میشیم / می شیم) and plain WER counts those as errors.

| Test set | WER base → **fine-tuned** | CER (no spaces) base → **fine-tuned** |
|---|---|---|
| Common Voice 17 fa test | 50.2% → **19.4%** | 23.6% → **4.2%** |
| YouTube, 82 held-out videos | 45.3% → **20.2%** | 24.4% → **10.0%** |
| Filimo test (5k) | 86.5% → **31.2%** | 50.4% → **16.8%** |

- **Training:** 1 epoch (7,500 steps × 64 clips), lr 1e-5, ~3.9 h on a GH200.
- **Checkpoint choice:** made on a 5k-clip validation sample, not on test. The final checkpoint beat the best intermediate one (`results/wh_select.json`).
- **Runaway outputs** (more than twice the reference length) dropped, e.g. YouTube 2.3% → 0.75%.
- **Per-clip outputs:** every test transcription, base and fine-tuned, is in `results/wh_{base,ft}_*_hyps.tsv`.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
venv/bin/pip install -r requirements.txt
```

Run scripts with `venv/bin/python -I scripts/<name>.py` from the repo root. Data and outputs go under the repo root by default; set `EG2_ROOT=/path` to use a different location. Use bf16 or fp32; the model card warns fp16 produces NaNs.

## Reproducing

### Speech Commands

```bash
mkdir -p data/sc && curl -L http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz | tar -xz -C data/sc
mkdir -p results
python -I scripts/zeroshot.py                                   # zero-shot accuracy per text prompt (val + test)
python -I scripts/analyze.py base                               # confusions + class-centroid baselines
python -I scripts/train.py                                      # contrastive fine-tune, ~30 min on a GH200
python -I scripts/zeroshot.py eg2-speechcommands-ft ft
python -I scripts/analyze.py ft eg2-speechcommands-ft
```

### Persian

```bash
python -I scripts/fa_download.py                                # Common Voice train, YouTube, Filimo parquet from the HF Hub
python -I scripts/fa_prep.py                                    # -> 16 kHz mono WAVs + data/fa/manifest_*.parquet
python -I scripts/fa_eval.py google/embeddinggemma-2 base       # baseline on the three test sets
python -I scripts/fa_train.py --repo <user>/<model>             # ~3 h on a GH200; uploads best checkpoint as it goes
python -I scripts/fa_eval.py fa_run/final ft
```

### Persian Whisper (after `fa_prep.py`)

```bash
export OMP_NUM_THREADS=1                                        # required on aarch64, see Gotchas
python -I scripts/wh_eval.py openai/whisper-large-v3-turbo base # baseline WER/CER on the three test sets
python -I scripts/wh_train.py --max_steps 7500 --bs 64 --grad_ckpt --repo <user>/<model>   # ~3.9 h on a GH200
python -I scripts/wh_select.py wh_run/best wh_run/final         # pick checkpoint on validation
python -I scripts/wh_eval.py wh_run/final ft
```

`fa_zeroshot.py` evaluates on Common Voice validation/test and FLEURS dev/test. It expects the Common Voice validation/test and FLEURS parquet files under `data/fa/cv` and `data/fa/fleurs` (from `MohammadGholizadeh/common_voice_17.0_fa_ttv` and `MohammadGholizadeh/fleurs-farsi`). The remaining `fa_*` scripts are the data checks behind the evaluation choices above: train/test sentence overlap, the Filimo data inspection, and failure analysis.

## Scripts

| Script | Purpose |
|---|---|
| `common.py` | Speech Commands splits, audio loading, model loading |
| `zeroshot.py`, `analyze.py`, `train.py` | Speech Commands eval / analysis / fine-tune |
| `fa_download.py`, `fa_prep.py` | Persian data download and WAV/manifest preparation |
| `fa_train.py` | Persian fine-tune (`CachedMultipleNegativesRankingLoss`, batch 256, no duplicate texts per batch, resumable checkpoints) |
| `fa_eval.py`, `fa_zeroshot.py` | Persian retrieval evaluation |
| `fa_leak_check.py`, `fa_inspect_filimo.py`, `fa_errors.py` | Data overlap checks, dataset inspection, error analysis |
| `wh_common.py` | Whisper test sets, Persian normalization, batched transcription, WER/CER |
| `wh_train.py`, `wh_select.py`, `wh_eval.py`, `wh_rescore.py` | Whisper fine-tune / checkpoint selection on validation / test eval / rescoring saved outputs |
| `smoke.py` | Minimal check that audio encoding works |

## Gotchas

**Saving a fine-tuned EmbeddingGemma 2 audio model.** After training, `save_pretrained` writes the tokenizer's in-training padding state into `tokenizer.json`. On reload the processor passes `max_length=None` to the audio feature extractor, and audio encoding fails with `When setting truncation=True, make sure that max_length is defined`. Text encoding still works, so the bug is easy to miss. The tokenizer isn't trained, so both training scripts restore the base model's `tokenizer.json` and `preprocessor_config.json` after saving. Copy them with `shutil.copyfile`, not `shutil.copy`, because Hugging Face cache files are read-only.

**DataLoader deadlock on aarch64 (Grace/GH200).** Whisper's feature extractor runs a CPU matmul through oneDNN/Arm Compute Library. Its libgomp thread pool deadlocks inside forked DataLoader workers, and training sits at 0% GPU forever. `torch.set_num_threads(1)` alone doesn't help. Launch with `OMP_NUM_THREADS=1`; `wh_train.py` also disables oneDNN inside workers.

**Whisper memory.** Whisper always encodes 30 s of audio (1,500 frames × 32 layers), so batch 32 already runs out of memory on 95 GB without gradient checkpointing. With `--grad_ckpt`, batch 64 uses ~56 GB at ~1.9 s/step.
