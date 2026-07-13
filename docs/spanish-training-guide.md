# Training a Spanish OptiSpeech Model

This guide covers everything needed to go from the raw Spanish audio pipeline output to a trained model.

## Prerequisites

Run all commands from the **optispeech repo root** (`/mnt/Projects/Projects/TTS/optispeech`).

```bash
uv sync --extra es
source .venv/bin/activate
```

`babygruut`, `text_processor`, and `gruut-lang-es` are all installed automatically.

> For Swahili too: `uv sync --extra es --extra sw`

---

## Step 1 — Prepare the Raw Data

```bash
python scripts/prepare_es_dataset.py \
    --narration-manifest /mnt/Projects/Projects/Misc/audiobooks-pipeline/outputs/spanish_tts_training/narration/_manifest.jsonl \
    --words-manifest /mnt/Projects/Projects/Misc/audiobooks-pipeline/outputs/spanish_tts_training/words/_manifest.jsonl \
    --output-dir /path/to/es_speaker_data \
    --max-duration 20.0 \
    --val-ratio 0.05 \
    --workers 8
```

Reads both manifests, filters utterances >20s, converts Opus → WAV at 44.1 kHz, and writes a 95/5 train/val split in LJSpeech format (~13,900 files).

---

## Step 2 — Preprocess the Dataset

### 2a — Extract features

```bash
python -m optispeech.tools.preprocess_dataset \
    es_speaker \
    /path/to/es_speaker_data \
    ./data/es_speaker \
    --text-normalization-recipe spanish_vox_cpm
```

Text normalization recipes live in `recipes/text_normalization/`. A recipe is a
Python file that defines `preprocess_text(text: str) -> str`; preprocessing
applies it immediately before `TTSTextProcessor` phonemization.

### 2b — Compute data statistics

```bash
python -m optispeech.tools.generate_data_statistics es_speaker
```

Copy the printed values into `configs/data/es_speaker.yaml` under `data_statistics`.

---

## Step 3 — Train

The configs are already created: [configs/data/es_speaker.yaml](../configs/data/es_speaker.yaml) and [configs/experiment/es_speaker.yaml](../configs/experiment/es_speaker.yaml).

**Fine-tune from an English checkpoint (recommended):**

Set `ckpt_path` in `configs/experiment/es_speaker.yaml` to your best English checkpoint (e.g. `en_multi_althaf`). The `forced_resume: true` flag loads it with `strict=False`, so the Spanish phoneme embedding layer is re-initialised while all acoustic weights (decoder, discriminator, duration/pitch/energy predictors) transfer from English.

```bash
CUDA_VISIBLE_DEVICES=0 python -m optispeech.train experiment=es_speaker
```

**Or pass the checkpoint on the command line:**

```bash
CUDA_VISIBLE_DEVICES=0 python -m optispeech.train experiment=es_speaker \
    ckpt_path=/path/to/en_multi_althaf.ckpt
```

**Resume Spanish training from a checkpoint:**

```bash
CUDA_VISIBLE_DEVICES=0 python -m optispeech.train experiment=es_speaker \
    ckpt_path=/path/to/es_speaker.ckpt \
    forced_resume=false
```

---

## Step 4 — Export to ONNX

```bash
python -m optispeech.onnx.export <checkpoint.ckpt> es_speaker.onnx
python -m optispeech.onnx.infer es_speaker.onnx "Hola, ¿cómo estás?" ./outputs/
```

---

## Future: Turso DB + Emphasis Model for Spanish

When ready:

1. `"es_phonemes"` table is already reserved in `TURSO_TABLES` in `tokenizers.py`.
2. Update `TEXT_PROCESSOR_CONFIG` in `optispeech/text/__init__.py`:
   ```python
   "MODEL_DIRS": {"es": "bookbot/<spanish-emphasis-model-repo>"},
   "DB_PATHS":   {"es": resource_filename('text_processor', 'data/es_word_emphasis_lookup.json')},
   ```
3. Create `text_processor/data/es_word_emphasis_lookup.json`.
