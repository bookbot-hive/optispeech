# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OptiSpeech is a lightweight end-to-end TTS model for on-device deployment. It has two packages:
- **`optispeech/`** — Training package (PyTorch + Lightning + Hydra)
- **`ospeech/`** — Inference-only library using ONNX (no PyTorch required)

## Setup

```bash
# Base install (English only)
uv sync

# With additional language support
uv sync --extra es   # Spanish
uv sync --extra sw   # Swahili
uv sync --extra es --extra sw  # both

source .venv/bin/activate
```

> On machines with conda active, the conda `CC`/`CXX` compilers break building `pesq` and `pyworld`. `uv.toml` pins these to `/usr/bin/gcc` and `/usr/bin/g++` so `uv sync` works without manual env overrides.

The `TextProcessor` (Bookbot's custom G2P) requires env vars `COSMOS_DB_URL` and `COSMOS_DB_KEY` (see `.env.example`).

## Commands

### Data Preprocessing
```bash
# Preprocess dataset: <recipe> matches a name under configs/data/ or configs/experiment/
python -m optispeech.tools.preprocess_dataset <recipe> <raw_data_dir> ./data/<recipe>

# Compute feature statistics (mean/std for pitch, energy, mel) — must run after preprocessing
python -m optispeech.tools.generate_data_statistics <recipe>
```

### Training
```bash
# Train with an experiment config (configs/experiment/<name>.yaml)
python -m optispeech.train experiment=<name>

# Resume from checkpoint
python -m optispeech.train experiment=<name> ckpt_path=<path>

# Override any config value via CLI
python -m optispeech.train experiment=<name> trainer.max_epochs=100
```

### Inference
```bash
# PyTorch inference from checkpoint
python -m optispeech.infer checkpoint.ckpt "Text to speak" output_dir/ [--d-factor 1.0] [--p-factor 1.0] [--e-factor 1.0] [--cuda]

# Export to ONNX
python -m optispeech.onnx.export checkpoint.ckpt output.onnx

# ONNX inference
python -m optispeech.onnx.infer model.onnx "Text to speak" output_dir/
```

### ospeech (inference-only package)
```bash
cd ospeech && pip install -e .
ospeech model.onnx "Text" output_dir/
ospeech-models ls            # List available HuggingFace models
ospeech-models dl <id> .     # Download model
ospeech-gradio model.onnx    # Launch Gradio web UI
```

### Tests & Linting
```bash
pytest                        # Run tests (path: tests/)
pytest -m "not slow"          # Skip slow tests
pre-commit run --all-files    # Run Black, isort, Flake8, Pylint
```

## Architecture

### Configuration System (Hydra)
All config is under `configs/` and composed at runtime. Key directories:
- `configs/experiment/` — Per-experiment overrides (the primary way to configure a run)
- `configs/data/` — Dataset configs; `configs/data/text_processor/` — tokenizer selection
- `configs/model/` — Model architecture choices (optispeech, convnext_tts, lightspeech, conformer)
- `configs/trainer/` — Lightning trainer settings

The `run_name` field is required (`???`) — it must be set in the experiment config.

### Text Processing Pipeline
`optispeech/text/__init__.py` defines `TextProcessor`, which wraps Bookbot's internal `text_processor` package for English (RoBERTa-based emphasis + homograph handling). For other languages it falls back to:
- `GruutTokenizer` — IPA phonemes via gruut (supports en, de, fr, etc.)
- `GruutSwahiliTokenizer` — Swahili-specific gruut
- `G2pIdTokenizer` — Indonesian (g2p_id)
- `IPATokenizer` — espeak-based IPA via piper-phonemize

Tokenizer selection is configured in `configs/data/text_processor/<name>.yaml`.

### Model Architecture
The model is GAN-based with **manual Lightning optimization** (no automatic_optimization):
- **Generator** (`optispeech/model/generator/`) — Encoder → duration/pitch/energy predictors → decoder → waveform. Multiple backbone choices: ConvNeXt (fastest), Transformer (default), Conformer, LightSpeech.
- **Discriminator** (`optispeech/model/discriminator/`) — Adversarial training component.
- `InferenceInputs` / `InferenceOutputs` in `optispeech/values.py` are the main data structures for inference calls.

### Training Data Flow
```
Raw audio + metadata
  → preprocess_dataset  (extracts mel / pitch / energy as .npy files)
  → generate_data_statistics  (computes mean/std, saved alongside data)
  → TextWavDataModule (optispeech/dataset/)
  → Generator + Discriminator training loop
```

## Key Conventions

- **Code style**: Black (line length 120), isort with black profile.
- **Adding a new language recipe**: Create `configs/experiment/<name>.yaml`, `configs/data/<name>.yaml`, and a matching `configs/data/text_processor/<lang>.yaml`. Register any new tokenizer as a `BaseTokenizer` subclass in `optispeech/text/tokenizers.py` (auto-registered via `__init_subclass__`).
- **Adding a new model backbone**: Implement under `optispeech/model/generator/` and add a corresponding config under `configs/model/`.
