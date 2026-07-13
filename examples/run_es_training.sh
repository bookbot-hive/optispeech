#!/bin/bash
# Full Spanish TTS training pipeline.
# Run inside the optispeech repo root from a tmux session.
set -euo pipefail

REPO=/mnt/Projects/Projects/TTS/optispeech
RAW_DATA=$REPO/data/es_speaker_raw
NARRATION_MANIFEST=/mnt/Projects/Projects/Misc/audiobooks-pipeline/outputs/spanish_tts_training/narration/_manifest.jsonl
WORDS_MANIFEST=/mnt/Projects/Projects/Misc/audiobooks-pipeline/outputs/spanish_tts_training/words/_manifest.jsonl

cd "$REPO"
source .venv/bin/activate

# ── Step 1: Prepare raw dataset ─────────────────────────────────────────────
echo ""
echo "=========================================="
echo " STEP 1: Preparing dataset"
echo "=========================================="
python scripts/prepare_es_dataset.py \
    --narration-manifest "$NARRATION_MANIFEST" \
    --words-manifest "$WORDS_MANIFEST" \
    --output-dir "$RAW_DATA" \
    --max-duration 30.0 \
    --val-ratio 0.05 \
    --workers 16

# ── Step 2: Preprocess features ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo " STEP 2: Preprocessing features"
echo "=========================================="
python -m optispeech.tools.preprocess_dataset \
    es_speaker \
    "$RAW_DATA" \
    ./data/es_speaker \
    --text-normalization-recipe spanish_vox_cpm

# ── Step 3: Compute data statistics ──────────────────────────────────────────
echo ""
echo "=========================================="
echo " STEP 3: Generating data statistics"
echo "=========================================="
python -m optispeech.tools.generate_data_statistics es_speaker

# ── Step 4: Patch config with computed stats (skipped if already done) ────────
echo ""
echo "=========================================="
echo " STEP 4: Updating config with stats"
echo "=========================================="
python - <<'EOF'
import json, re
from pathlib import Path

stats_path = Path("data/es_speaker/stats.json")
cfg_path   = Path("configs/data/es_speaker.yaml")

with open(stats_path) as f:
    stats = json.load(f)

cfg = cfg_path.read_text()
for key, val in stats.items():
    cfg = re.sub(rf"(  {key}:) .*", rf"\1 {val}", cfg)
cfg_path.write_text(cfg)
print("Config updated with:")
for k, v in stats.items():
    print(f"  {k}: {v}")
EOF

# ── Step 5: Train ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " STEP 5: Starting training"
echo "=========================================="
python -m optispeech.train experiment=es_speaker
