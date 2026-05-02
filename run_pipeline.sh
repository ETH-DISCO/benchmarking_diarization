#!/bin/bash
# run_pipeline.sh — orchestrate generation and evaluation across conda environments.
#
# Usage:
#   ./run_pipeline.sh <model> <dataset>
#   ./run_pipeline.sh pyannote callhome
#   ./run_pipeline.sh nemo voxconverse
#
# Models: pyannote, nemo, diarizen, pyannoteai
# Datasets: callhome, voxconverse, ami, ali

set -e

MODEL=$1
DATASET=$2
CONFIG="${3:-config.yaml}"
CONDA_BASE="/itet-stor/ceblaser/net_scratch/conda" #TODO fill in with your actual conda base

if [[ -z "$MODEL" ]] || [[ -z "$DATASET" ]]; then
    echo "Usage: $0 <model> <dataset> [config.yaml]"
    echo ""
    echo "Models:   pyannote, nemo, diarizen, pyannoteai"
    echo "Datasets: callhome, voxconverse, ami, ali"
    exit 1
fi

activate_env() {
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$1"
    echo "Activated environment: $1"
}

deactivate_env() {
    conda deactivate
}

# ─── Map model to conda environment ───
# TODO: verify these environment names match yours
declare -A ENV_MAP
ENV_MAP[pyannote]="pyannote"
ENV_MAP[nemo]="nemo"
ENV_MAP[diarizen]="diarizen"
ENV_MAP[pyannoteai]="pyannoteai"

GEN_ENV="${ENV_MAP[$MODEL]}"

if [[ -z "$GEN_ENV" ]]; then
    echo "ERROR: Unknown model '$MODEL'"
    exit 1
fi

echo "============================================================"
echo "DIARIZATION PIPELINE"
echo "  Model:   $MODEL"
echo "  Dataset: $DATASET"
echo "  Config:  $CONFIG"
echo "============================================================"

# ─── Step 1: Generation (model-specific environment) ───
echo ""
echo ">>> Step 1: Generation ($GEN_ENV)"
activate_env "$GEN_ENV"
python -m src.gen --model "$MODEL" --dataset "$DATASET" --config "$CONFIG"
deactivate_env

# ─── Step 2: Evaluation (pyannote_env, needs pyannote.metrics) ───
echo ""
echo ">>> Step 2: Evaluation (pyannote_env)"
activate_env "pyannote"
python -m src.eval --model "$MODEL" --dataset "$DATASET" --config "$CONFIG"
deactivate_env

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"