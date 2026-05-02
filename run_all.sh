#!/bin/bash
# run_all.sh — run all model/dataset combinations.
#
# Usage:
#   ./run_all.sh              # all models × all datasets
#   ./run_all.sh config.yaml  # with custom config

set -e

CONFIG="${1:-config.yaml}"

MODELS=("pyannote" "nemo" "diarizen" "pyannoteai")
DATASETS=("callhome" "voxconverse" "ami" "ali")

echo "============================================================"
echo "RUNNING ALL COMBINATIONS"
echo "Models:   ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Config:   $CONFIG"
echo "============================================================"

FAILED=()

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo ""
        echo ">>> $model / $dataset"
        if ./run_pipeline.sh "$model" "$dataset" "$CONFIG"; then
            echo "<<< $model / $dataset DONE"
        else
            echo "<<< $model / $dataset FAILED"
            FAILED+=("$model/$dataset")
        fi
    done
done

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All succeeded."
else
    echo "Failed (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
fi
echo "============================================================"