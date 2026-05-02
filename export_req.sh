#!/bin/bash
# export_requirements.sh — dump requirements.txt for each model environment.
#
# Usage: ./export_requirements.sh
# Output: requirements/ directory with one file per env

set -e

CONDA_BASE="/itet-stor/ceblaser/net_scratch/conda"
OUTPUT_DIR="requirements"

declare -A ENVS
ENVS[pyannote]="pyannote"
ENVS[nemo]="nemo"
ENVS[diarizen]="diarizen"
ENVS[pyannoteai]="pyannoteai"

source "$CONDA_BASE/etc/profile.d/conda.sh"
mkdir -p "$OUTPUT_DIR"

for key in "${!ENVS[@]}"; do
    env_name="${ENVS[$key]}"
    echo ">>> Exporting $env_name ..."

    conda activate "$env_name"
    pip freeze > "$OUTPUT_DIR/requirements_${key}.txt"
    conda env export --from-history > "$OUTPUT_DIR/environment_${key}.yml"
    conda deactivate

    echo "    requirements_${key}.txt"
    echo "    environment_${key}.yml"
done

echo ""
echo "Done. Files in $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"