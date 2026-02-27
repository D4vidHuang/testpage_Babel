#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_batch_experiment.sh [workflow_type] [num_samples] [language]
# Defaults: workflow_type=standard, num_samples=5, language=English

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKFLOW_TYPE="${1:-standard}"
NUM_SAMPLES="${2:-5}"
LANGUAGE="${3:-English}"

echo "Running ${WORKFLOW_TYPE} workflow on ${LANGUAGE} data (${NUM_SAMPLES} samples)..."

python run_workflow.py \
  --type "$WORKFLOW_TYPE" \
  --num "$NUM_SAMPLES" \
  --language "$LANGUAGE"
