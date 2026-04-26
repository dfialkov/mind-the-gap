#!/usr/bin/env zsh
set -euo pipefail

cd /Users/danielfialkov/Code/ml-project

set -a
[[ -f .env ]] && source .env
set +a

: "${HF_ENDPOINT_URL:?Set HF_ENDPOINT_URL in the environment or .env}"

endpoint="${HF_ENDPOINT_URL%/}"
if [[ "$endpoint" != */v1 ]]; then
  endpoint="$endpoint/v1"
fi
endpoint="$endpoint/"

echo "Starting Qwen3.5 full run at $(date)"
echo "endpoint=${endpoint}"

.venv/bin/python -u generate_answers.py \
  --endpoint-url "$endpoint" \
  --model Qwen/Qwen3.5-27B \
  --hint-types none metadata grader_hacking unethical \
  --dataset data/dataset_full.jsonl \
  --runs-out data/runs_qwen35_27b_full.jsonl \
  --max-tokens 16384 \
  --concurrency 128
