#!/usr/bin/env zsh
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_remote_activations.sh PROJECT SSH_HOST [REMOTE_DIR]

Example:
  ./run_remote_activations.sh qwen35_27b_full root@1.2.3.4 /tmp/ml-project-activation-run

Environment overrides:
  DEVICE=cuda                 Device passed to extract_activations.py.
  MODEL=MODEL_ID              Optional local activation model override.
  LIMIT=N                     Optional extraction limit.
  THINKING_BOUNDARY=TEXT      Optional thinking boundary override.
  REMOTE_PYTHON=python3       Python executable on the remote host.
  BOOTSTRAP=1                 Create .venv and pip install requirements.txt remotely.
  COPY_ENV=1                  Copy local .env to the remote repo.
  PUSH_ACTIVATIONS=1          Also push local data/activations/PROJECT before running.
  SKIP_PUSH=1                 Do not rsync code/data to remote before running.
  SKIP_PULL=1                 Do not rsync outputs back after running.
  DRY_RUN=1                   Print rsync/ssh operations without executing them.
  SSH_OPTS="-p 1234 -i key"   Extra ssh options.
  RSYNC_OPTS="--progress"     Extra rsync options.

By default the remote worktree lives under /tmp instead of /workspace. On
RunPod, /workspace is often a FUSE/network mount; activation extraction writes
thousands of small tensor files, and the local filesystem path is less fragile.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if (( $# < 2 || $# > 3 )); then
  usage >&2
  exit 2
fi

PROJECT="$1"
REMOTE_HOST="$2"
REMOTE_DIR="${3:-${REMOTE_DIR:-/tmp/ml-project-activation-run}}"

case "$PROJECT" in
  ""|"."|".."|*/*|*\\*)
    echo "PROJECT must be a simple project name, not a path: $PROJECT" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-}"
LIMIT="${LIMIT:-}"
THINKING_BOUNDARY="${THINKING_BOUNDARY:-}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python3}"
BOOTSTRAP="${BOOTSTRAP:-0}"
COPY_ENV="${COPY_ENV:-0}"
PUSH_ACTIVATIONS="${PUSH_ACTIVATIONS:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"
SKIP_PULL="${SKIP_PULL:-0}"
DRY_RUN="${DRY_RUN:-0}"
SSH_OPTS="${SSH_OPTS:-}"
RSYNC_OPTS="${RSYNC_OPTS:-}"

DATASET="data/datasets/${PROJECT}.jsonl"
RUNS="data/runs/${PROJECT}.jsonl"
LABELS="data/labels/${PROJECT}.jsonl"
ACTIVATIONS_DIR="data/activations/${PROJECT}"

if [[ ! -f "$DATASET" ]]; then
  echo "Missing project dataset: $DATASET" >&2
  exit 1
fi
if [[ ! -f "$RUNS" ]]; then
  echo "Missing project runs file: $RUNS" >&2
  exit 1
fi

ssh_args=()
if [[ -n "$SSH_OPTS" ]]; then
  ssh_args=(${=SSH_OPTS})
fi

rsync_args=(-az)
if [[ -n "$RSYNC_OPTS" ]]; then
  rsync_args+=(${=RSYNC_OPTS})
fi
if [[ "$DRY_RUN" == "1" ]]; then
  rsync_args+=(--dry-run)
fi

rsync_ssh="ssh"
if [[ -n "$SSH_OPTS" ]]; then
  rsync_ssh="ssh $SSH_OPTS"
fi

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_rsync() {
  run_cmd rsync "${rsync_args[@]}" -e "$rsync_ssh" "$@"
}

echo "Project: $PROJECT"
echo "Remote:  $REMOTE_HOST:$REMOTE_DIR"
echo "Device:  $DEVICE"

if [[ "$SKIP_PUSH" != "1" ]]; then
  echo
  echo "==> Pushing code"
  run_cmd ssh "${ssh_args[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
  run_rsync \
    --exclude .git/ \
    --exclude .venv/ \
    --exclude __pycache__/ \
    --exclude '*.pyc' \
    --exclude .ipynb_checkpoints/ \
    --exclude data/ \
    --exclude logs/ \
    --exclude .env \
    ./ "$REMOTE_HOST:$REMOTE_DIR/"

  if [[ "$COPY_ENV" == "1" && -f ".env" ]]; then
    echo
    echo "==> Copying .env"
    run_rsync .env "$REMOTE_HOST:$REMOTE_DIR/.env"
  fi

  echo
  echo "==> Pushing project inputs"
  run_cmd ssh "${ssh_args[@]}" "$REMOTE_HOST" \
    "mkdir -p '$REMOTE_DIR/data/datasets' '$REMOTE_DIR/data/runs' '$REMOTE_DIR/data/labels' '$REMOTE_DIR/data/activations/$PROJECT' '$REMOTE_DIR/logs'"
  run_rsync "$DATASET" "$REMOTE_HOST:$REMOTE_DIR/data/datasets/"
  run_rsync "$RUNS" "$REMOTE_HOST:$REMOTE_DIR/data/runs/"
  if [[ -f "$LABELS" ]]; then
    run_rsync "$LABELS" "$REMOTE_HOST:$REMOTE_DIR/data/labels/"
  fi
  if [[ "$PUSH_ACTIVATIONS" == "1" && -d "$ACTIVATIONS_DIR" ]]; then
    run_rsync "$ACTIVATIONS_DIR/" "$REMOTE_HOST:$REMOTE_DIR/$ACTIVATIONS_DIR/"
  fi
fi

echo
echo "==> Running remote activation extraction"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "+ ssh ${SSH_OPTS} ${REMOTE_HOST} bash -s -- ${PROJECT} ${REMOTE_DIR} ..."
else
  remote_env=(
    "PROJECT=$(printf '%q' "$PROJECT")"
    "REMOTE_DIR=$(printf '%q' "$REMOTE_DIR")"
    "DEVICE=$(printf '%q' "$DEVICE")"
    "MODEL=$(printf '%q' "$MODEL")"
    "LIMIT=$(printf '%q' "$LIMIT")"
    "THINKING_BOUNDARY=$(printf '%q' "$THINKING_BOUNDARY")"
    "REMOTE_PYTHON=$(printf '%q' "$REMOTE_PYTHON")"
    "BOOTSTRAP=$(printf '%q' "$BOOTSTRAP")"
  )
  ssh "${ssh_args[@]}" "$REMOTE_HOST" "${remote_env[*]} bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

project="$PROJECT"
remote_dir="$REMOTE_DIR"
device="$DEVICE"
model="${MODEL:-}"
limit="${LIMIT:-}"
thinking_boundary="${THINKING_BOUNDARY:-}"
remote_python="$REMOTE_PYTHON"
bootstrap="$BOOTSTRAP"

cd "$remote_dir"
mkdir -p "logs" "data/activations/$project"

if [[ "$bootstrap" == "1" ]]; then
  if [[ ! -x ".venv/bin/python" ]]; then
    python3 -m venv .venv
  fi
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  remote_python=".venv/bin/python"
fi

stamp="$(date +%Y%m%d_%H%M%S)"
log_path="logs/activation_${project}_${stamp}.log"

cmd=("$remote_python" -u extract_activations.py --project "$project" --device "$device")
if [[ -n "$model" ]]; then
  cmd+=(--model "$model")
fi
if [[ -n "$limit" ]]; then
  cmd+=(--limit "$limit")
fi
if [[ -n "$thinking_boundary" ]]; then
  cmd+=(--thinking-boundary "$thinking_boundary")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "$log_path"

if command -v sha256sum >/dev/null 2>&1; then
  checksum_cmd=(sha256sum)
else
  checksum_cmd=(shasum -a 256)
fi

find "data/activations/$project" -type f -name '*.pt' | sort > "data/activations/$project/.files_for_manifest"
if [[ -s "data/activations/$project/.files_for_manifest" ]]; then
  xargs "${checksum_cmd[@]}" < "data/activations/$project/.files_for_manifest" > "data/activations/$project/MANIFEST.sha256"
else
  : > "data/activations/$project/MANIFEST.sha256"
fi
rm -f "data/activations/$project/.files_for_manifest"

"${checksum_cmd[@]}" \
  "data/datasets/$project.jsonl" \
  "data/runs/$project.jsonl" \
  > "data/activations/$project/RUN_INPUTS.sha256"

echo "Wrote $log_path"
echo "Wrote data/activations/$project/MANIFEST.sha256"
REMOTE_SCRIPT
fi

if [[ "$SKIP_PULL" != "1" ]]; then
  echo
  echo "==> Pulling outputs"
  mkdir -p "data/runs" "$ACTIVATIONS_DIR" "logs"
  run_rsync "$REMOTE_HOST:$REMOTE_DIR/data/runs/${PROJECT}.jsonl" "data/runs/"
  run_rsync "$REMOTE_HOST:$REMOTE_DIR/data/activations/${PROJECT}/" "$ACTIVATIONS_DIR/"
  run_rsync \
    --include "activation_${PROJECT}_*.log" \
    --exclude '*' \
    "$REMOTE_HOST:$REMOTE_DIR/logs/" "logs/"

  if [[ "$DRY_RUN" != "1" && -f "$ACTIVATIONS_DIR/MANIFEST.sha256" ]]; then
    echo
    echo "==> Verifying pulled activation manifest"
    shasum -a 256 -c "$ACTIVATIONS_DIR/MANIFEST.sha256"
  fi
fi

echo
echo "Done."
