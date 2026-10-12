#!/usr/bin/env bash
set -euo pipefail
# Evaluation is owned by the benchmark operator. This wrapper never kills
# processes by name and only terminates the server it started itself.
MRAG_EXPERIMENTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$MRAG_EXPERIMENTS/run.sh" run_rag_build
bash "$MRAG_EXPERIMENTS/run.sh" run_rag_server &
MRAG_OWN_SERVER_PID=$!
trap 'kill -TERM "$MRAG_OWN_SERVER_PID" 2>/dev/null || true; wait "$MRAG_OWN_SERVER_PID" 2>/dev/null || true' EXIT
"${MRAG_PYTHON:-python3}" -I -B "$MRAG_EXPERIMENTS/utils/wait_ready.py"
kill -0 "$MRAG_OWN_SERVER_PID"
bash "$MRAG_EXPERIMENTS/start.sh" unidocbench_rag
