#!/usr/bin/env bash
set -euo pipefail
MRAG_PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$MRAG_PACKAGE_ROOT"
MRAG_INTERPRETER="${MRAG_PYTHON:-python3}"
case "${1:-}" in
  run_rag_build) MRAG_SCRIPT=rag_scripts/rag_build_index_vllm.py ;;
  run_rag_server) MRAG_SCRIPT=rag_scripts/rag_uvicorn_server_vllm.py ;;
  run_rag_unidocbench) MRAG_SCRIPT=bench_scripts/UniDoc-Bench/run_unidocbench.py ;;
  run_rag_eval_qwen_plus) MRAG_SCRIPT=utils/judger.py ;;
  calc_final_score) MRAG_SCRIPT=rag_scripts/calculate_final_score.py ;;
  *) echo 'Usage: run.sh {run_rag_build|run_rag_server|run_rag_unidocbench|run_rag_eval_qwen_plus|calc_final_score}' >&2; exit 2 ;;
esac
shift
exec "$MRAG_INTERPRETER" -I -B "$MRAG_PACKAGE_ROOT/rag-anything-experiments/launch.py" "$MRAG_PACKAGE_ROOT/rag-anything-experiments/$MRAG_SCRIPT" "$@"
