#!/usr/bin/env bash
set -euo pipefail
MRAG_EXPERIMENTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  unidocbench_rag)
    bash "$MRAG_EXPERIMENTS/run.sh" run_rag_unidocbench
    bash "$MRAG_EXPERIMENTS/run.sh" run_rag_eval_qwen_plus
    if [[ -f "${DOTENV_FILE:-.env}" ]] && grep -Eq '^BASELINE_SEARCH_TIME=[^[:space:]]+' "${DOTENV_FILE:-.env}" && grep -Eq '^BASELINE_BUILD_TIME=[^[:space:]]+' "${DOTENV_FILE:-.env}"; then
      bash "$MRAG_EXPERIMENTS/run.sh" calc_final_score
    fi
    ;;
  unidocbench_eval) bash "$MRAG_EXPERIMENTS/run.sh" run_rag_eval_qwen_plus ;;
  *) echo 'Usage: start.sh {unidocbench_rag|unidocbench_eval}' >&2; exit 2 ;;
esac
