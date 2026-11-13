#!/usr/bin/env bash
# Run the reproduction. Fill in config.env and run setup.sh first.
#
#   bash run.sh all                 SFT -> TSR iter0 -> TSR iter1 -> SR iter1 -> w/o Future iter1 -> AlpacaEval
#   bash run.sh sft | iter0 | iter1 | sr | wof
#   bash run.sh iter1 <variant>     one iter1 variant from configs/variants.sh (e.g. rej_past, strict, mf1_from_mf0)
#   bash run.sh iter1               best current/future, ties current; positive gap; saves M2_current_tie
#   bash run.sh iter1 future_only   explicit historical future-only variant
#   bash run.sh iter1 main          explicitly select the historical current/future comparison
#   bash run.sh iter1_mix | iter0_strict | iter0_conf
#   bash run.sh eval [MODEL ...]    AlpacaEval 2.0 for the given models (default: all main models that exist)
#   bash run.sh status              what has been produced so far
#
# SMOKE=1 bash run.sh all   runs the whole pipeline on 16 prompts with 2 optimizer steps (plumbing check, ~1 h).
# Every step is skipped if its output exists, so an interrupted run can be resumed with the same command.
set -euo pipefail
cd "$(dirname "$0")"
. scripts/lib.sh
. scripts/stages.sh
load_config
[ -x "$PY" ] && [ -d "$TSR_DIR" ] || die "environment not set up: run  bash setup.sh"

cmd=${1:-}; shift || true
case "$cmd" in
  all)        stage_sft; stage_iter0; stage_iter1; stage_sr; stage_wof; stage_eval ;;
  sft)        stage_sft ;;
  iter0)      stage_iter0 ;;
  iter1)      stage_iter1 "${1:-${TSR_PAIR_SELECTION:-current_tie}}" ;;
  sr)         stage_sr ;;
  wof)        stage_wof ;;
  iter1_mix)  stage_iter1_mix ;;
  iter0_strict) stage_iter0_strict ;;
  iter0_conf) stage_iter0_conf ;;
  eval)       stage_eval "$@" ;;
  status)     ls "$MODELS" 2>/dev/null | sed 's/^/model  /'; [ -f "$EVAL/summary.csv" ] && cat "$EVAL/summary.csv" || true ;;
  *)          sed -n '2,13p' "$0"; exit 1 ;;
esac
log "done: $cmd $*"
