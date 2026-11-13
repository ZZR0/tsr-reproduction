# Shared helpers, sourced by setup.sh and run.sh.
# Every step writes its output once and is skipped if the output already exists, so an interrupted run
# can simply be restarted with the same command.

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/TemporalSelfRewarding/TSR.git}  # override to clone through a mirror
UPSTREAM_COMMIT=020266aa9ef3cc35bf81384c9b59f98cff62aa1c

load_config() {
  [ -f "$REPO/config.env" ] || { echo "config.env not found. Run: cp config.env.example config.env  and fill it in." >&2; exit 1; }
  set -a; . "$REPO/config.env"; set +a
  : "${GPUS:?set GPUS in config.env}"
  WORK_DIR=$(cd "$REPO" && realpath -m "${WORK_DIR:-./work}")
  if [ -d "${BASE_MODEL:-}" ]; then BASE=$(realpath "$BASE_MODEL"); else BASE=$WORK_DIR/base/Llama-3.1-8B; fi
  export HF_HOME=${HF_HOME:-$WORK_DIR/hf_cache}
  SMOKE=${SMOKE:-0}
  LIMIT=0; MAX_STEPS=-1; ALPACA_LIMIT=0
  if [ "$SMOKE" = 1 ]; then  # plumbing test: few prompts, 2 optimizer steps, separate work dir
    LIMIT=${SMOKE_PROMPTS:-16}; MAX_STEPS=2; ALPACA_LIMIT=8; WORK_DIR=$WORK_DIR/smoke
  fi
  TSR_DIR=$REPO/third_party/TSR
  PY=$REPO/.venv/bin/python
  IFS=, read -ra GPU_LIST <<< "$GPUS"; NGPU=${#GPU_LIST[@]}
  MODELS=$WORK_DIR/models; DATA=$WORK_DIR/data; LOGS=$WORK_DIR/logs; EVAL=$WORK_DIR/alpaca_eval
  mkdir -p "$MODELS" "$DATA" "$LOGS" "$EVAL"
  M0=$MODELS/M0
  export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACCELERATE_LOG_LEVEL=info TSR_DIR
  [ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
  [ -n "${HF_ENDPOINT:-}" ] && export HF_ENDPOINT
  return 0
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/run.log" >&2; }
die() { log "ERROR: $*"; exit 1; }
need() { for f in "$@"; do [ -e "$f" ] || die "missing input $f (run the earlier stage first)"; done; }
expected_rows() { [ "$LIMIT" -gt 0 ] && echo "$LIMIT" || echo 5000; }

# gen MODEL PROMPTS.jsonl OUT.json
# Official TSR generation code (K = 7 responses per prompt), one data shard per GPU.
gen() {
  local model=$1 prompts=$2 out=$3 name; name=$(basename "$out" .json)
  [ -f "$out" ] && { log "skip gen $name (exists)"; return 0; }
  need "$model/config.json" "$prompts"
  log "gen $name  <- $(basename "$model")"
  _shard gen "$model" "$prompts" "$out"
}

# judge MODEL RESPONSES.json OUT.json
# Official TSR judging code (1 judgement per response), then a health check on parse failures.
judge() {
  local model=$1 inp=$2 out=$3 name; name=$(basename "$out" .json)
  [ -f "$out" ] && { log "skip judge $name (exists)"; return 0; }
  need "$model/config.json" "$inp"
  log "judge $name  <- $(basename "$model")"
  _shard judge "$model" "$inp" "$out"
  "$PY" "$REPO/src/judge_stats.py" "$out" > "${out%.json}.stats.json"
  [ "$SMOKE" = 1 ] && return 0  # a 2-step smoke model writes few parsable scores
  # Unparsable judgements get score 0 in the official code; we saw 7.5% (iter0) and <0.1% (iter1).
  "$PY" - "${out%.json}.stats.json" <<'EOF' || die "judge health check failed for $out"
import json, sys
s = json.load(open(sys.argv[1]))
sys.exit(0 if s["parse_fail_rate"] <= 0.15 else 1)
EOF
}

_shard() {
  local mode=$1 model=$2 inp=$3 out=$4 name i=0 pids=() parts=()
  name=$(basename "$out" .json)
  for g in "${GPU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES=$g "$PY" "$REPO/src/tsr_shard.py" "$mode" --model "$model" --input "$inp" \
      --out "$out.part$i" --shard $i --nshards "$NGPU" --limit "$LIMIT" ${VLLM_SEED:+--vllm_seed $VLLM_SEED} \
      > "$LOGS/$name.part$i.log" 2>&1 &
    pids+=($!); parts+=("$out.part$i"); i=$((i + 1))
  done
  for p in "${pids[@]}"; do wait "$p" || die "$mode $name failed, see $LOGS/$name.part*.log"; done
  "$PY" "$REPO/src/tsr_shard.py" merge --out "$out.tmp" "${parts[@]}" >> "$LOGS/run.log" || die "merge $name"
  local n; n=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$out.tmp")
  [ "$n" = "$(expected_rows)" ] || die "$name has $n rows, expected $(expected_rows)"
  mv "$out.tmp" "$out"; rm -f "${parts[@]}"
}

# pairs past|future OUT_DIR CHOSEN_J REJECTED_J [FUTURE_J] [-- build_pairs flags]
# Builds DPO pairs with src/build_pairs.py. Without extra flags it also runs the official upstream script
# (settings/tsr/reformat2DPO_{past,future}.py) and requires both outputs to be identical.
pairs() {
  local mode=$1 out=$2 c=$3 r=$4 f=""; shift 4
  if [ "$mode" = future ]; then f=$1; shift; fi
  [ "${1:-}" = "--" ] && shift
  local strict_gap=0 flag
  for flag in "$@"; do
    case "$flag" in --filter|--future-only) strict_gap=1 ;; esac
  done
  if [ -f "$out/train.json" ]; then
    if [ "$strict_gap" = 1 ]; then
      # Do not relabel an old unfiltered dataset or reuse models trained on it.
      validate_filtered_pairs "$out" "$@" || die "cached pairs are unfiltered or empty: $out; use a fresh WORK_DIR/run (old models must not be reused)"
    fi
    log "skip pairs $(basename "$out") (exists)"; return 0
  fi
  need "$c" "$r" ${f:+"$f"}
  local check=()
  if [ $# -eq 0 ]; then
    # datasets recursively discovers train.json; keep comparison output outside the dataset directory.
    local official="${out%/}.official.json"
    mkdir -p "$(dirname "$official")"
    if [ "$mode" = past ]; then
      "$PY" "$TSR_DIR/settings/tsr/reformat2DPO_past.py" --input_path_chosen "$c" --input_path_rejected "$r" \
        --output_path "$official" >> "$LOGS/run.log" 2>&1 || die "official past script"
    else
      "$PY" "$TSR_DIR/settings/tsr/reformat2DPO_future.py" --input_path_chosen "$c" --input_path_future "$f" \
        --input_path_rejected "$r" --output_path "$official" >> "$LOGS/run.log" 2>&1 || die "official future script"
    fi
    check=(--check "$official")
  fi
  "$PY" "$REPO/src/build_pairs.py" "$mode" --chosen "$c" --rejected "$r" ${f:+--future "$f"} --out "$out.tmp" \
    "${check[@]}" "$@" > "$LOGS/pairs_$(basename "$out").log" 2>&1 || die "build_pairs $(basename "$out")"
  if [ ${#check[@]} -gt 0 ]; then
    grep -q '"matches_official": true' "$out.tmp/stats.json" || die "pairs $(basename "$out") differ from the official script"
  fi
  if [ "$strict_gap" = 1 ]; then
    validate_filtered_pairs "$out.tmp" "$@" || die "filtered pairs are empty or invalid: $out.tmp; refusing DPO training"
  fi
  mkdir -p "$out"; mv "$out.tmp"/* "$out"/; rmdir "$out.tmp"
  log "pairs $(basename "$out"): $(tr -d '\n ' < "$out/stats.json")"
}

# A strict-gap dataset must have been built with filtering and retain at least one pair.
validate_filtered_pairs() {
  "$PY" - "$@" <<'EOF'
import argparse, json, math, sys
from pathlib import Path
try:
    root = Path(sys.argv[1])
    stats = json.loads((root / "stats.json").read_text())
    flags = argparse.ArgumentParser(add_help=False)
    flags.add_argument("--rej_tie", choices=["mi", "past"], default="mi")
    expected, _ = flags.parse_known_args(sys.argv[2:])
    if stats.get("rejected_tie_break") != ("current" if expected.rej_tie == "mi" else "past"):
        raise ValueError("Cached rejected tie-breaking rule differs")
    if "--strict" in sys.argv[2:] and stats.get("chosen_tie_break") != "current":
        raise ValueError("Cached chosen current/future tie-breaking rule differs")
    gap = stats.get("gap_min")
    if (stats.get("strict_score_gap") is not True or stats.get("retained_tie_pairs") != 0
            or not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap <= 0):
        raise ValueError("Filtering/positive gap evidence missing")
    count = sum(1 for line in (root / "train.json").open() if line.strip())
    if count <= 0 or count != stats.get("n_pairs"):
        raise ValueError("Empty dataset or pair count mismatch")
except (OSError, ValueError, TypeError) as exc:
    print(f"Invalid filtered pairs: {exc}", file=sys.stderr)
    sys.exit(1)
EOF
}

# dpo NAME INIT_MODEL PAIRS_DIR  -> $MODELS/NAME
# Upstream config (settings/sr/config/dpo/config_full_iter0.yaml, identical to the TSR ones except paths),
# upstream ZeRO-3 accelerate config, patched upstream run_dpo.py. Global batch 32 on any number of GPUs.
dpo() {
  local name=$1 init=$2 data=$3
  [ -f "$MODELS/$name/.done" ] && { log "skip dpo $name (exists)"; return 0; }
  need "$init/config.json" "$data/train.json"
  local cfg=$WORK_DIR/configs/$name.yaml
  "$PY" "$REPO/src/make_cfg.py" dpo --name "$name" --init "$init" --data "$data" --out_dir "$MODELS/$name" \
    --ngpu "$NGPU" --max_steps "$MAX_STEPS" --cfg "$cfg" > /dev/null || die "config $name"
  log "dpo $name  init=$(basename "$init") data=$(basename "$data")  ($NGPU GPUs)"
  _train "$cfg" "$TSR_DIR/alignment-handbook/scripts/run_dpo.py" "$LOGS/dpo_$name.log" || die "dpo $name, see $LOGS/dpo_$name.log"
  [ -f "$MODELS/$name/config.json" ] || die "dpo $name produced no model"
  touch "$MODELS/$name/.done"
}

# sft  -> $M0
sft() {
  [ -f "$M0/.done" ] && { log "skip sft (exists)"; return 0; }
  need "$BASE/config.json"
  local cfg=$WORK_DIR/configs/M0.yaml
  "$PY" "$REPO/src/make_cfg.py" sft --name M0 --init "$BASE" --data "$TSR_DIR/data/sft_data" --out_dir "$M0" \
    --ngpu "$NGPU" --max_steps "$MAX_STEPS" --cfg "$cfg" > /dev/null || die "config M0"
  log "sft M0  ($NGPU GPUs)"
  _train "$cfg" "$TSR_DIR/alignment-handbook/scripts/run_sft.py" "$LOGS/sft_M0.log" || die "sft, see $LOGS/sft_M0.log"
  [ -f "$M0/config.json" ] || die "sft produced no model"
  touch "$M0/.done"
}

_train() {
  local cfg=$1 script=$2 logf=$3
  local entry=("$script")
  if [ -n "${TSR_RANK_CACHE_ROOT:-}" ]; then
    entry=("$REPO/src/rank_training.py" "$script")
  fi
  local train_result=0
  ( cd "$TSR_DIR/alignment-handbook" && CUDA_VISIBLE_DEVICES=$GPUS "$PY" -m accelerate.commands.launch \
      --config_file "$TSR_DIR/settings/tsr/config/deepspeed_zero3.yaml" --num_processes "$NGPU" \
      --main_process_port $((29500 + RANDOM % 1000)) "${entry[@]}" "$cfg" ) > "$logf" 2>&1 || train_result=$?
  if [ -n "${TSR_TENSORBOARD_LOCAL_ROOT:-}" ]; then
    "$PY" "$REPO/src/rank_training.py" --sync-tensorboard "$cfg" "$WORK_DIR/tensorboard" || return $?
  fi
  return "$train_result"
}

# alpaca NAME MODEL  -> $EVAL/NAME/{model_outputs.json, leaderboard.csv, annotations.json}
alpaca() {
  local name=$1 model=$2 out=$EVAL/$1
  if [ ! -f "$out/model_outputs.json" ]; then
    need "$model/config.json"
    log "alpaca_eval generate $name"
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} "$PY" "$REPO/src/gen_alpaca.py" --model "$model" --out "$out/model_outputs.json" \
      --limit "$ALPACA_LIMIT" > "$LOGS/alpaca_gen_$name.log" 2>&1 || die "alpaca generation $name"
  fi
  if [ ! -f "$out/leaderboard.csv" ]; then
    : "${JUDGE_API_KEY:?set JUDGE_API_KEY in config.env}"
    log "alpaca_eval judge $name  (judge $JUDGE_MODEL)"
    "$PY" "$REPO/src/eval_alpaca.py" --model_outputs "$out/model_outputs.json" --name "$name" --out_dir "$out" \
      > "$LOGS/alpaca_eval_$name.log" 2>&1 || die "alpaca_eval $name, see $LOGS/alpaca_eval_$name.log"
  fi
  log "alpaca_eval $name: $(tail -1 "$out/leaderboard.csv")"
}
