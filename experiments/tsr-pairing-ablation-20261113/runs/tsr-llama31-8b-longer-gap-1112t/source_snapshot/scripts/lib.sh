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

# genjudge MODEL QA CURRENT_JUDGEMENT [PAST_RESPONSE PAST_JUDGEMENT]
genjudge() {
  local model=$1 prompts=$2 out=$3 extra_in=${4:-} extra_out=${5:-}
  if [ -f "$out" ] && { [ -z "$extra_out" ] || [ -f "$extra_out" ]; }; then
    log "skip fused genjudge $(basename "$out") (exists)"; return 0
  fi
  need "$model/config.json" "$prompts" ${extra_in:+"$extra_in"}
  log "genjudge $(basename "$out") <- $(basename "$model") (shared engine)"
  _shard genjudge "$model" "$prompts" "$out" "$extra_in" "$extra_out"
  local target
  for target in "$out" ${extra_out:+"$extra_out"}; do
    "$PY" "$REPO/src/judge_stats.py" "$target" > "${target%.json}.stats.json"
    "$PY" - "${target%.json}.stats.json" <<'EOF' || die "fused judge health check failed"
import json, sys
assert json.load(open(sys.argv[1]))["parse_fail_rate"] <= 0.15
EOF
  done
}

_merge_shards() {
  local out=$1; shift
  "$PY" "$REPO/src/tsr_shard.py" merge --out "$out.tmp" "$@" >> "$LOGS/run.log" || die "merge $out"
  local n; n=$("$PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$out.tmp")
  [ "$n" = "$(expected_rows)" ] || die "$out has $n rows, expected $(expected_rows)"
  mv "$out.tmp" "$out"; rm -f "$@"
}

_shard() {
  local mode=$1 model=$2 inp=$3 out=$4 extra_in=${5:-} extra_out=${6:-}
  local name i=0 pids=() parts=() extra_parts=() extra_flags=() seed_flags=() engine_seed
  name=$(basename "$out" .json)
  for g in "${GPU_LIST[@]}"; do
    extra_flags=()
    seed_flags=(--seed "${TSR_TRAIN_SEED:-42}")
    if [ -n "${VLLM_SEED:-}" ]; then
      seed_flags+=(--vllm_seed "$VLLM_SEED")
    elif [ -n "${TSR_GENERATION_SEED:-}" ]; then
      engine_seed=$((${TSR_GENERATION_SEED} + i))
      seed_flags+=(--vllm_seed "$engine_seed")
    fi
    if [ -n "$extra_in" ]; then
      extra_flags=(--extra-judge-input "$extra_in" --extra-judge-out "$extra_out.part$i")
      extra_parts+=("$extra_out.part$i")
    fi
    CUDA_VISIBLE_DEVICES=$g "$PY" "$REPO/src/tsr_shard.py" "$mode" --model "$model" --input "$inp" \
      --out "$out.part$i" --shard $i --nshards "$NGPU" --limit "$LIMIT" "${seed_flags[@]}" \
      "${extra_flags[@]}" > "$LOGS/$name.part$i.log" 2>&1 &
    pids+=($!); parts+=("$out.part$i"); i=$((i + 1))
  done
  for p in "${pids[@]}"; do wait "$p" || die "$mode $name failed, see $LOGS/$name.part*.log"; done
  mkdir -p "$WORK_DIR/metrics/inference"
  for ((i=0; i<NGPU; i++)); do
    if [ -f "$out.part$i.metrics.json" ]; then
      mv "$out.part$i.metrics.json" "$WORK_DIR/metrics/inference/$name.part$i.json"
    fi
  done
  _merge_shards "$out" "${parts[@]}"
  if [ -n "$extra_out" ]; then _merge_shards "$extra_out" "${extra_parts[@]}"; fi
}

# pairs past|future OUT_DIR CHOSEN_J REJECTED_J [FUTURE_J] [-- build_pairs flags]
# Builds DPO pairs with src/build_pairs.py. Without extra flags it also runs the official upstream script
# (settings/tsr/reformat2DPO_{past,future}.py) and requires both outputs to be identical.
pairs() {
  local mode=$1 out=$2 c=$3 r=$4 f=""; shift 4
  if [ "$mode" = future ]; then f=$1; shift; fi
  [ "${1:-}" = "--" ] && shift
  if [ "${TSR_DROP_INVALID_SCORES:-0}" = 1 ]; then
    set -- "$@" --drop-invalid-scores
  fi
  local strict_gap=0 keep_equal=0 flag
  for flag in "$@"; do
    case "$flag" in --filter|--future-only|--min-length-gap|--source-priority-length-gap) strict_gap=1 ;; --keep-equal-scores) keep_equal=1 ;; esac
  done
  if [ -f "$out/train.json" ]; then
    if [ "$strict_gap" = 1 ]; then
      # Do not relabel an old unfiltered dataset or reuse models trained on it.
      validate_filtered_pairs "$out" --pair-mode "$mode" "$@" || die "cached pairs are unfiltered or empty: $out; use a fresh WORK_DIR/run (old models must not be reused)"
    fi
    if [ "$keep_equal" = 1 ]; then
      validate_unfiltered_pairs "$out" "$mode" || die "cached pairs differ from the unfiltered source-priority rules: $out; use a fresh run"
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
    validate_filtered_pairs "$out.tmp" --pair-mode "$mode" "$@" || die "filtered pairs are empty or invalid: $out.tmp; refusing DPO training"
  fi
  if [ "$keep_equal" = 1 ]; then
    validate_unfiltered_pairs "$out.tmp" "$mode" || die "unfiltered pair rules differ: $out.tmp; refusing DPO training"
  fi
  mkdir -p "$out"; mv "$out.tmp"/* "$out"/; rmdir "$out.tmp"
  log "pairs $(basename "$out"): $(tr -d '\n ' < "$out/stats.json")"
}

# Validate actual training inputs before DPO, including on resume. This is not a standalone test run.
validate_unfiltered_pairs() {
  "$PY" - "$@" <<'EOF'
import json, sys
from pathlib import Path
root, mode = Path(sys.argv[1]), sys.argv[2]
s = json.loads((root / "stats.json").read_text())
assert s["pair_selection"] == "future_tie_unfiltered"
assert s["strict_score_gap"] is False and s["filter_equal_scores"] is False
assert s["model_source_priority"] is True and s["length_selection"] == "none"
assert s["rejected_tie_break"] == "current"
assert s["chosen_tie_break"] == ("future" if mode == "future" else "not_applicable")
assert s["chosen_policy"] == ("current_future_best" if mode == "future" else "current_best")
assert s.get("filtered_out", 0) == 0 and s["gap_min"] >= 0
assert s.get("retained_tie_pairs", 0) == s.get("tie_pair", 0)
assert s["n_pairs"] == s["n_prompts"] > 0
assert sum(1 for line in (root / "train.json").open() if line.strip()) == s["n_pairs"]
EOF
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
    flags.add_argument("--min-length-gap", action="store_true")
    flags.add_argument("--source-priority-length-gap", action="store_true")
    flags.add_argument("--different-text", action="store_true")
    flags.add_argument('--chosen-longer', action='store_true')
    flags.add_argument("--drop-invalid-scores", action="store_true")
    flags.add_argument("--pair-mode", choices=["past", "future"])
    expected, _ = flags.parse_known_args(sys.argv[2:])
    if expected.drop_invalid_scores:
        invalid = stats.get('invalid_score_filter', {})
        if stats.get('drop_invalid_scores') is not True or invalid.get('enabled') is not True or invalid.get('parser') != 'explicit-score-v1':
            raise ValueError('Missing invalid-score candidate filtering evidence')
    if expected.source_priority_length_gap:
        selection = 'future_tie_longer_distinct' if expected.chosen_longer else 'future_tie_length_distinct' if expected.different_text else 'future_tie_length'
        length_selection = 'minimum_positive_gap_after_source_priority' if expected.chosen_longer else 'joint_minimum_after_source_priority'
        order = ['strict_score_gap','model_source_priority','distinct_text_chosen_longer','minimum_positive_character_length_gap'] if expected.chosen_longer else ['strict_score_gap','model_source_priority','joint_minimum_character_length_gap']
        if (stats.get("pair_selection") != selection
                or stats.get("model_source_priority") is not True
                or stats.get("length_selection") != length_selection
                or stats.get("selection_order") != order
                or stats.get("rejected_tie_break") != "current"):
            raise ValueError("Cached score/source/length priority order differs")
    if expected.different_text and not expected.source_priority_length_gap:
        if (stats.get("pair_selection") != "future_tie_distinct"
                or stats.get("model_source_priority") is not True
                or stats.get("length_selection") != "none"
                or stats.get("selection_order") != ["strict_score_gap", "model_source_priority", "first_distinct_combination"]):
            raise ValueError("Cached first-distinct source-priority policy differs")
    if expected.different_text:
        if stats.get("require_distinct_text") is not True or stats.get("identical_pair", 0) != 0:
            raise ValueError("Distinct-text rule missing")
        with (root / "train.json").open() as data:
            for line in data:
                pair = json.loads(line)
                if pair["chosen"][-1]["content"].strip() == pair["rejected"][-1]["content"].strip():
                    raise ValueError("Identical chosen/rejected training text")
                if expected.chosen_longer and len(pair['chosen'][-1]['content']) <= len(pair['rejected'][-1]['content']):
                    raise ValueError('Chosen must be strictly longer than rejected')
    if expected.chosen_longer:
        if stats.get('require_chosen_longer') is not True or stats.get('signed_length_gap_min',0) <= 0 or stats.get('chosen_longer_pairs') != stats.get('n_pairs'):
            raise ValueError('Missing strict chosen-longer evidence')
    if expected.min_length_gap:
        if (stats.get("pair_selection") != "min_length_gap"
                or stats.get("chosen_tie_break") != "joint_minimum_length_gap"
                or stats.get("rejected_tie_break") != "joint_minimum_length_gap"
                or stats.get("model_source_priority") is not False):
            raise ValueError("Cached joint length matching rule differs")
    elif stats.get("rejected_tie_break") != ("current" if expected.rej_tie == "mi" else "past"):
        raise ValueError("Cached rejected tie-breaking rule differs")
    if "--strict" in sys.argv[2:] and stats.get("chosen_tie_break") != "current":
        raise ValueError("Cached chosen current/future tie-breaking rule differs")
    if (expected.pair_mode == "future" and not expected.min_length_gap
            and "--strict" not in sys.argv[2:] and "--future-only" not in sys.argv[2:]
            and stats.get("chosen_tie_break") != "future"):
        raise ValueError("Cached chosen current/future tie-breaking must prefer future")
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
  if [ "${TSR_PERFORMANCE_PROFILE:-reference}" = h200-fast ]; then
    local micro=${TSR_DPO_MICRO_BATCH:-4} gc=${TSR_GRADIENT_CHECKPOINTING:-0} attempt=0
    while true; do
      attempt=$((attempt + 1))
      log "DPO performance attempt $attempt: $name micro=$micro accumulation=$((32 / NGPU / micro)) checkpointing=$gc ZeRO=${TSR_ZERO_STAGE:-1}"
      if _train "$cfg" "$TSR_DIR/alignment-handbook/scripts/run_dpo.py" "$LOGS/dpo_$name.log"; then break; fi
      if ! grep -Eqi 'CUDA out of memory|torch.OutOfMemoryError|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$LOGS/dpo_$name.log"; then
        die "dpo $name failed for a non-memory reason; see log"
      fi
      cp "$LOGS/dpo_$name.log" "$LOGS/dpo_$name.oom-attempt$attempt.log"
      "$PY" - "$MODELS" "$MODELS/$name" <<'EOF' || die "unsafe fallback output directory"
from pathlib import Path
import shutil, sys
base, model = map(Path, sys.argv[1:])
assert model.parent.resolve() == base.resolve() and not model.is_symlink() and not (model / ".done").exists()
if model.exists(): shutil.rmtree(model)
EOF
      if [ "$micro" -gt 1 ]; then micro=$((micro / 2)); elif [ "$gc" = 0 ]; then gc=1; else die "DPO memory fallback exhausted"; fi
      TSR_DPO_MICRO_BATCH=$micro TSR_GRADIENT_CHECKPOINTING=$gc "$PY" "$REPO/src/make_cfg.py" dpo \
        --name "$name" --init "$init" --data "$data" --out_dir "$MODELS/$name" --ngpu "$NGPU" \
        --max_steps "$MAX_STEPS" --cfg "$cfg" > /dev/null || die "fallback config $name"
    done
  else
    _train "$cfg" "$TSR_DIR/alignment-handbook/scripts/run_dpo.py" "$LOGS/dpo_$name.log" || die "dpo $name, see $LOGS/dpo_$name.log"
  fi
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
      --config_file "${TSR_ACCELERATE_CONFIG:-$TSR_DIR/settings/tsr/config/deepspeed_zero3.yaml}" --num_processes "$NGPU" \
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
