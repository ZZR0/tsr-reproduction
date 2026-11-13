# Pipeline stages. Model names follow the paper:
#   M0 = SFT;  Mf0 = DPO(M0, D1_0) = SR iter0 = w/o Future iter0;  M1 = DPO(M0, D2_0) = TSR iter0
#   Mf1 = DPO(M1, D1_1);  M2 = DPO(M1, D2_1) = TSR iter1;  SR1 = SR iter1;  W1 = w/o Future iter1
# D1 = past anchoring (chosen: best of current; rejected: worst of current and M0, ties current),
# with strict chosen_score > rejected_score filtering for future-model training (Mf0/Mf1).
# Default D2: chosen is upgraded only if future scores strictly higher; current wins ties.
# Default D1/D2 both require chosen_score > rejected_score; rejected ties select current.

QA() { echo "$TSR_DIR/data/SelfPlayBeginData/qa$1.jsonl"; }

stage_sft() {
  sft
}

stage_iter0() {
  local -a d1_flags=(--filter --rej_tie mi) d2_flags=(--strict --filter --rej_tie mi)
  if [ "${TSR_PAIR_SELECTION:-current_tie}" = min_length_gap ]; then
    d1_flags=(--min-length-gap --filter); d2_flags=(--min-length-gap --filter)
  elif [ "${TSR_PAIR_SELECTION:-current_tie}" = future_tie_longer_distinct ]; then
    d1_flags=(--source-priority-length-gap --different-text --chosen-longer --filter --rej_tie mi); d2_flags=(--source-priority-length-gap --different-text --chosen-longer --filter --rej_tie mi)
  elif [ "${TSR_PAIR_SELECTION:-current_tie}" = future_tie_charcap_distinct ]; then
    d1_flags=(--different-text --max-chosen-characters "${TSR_MAX_CHOSEN_CHARACTERS:-2500}" --filter --rej_tie mi); d2_flags=(--different-text --max-chosen-characters "${TSR_MAX_CHOSEN_CHARACTERS:-2500}" --filter --rej_tie mi)
  elif [ "${TSR_PAIR_SELECTION:-current_tie}" = future_tie_distinct ]; then
    d1_flags=(--different-text --filter --rej_tie mi); d2_flags=(--different-text --filter --rej_tie mi)
  elif [ "${TSR_PAIR_SELECTION:-current_tie}" = future_tie_length_distinct ]; then
    d1_flags=(--source-priority-length-gap --different-text --filter --rej_tie mi); d2_flags=(--source-priority-length-gap --different-text --filter --rej_tie mi)
  elif [ "${TSR_PAIR_SELECTION:-current_tie}" = future_tie_length ]; then
    d1_flags=(--source-priority-length-gap --filter --rej_tie mi); d2_flags=(--source-priority-length-gap --filter --rej_tie mi)
  elif [ "${TSR_PAIR_SELECTION:-current_tie}" = future_tie_filtered ]; then
    d1_flags=(--filter --rej_tie mi); d2_flags=(--filter --rej_tie mi)
  elif [ "${TSR_PAIR_SELECTION:-current_tie}" = future_tie_unfiltered ]; then
    d1_flags=(--keep-equal-scores --rej_tie mi); d2_flags=(--keep-equal-scores --rej_tie mi)
  fi
  local W=$DATA/iter0 Q; Q=$(QA 1); mkdir -p "$W"; need "$M0/.done"
  if [ "${TSR_FUSE_INFERENCE:-0}" = 1 ]; then
    if [ -f "$W/0_chosen_response.json" ] && [ ! -f "$W/0_chosen_judgement.json" ]; then
      # Responses may be reused independently; scores must be recomputed at this run's temperature.
      judge "$M0" "$W/0_chosen_response.json" "$W/0_chosen_judgement.json"
    else
      genjudge "$M0" "$Q" "$W/0_chosen_judgement.json"
    fi
  else
    gen   "$M0" "$Q" "$W/0_chosen_response.json"
    judge "$M0" "$W/0_chosen_response.json" "$W/0_chosen_judgement.json"
  fi
  # Upstream also generates a "rejected" set at iter0 with the same model, prompts and seed, i.e. an identical copy;
  # we pass the chosen file twice.
  pairs past "$W/D1" "$W/0_chosen_judgement.json" "$W/0_chosen_judgement.json" -- "${d1_flags[@]}"
  dpo Mf0 "$M0" "$W/D1"
  gen   "$MODELS/Mf0" "$Q" "$W/0_future_response.json"
  judge "$M0" "$W/0_future_response.json" "$W/0_future_judgement.json"
  pairs future "$W/D2" "$W/0_chosen_judgement.json" "$W/0_chosen_judgement.json" "$W/0_future_judgement.json" -- "${d2_flags[@]}"
  dpo M1 "$M0" "$W/D2"
}

# stage_iter1 [VARIANT] defaults to current_tie; explicit main keeps the historical D2 rules.
stage_iter1() {
  local tag=${1:-${TSR_PAIR_SELECTION:-current_tie}} d1="-" d2="-" init=M1 seed="-" conf=0
  if [ "$tag" != main ]; then
    . "$REPO/configs/variants.sh"; local row="" r
    for r in "${VARIANTS[@]}"; do [ "${r%%|*}" = "$tag" ] && row=$r; done
    [ -n "$row" ] || die "unknown iter1 variant '$tag' (see configs/variants.sh)"
    IFS='|' read -r _ d1 d2 init seed conf <<< "$row"
  fi
  local -a base_d1_flags=(--filter --rej_tie mi)
  if [ "$tag" = min_length_gap ]; then
    base_d1_flags=(--min-length-gap --filter)
  elif [ "$tag" = future_tie_longer_distinct ]; then
    base_d1_flags=(--source-priority-length-gap --different-text --chosen-longer --filter --rej_tie mi)
  elif [ "$tag" = future_tie_charcap_distinct ]; then
    base_d1_flags=(--different-text --max-chosen-characters "${TSR_MAX_CHOSEN_CHARACTERS:-2500}" --filter --rej_tie mi)
  elif [ "$tag" = future_tie_distinct ]; then
    base_d1_flags=(--different-text --filter --rej_tie mi)
  elif [ "$tag" = future_tie_length_distinct ]; then
    base_d1_flags=(--source-priority-length-gap --different-text --filter --rej_tie mi)
  elif [ "$tag" = future_tie_length ]; then
    base_d1_flags=(--source-priority-length-gap --filter --rej_tie mi)
  elif [ "$tag" = future_tie_unfiltered ]; then
    base_d1_flags=(--keep-equal-scores --rej_tie mi)
  fi
  local B=$DATA/iter1 W=$DATA/iter1 Q; Q=$(QA 2); mkdir -p "$B"; need "$MODELS/M1/.done" "$M0/.done"
  [ "$tag" != main ] && W=$DATA/iter1_$tag && mkdir -p "$W"
  local G=$B; [ "$seed" != "-" ] && G=$W   # where current/past responses live
  local sfx=""; [ "$tag" != main ] && sfx=_$tag
  local c=$G/1_chosen_judgement.json r=$G/1_rejected_judgement.json

  # current (M1) and past (M0) responses on qa2, both judged by M1
  ( [ "$seed" != "-" ] && export VLLM_SEED=$seed
    if [ "${TSR_FUSE_INFERENCE:-0}" = 1 ]; then
      # Generate past answers first; one M1 engine generates current answers and judges both sets.
      gen "$M0" "$Q" "$G/1_rejected_response.json"
      genjudge "$MODELS/M1" "$Q" "$c" "$G/1_rejected_response.json" "$r"
    else
      gen   "$MODELS/M1" "$Q" "$G/1_chosen_response.json"
      judge "$MODELS/M1" "$G/1_chosen_response.json" "$c"
      gen   "$M0" "$Q" "$G/1_rejected_response.json"
      judge "$MODELS/M1" "$G/1_rejected_response.json" "$r"
    fi ) || exit 1
  [ "$conf" = 1 ] && { conf_of "$MODELS/M1" "$c"; conf_of "$MODELS/M1" "$r"; }

  # past anchoring -> Mf1 (reuse the main run's Mf1 when nothing upstream of it changed)
  local mf1=Mf1 fut=$B/1_future_judgement.json
  if [ "$d1" != "-" ] || [ "$init" != M1 ] || [ "$seed" != "-" ]; then
    mf1=Mf1$sfx; fut=$W/1_future_judgement.json
    if [ "$d1" = "-" ]; then pairs past "$W/D1" "$c" "$r" -- "${base_d1_flags[@]}"; else pairs past "$W/D1" "$c" "$r" -- $d1 --filter; fi
    dpo "$mf1" "$MODELS/$init" "$W/D1"
  else
    pairs past "$B/D1" "$c" "$r" -- "${base_d1_flags[@]}"
    dpo Mf1 "$MODELS/M1" "$B/D1"
  fi

  # future responses from Mf1 on qa2, judged by M1 -> D2 -> M2
  ( [ "$seed" != "-" ] && export VLLM_SEED=$seed
    gen   "$MODELS/$mf1" "$Q" "${fut%_judgement.json}_response.json"
    judge "$MODELS/M1" "${fut%_judgement.json}_response.json" "$fut" ) || exit 1
  [ "$conf" = 1 ] && conf_of "$MODELS/M1" "$fut"
  if [ "$tag" = future_only ]; then
    # Rebuild this small dataset even on resume: an existing train.json must not silently bypass the new rules.
    "$PY" "$REPO/src/build_pairs.py" future --chosen "$c" --rejected "$r" --future "$fut" \
      --out "$W/D2" --future-only --filter > "$LOGS/pairs_M2_future_only.log" 2>&1 || return $?
    "$PY" - "$W/D2/stats.json" <<'EOF' || return $?
import json, sys
s = json.load(open(sys.argv[1]))
assert s["chosen_policy"] == "future_only" and s["strict_score_gap"]
assert s["n_pairs"] > 0 and s["gap_min"] > 0
assert s.get("retained_chosen_from_Mi", 0) == 0 and s["retained_chosen_from_Mf"] == s["n_pairs"]
assert s["retained_tie_pairs"] == 0
EOF
  elif [ "$d2" = "-" ]; then pairs future "$W/D2" "$c" "$r" "$fut"; else pairs future "$W/D2" "$c" "$r" "$fut" -- $d2; fi
  dpo "M2$sfx" "$MODELS/M1" "$W/D2"
}

# SR iter1: Mf0 (= SR iter0) generates on qa2 and judges itself; chosen = its best, rejected = its worst.
stage_sr() {
  local W=$DATA/sr Q; Q=$(QA 2); mkdir -p "$W"; need "$MODELS/Mf0/.done"
  gen   "$MODELS/Mf0" "$Q" "$W/sr1_response.json"
  judge "$MODELS/Mf0" "$W/sr1_response.json" "$W/sr1_judgement.json"
  pairs past "$W/D_SR1" "$W/sr1_judgement.json" "$W/sr1_judgement.json"
  dpo SR1 "$MODELS/Mf0" "$W/D_SR1"
}

# w/o Future iter1 (paper Table 2): past anchoring from Mf0 (= w/o Future iter0). Mf0's qa2 responses (from stage_sr)
# vs M0's qa2 responses (from stage_iter1), all judged by Mf0.
stage_wof() {
  local W=$DATA/wof; mkdir -p "$W"
  need "$DATA/sr/sr1_judgement.json" "$DATA/iter1/1_rejected_response.json" "$MODELS/Mf0/.done"
  judge "$MODELS/Mf0" "$DATA/iter1/1_rejected_response.json" "$W/wof1_m0_judgement.json"
  pairs past "$W/D_W1" "$DATA/sr/sr1_judgement.json" "$W/wof1_m0_judgement.json"
  dpo W1 "$MODELS/Mf0" "$W/D_W1"
}

# Diagnostic: iter1 DPO on iter0 D2 + iter1 D2 (not the authors' setup; they confirmed iter1 uses only iter1 data).
stage_iter1_mix() {
  local W=$DATA/iter1_mix; mkdir -p "$W/D2"; need "$DATA/iter0/D2/train.json" "$DATA/iter1/D2/train.json"
  [ -f "$W/D2/train.json" ] || cat "$DATA/iter0/D2/train.json" "$DATA/iter1/D2/train.json" > "$W/D2/train.json"
  dpo M2_mix "$MODELS/M1" "$W/D2"
}

# Variant: iter0 with the paper's Algorithm 1 rule (upgrade only if strictly better; keep only s_chosen > s_rejected).
stage_iter0_strict() {
  local W=$DATA/iter0; need "$W/0_chosen_judgement.json" "$W/0_future_judgement.json"
  pairs future "$DATA/iter0_strict/D2" "$W/0_chosen_judgement.json" "$W/0_chosen_judgement.json" "$W/0_future_judgement.json" -- --strict --filter
  dpo M1_strict "$M0" "$DATA/iter0_strict/D2"
}

# Variant: iter0 with score x p(score token) as the score.
stage_iter0_conf() {
  local W=$DATA/iter0; need "$W/0_chosen_judgement.json" "$W/0_future_judgement.json"
  conf_of "$M0" "$W/0_chosen_judgement.json"; conf_of "$M0" "$W/0_future_judgement.json"
  pairs future "$DATA/iter0_conf/D2" "$W/0_chosen_judgement.json" "$W/0_chosen_judgement.json" "$W/0_future_judgement.json" -- --score conf
  dpo M1_conf "$M0" "$DATA/iter0_conf/D2"
}

# conf_of JUDGE X_judgement.json -> X_conf.json : p(score token) of each saved judgement (teacher-forced, no resampling)
conf_of() {
  local out=${2%_judgement.json}_conf.json
  [ -f "$out" ] && return 0
  log "p(score token) $(basename "$2")"
  CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} "$PY" "$REPO/src/tsr_conf.py" --judge "$1" --inp "$2" --out "$out.tmp" --limit "$LIMIT" \
    > "$LOGS/conf_$(basename "$2" .json).log" 2>&1 || die "tsr_conf $(basename "$2")"
  mv "$out.tmp" "$out"
}

# AlpacaEval for every model that exists (or the ones given).
stage_eval() {
  local names=("$@") n
  [ ${#names[@]} -eq 0 ] && names=(M0 Mf0 M1 Mf1 M2_current_tie M2_future_only M2 SR1 W1)
  for n in "${names[@]}"; do
    if [ -f "$MODELS/$n/.done" ]; then alpaca "$n" "$MODELS/$n"; else log "skip eval $n (model not trained)"; fi
  done
  "$PY" "$REPO/src/summarize.py" "$EVAL" --csv "$EVAL/summary.csv" | tee -a "$LOGS/run.log"
}
