# iter1 variants used in the investigation (results in README.md).
# Each row changes one thing relative to the main TSR iter1 run; everything else is shared and reused.
#   name | pair flags for D1 (Mf1 data) | pair flags for D2 (M2 data) | Mf1 init | vLLM seed | needs p(score token)
# "-" = same as the main run. If D1, Mf1 init and seed are all "-", the main run's Mf1 and future responses are reused.
VARIANTS=(
  "future_tie_length_distinct|-|--source-priority-length-gap --different-text --filter --rej_tie mi|M1|-|0" # strict gap -> source priorities -> closest distinct-text pair
  "future_tie_length|-|--source-priority-length-gap --filter --rej_tie mi|M1|-|0" # strict gap -> source priority -> joint closest length, all four stages
  "future_tie_filtered|-|--filter --rej_tie mi|M1|-|0"                   # both rounds: future chosen ties, current rejected ties; positive-score-gap pairs only
  "future_tie_unfiltered|-|--keep-equal-scores --rej_tie mi|M1|-|0"         # both rounds: future chosen ties, current rejected ties, retain equal-score pairs
  "min_length_gap|-|--min-length-gap --filter|M1|-|0"                      # joint closest-length score-extreme pair; no model priority
  "current_tie|-|--strict --filter --rej_tie mi|M1|-|0"                    # current/future best; equal scores choose current
  "future_only|-|--future-only --filter|M1|-|0"                         # chosen exclusively from Mf1; positive score gap only
  "strict|-|--strict --filter|M1|-|0"                                    # paper Alg. 1: upgrade only if strictly better, keep s_chosen > s_rejected
  "rej_past|--rej_tie past|--rej_tie past|M1|-|0"                        # rejected past/current tie -> past
  "rej_past_conf|--rej_tie past --within conf|--rej_tie past --within conf|M1|-|1"  # + among same-score candidates take the highest p(score token)
  "cho_longest|-|--cho_len longest|M1|-|0"                               # chosen: longest among top-score ties
  "len_control|-|--cho_len shortest --rej_len longest|M1|-|0"            # Meta-Rewarding length control
  "mf1_from_mf0|-|-|Mf0|-|0"                                             # upstream config_full_iter1_tmp.yaml (init from iter0_tmp)
  "resample|-|-|M1|1234|0"                                               # redo every generation and judgement with another vLLM seed
)
# Three variants are not a single row of this table and have their own commands:
#   run.sh iter1_mix     iter1 DPO on iter0 D2 + iter1 D2 (10k pairs); diagnostic only, not the authors' setup
#   run.sh iter0_strict  iter0 with the paper's Algorithm 1 rule
#   run.sh iter0_conf    iter0 with score x p(score token) as the score
