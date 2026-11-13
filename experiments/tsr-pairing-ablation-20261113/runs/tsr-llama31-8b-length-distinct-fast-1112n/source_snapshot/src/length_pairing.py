"""Joint length matching inside score-extreme TSR candidate pools."""
import hashlib
import math


def candidates(record, source):
    responses, scores = record['responses'], record['scores']
    if not responses or len(responses) != len(scores):
        raise ValueError('Candidate scores/responses must be nonempty and aligned')
    if any(not isinstance(t, str) for t in responses) or any(
        not isinstance(s, (int, float)) or not math.isfinite(s) for s in scores
    ):
        raise ValueError('Candidate responses must be text and scores must be finite')
    return [dict(text=t, score=s, source=source) for t, s in zip(responses, scores)]


def select_min_length_pair(current, past, future=None, *, source_priority=False, require_distinct_text=False):
    """Maximize chosen score, minimize rejected score, then jointly minimize |len(c)-len(r)|.

    D1 chosen: current. D2 chosen: current + future. Rejected: current + past.
    With source_priority=True, first restrict extrema to future chosen/current rejected
    on source ties. Length matching cannot change the selected scores or sources.
    Remaining length ties use deterministic response-text hashes.
    """
    high_candidates = candidates(current, 'Mi')
    low_candidates = high_candidates + candidates(past, 'M0')
    if future is not None:
        high_candidates += candidates(future, 'Mf')
    high = max(x['score'] for x in high_candidates)
    low = min(x['score'] for x in low_candidates)
    if high <= low:
        return dict(valid=False, chosen_score=high, rejected_score=low, reason="nonpositive_score_gap")
    if source_priority:
        # Scores and source priorities are resolved before comparing any lengths.
        high_source = 'Mf' if future is not None and max(future['scores']) >= max(current['scores']) else 'Mi'
        low_source = 'Mi' if min(current['scores']) <= min(past['scores']) else 'M0'
        high_candidates = [x for x in high_candidates if x['source'] == high_source]
        low_candidates = [x for x in low_candidates if x['source'] == low_source]
    highs = [x for x in high_candidates if x['score'] == high]
    lows = [x for x in low_candidates if x['score'] == low]
    def key(pair):
        c, r = pair
        fingerprint = hashlib.sha256((current['prompt'] + '\0' + c['text'] + '\0' + r['text']).encode()).digest()
        return abs(len(c['text']) - len(r['text'])), fingerprint
    pairs = [(c, r) for c in highs for r in lows
             if not require_distinct_text or c['text'].strip() != r['text'].strip()]
    if not pairs:
        return dict(valid=False, chosen_score=high, rejected_score=low, reason="no_distinct_text_pair",
                    n_high_candidates=len(highs), n_low_candidates=len(lows))
    chosen, rejected = min(pairs, key=key)
    return dict(valid=True, chosen=chosen, rejected=rejected, chosen_score=high, rejected_score=low,
                length_gap=abs(len(chosen['text'])-len(rejected['text'])),
                n_high_candidates=len(highs), n_low_candidates=len(lows),
                valid_pair_combinations=len(pairs), identical_text_combinations_excluded=len(highs)*len(lows)-len(pairs))
