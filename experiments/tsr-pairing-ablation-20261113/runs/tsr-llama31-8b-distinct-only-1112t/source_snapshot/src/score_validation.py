"""Validate explicit local self-judge scores without treating parse failures as zero."""
import collections
import math
import re

PARSER = 'explicit-score-v1'
LABEL = re.compile(r'\bScore:\s*([^\s]+)', re.IGNORECASE)
NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)')


def extract_score(judgments):
    # This experiment requests exactly one judgment per response. Missing,
    # duplicated, conflicting and out-of-range score labels are invalid.
    if not isinstance(judgments, list) or len(judgments) != 1 or not isinstance(judgments[0], str):
        return None, 'invalid_judgment_count_or_type'
    matches = LABEL.findall(judgments[0])
    if not matches:
        return None, 'missing_score'
    if len(matches) != 1:
        return None, 'multiple_score_labels'
    token = matches[0].strip('`*').rstrip('.,;')
    if not NUMBER.fullmatch(token):
        return None, 'malformed_score'
    value = float(token)
    if not math.isfinite(value) or not 0 <= value <= 5:
        return None, 'out_of_range_score'
    return value, None


def annotate_record(record):
    responses, judgments = record['responses'], record.get('judgments')
    if not isinstance(judgments, list) or len(judgments) != len(responses):
        raise ValueError('Missing/misaligned raw judgments; cannot validate candidate scores')
    if any(not isinstance(text, str) for text in responses):
        raise ValueError('Candidate responses must be strings')
    parsed = [extract_score(j) for j in judgments]
    scores = [s for s, _ in parsed]
    valid = [i for i, s in enumerate(scores) if s is not None]
    return dict(record, scores=scores, score_parser=PARSER,
                score_errors=[reason for _, reason in parsed],
                chosen_response=responses[max(valid, key=lambda i: scores[i])] if valid else None,
                rejected_response=None)


def remove_invalid_candidates(record):
    row = annotate_record(record)  # Reparse raw text rather than trusting cached numeric zeroes.
    valid = [i for i, s in enumerate(row['scores']) if s is not None]
    return dict(row, responses=[row['responses'][i] for i in valid],
                scores=[row['scores'][i] for i in valid],
                judgments=[row['judgments'][i] for i in valid],
                score_errors=[None] * len(valid), original_candidate_indices=valid)


def validation_summary(records):
    rows = [annotate_record(row) for row in records]
    scores = [s for row in rows for s in row['scores'] if s is not None]
    errors = collections.Counter(e for row in rows for e in row['score_errors'] if e)
    total = sum(len(row['responses']) for row in rows)
    groups = [[s for s in row['scores'] if s is not None] for row in rows]
    return dict(score_parser=PARSER, n_items=len(rows), n_judgments=total,
                valid_score_candidates=len(scores), invalid_score_candidates=total-len(scores),
                parse_fail_rate=round((total-len(scores))/total, 6) if total else 0,
                out_of_range_rate=errors['out_of_range_score']/total if total else 0,
                invalid_reasons=dict(errors), groups_without_valid_scores=sum(not g for g in groups),
                group_all_tie_rate=sum(bool(g) and max(g)==min(g) for g in groups)/len(rows) if rows else 0,
                group_max_is_5_rate=sum(bool(g) and max(g)==5 for g in groups)/len(rows) if rows else 0,
                score_dist=dict(collections.Counter(str(s) for s in scores)),
                score_mean=sum(scores)/len(scores) if scores else None,
                resp_mean_chars=sum(len(t) for row in rows for t in row['responses'])/total if total else None)
