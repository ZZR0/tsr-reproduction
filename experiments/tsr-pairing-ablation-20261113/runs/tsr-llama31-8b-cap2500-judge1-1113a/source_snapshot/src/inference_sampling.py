"""Temporarily use independent sampling parameters for local self-judging."""
from contextlib import contextmanager
import copy
import math


@contextmanager
def self_judge_sampling(generator, factory, temperature):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('Self-judge temperature must be finite and nonnegative')
    original = generator.sampling_params
    was_judging = getattr(generator, '_tsr_self_judging', False)
    # Build fresh params so vLLM reruns greedy validation and avoids stale cached sampling_type.
    # These are exactly the explicit sampling fields in the pinned upstream generator.
    judge_params = factory(temperature=temperature, top_p=original.top_p,
                           max_tokens=original.max_tokens, stop=copy.deepcopy(original.stop))
    try:
        generator.sampling_params = judge_params
        generator._tsr_self_judging = True
        yield
    finally:
        generator.sampling_params = original
        generator._tsr_self_judging = was_judging
