"""Persist each successful official judge completion independently of annotation chunks."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid


def cached_helper(original, root, namespace):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    def call(args, **kwargs):
        request_kwargs = dict(kwargs)
        client = dict(request_kwargs.get("client_kwargs") or {})
        for operational in ["max_retries", "timeout"]:
            client.pop(operational, None)
        request_kwargs["client_kwargs"] = client
        request_kwargs.pop("n_retries", None)
        request_kwargs.pop("sleep_time", None)
        key = hashlib.sha256(json.dumps([namespace, args, request_kwargs], sort_keys=True,
                                      ensure_ascii=False, default=str).encode()).hexdigest()
        path = root / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("key") != key:
                raise RuntimeError("Judge request cache key mismatch")
            return saved["choices"]
        choices = original(args, **kwargs)
        # Do not cache an exhausted decoder's empty fallback or incomplete logprob response.
        if choices and all(isinstance(c, dict) and str(c.get("text", "")).strip()
                           and c.get("logprobs") for c in choices):
            temporary = root / f".{key}.{uuid.uuid4().hex}.tmp"
            temporary.write_text(json.dumps({"key":key, "choices":choices}, ensure_ascii=False, allow_nan=False))
            temporary.replace(path)
        return choices
    return call


@contextmanager
def official_completion_cache():
    directory = os.environ.get("JUDGE_REQUEST_CACHE")
    if not directory:
        yield
        return
    from alpaca_eval.decoders import openai as decoder
    # Keep the official decoder, prompt formatting, output parsing and scheduling unchanged.
    # Only memoize its single-request helper before its ThreadPoolExecutor gathers a chunk.
    namespace = hashlib.sha256(json.dumps({
        "endpoint":os.environ.get("JUDGE_BASE_URL"), "judge":os.environ.get("JUDGE_MODEL"),
        "credential_fingerprint":hashlib.sha256(os.environ["JUDGE_API_KEY"].encode()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()
    original = decoder._openai_completion_helper
    decoder._openai_completion_helper = cached_helper(original, directory, namespace)
    try:
        yield
    finally:
        decoder._openai_completion_helper = original
