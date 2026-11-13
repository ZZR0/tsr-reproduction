# H200 throughput profile

The `h200-fast` profile changes execution settings, not the requested TSR pair-selection rules.
Use `--variant future_tie_filtered --performance-profile h200-fast --gpus 8 --train-only --skip-smoke`.

- ZeRO-2, BF16 explicit model loading, FlashAttention-2, original PyTorch AdamW.
- Communication overlap is disabled to avoid known synchronization issues in the pinned older ZeRO-1/2 implementation.
- Every optimizer update logs finite metrics; any NaN/Inf aborts training immediately. The first ZeRO-1/fused-AdamW trial was stopped after non-finite metrics and is not a valid training result.
- DPO micro-batch 4 x accumulation 1 x 8 GPUs = global batch 32. Gradient checkpointing disabled.
- Incomplete final global batches are dropped so stages use floor(n_pairs/32) updates, as in the preceding accumulation-4 run; no extra short optimizer update is introduced.
- Memory-only failures trigger a stage restart from its original initialization with micro-batch 2 then 1; effective batch remains 32. Checkpointing is the final memory fallback. Other failures stop and preserve logs.
- vLLM token scheduling budget 16384, max concurrent sequences 512 per GPU, prefix caching and chunked prefill enabled. The original maximum context and maximum output lengths are retained.
- When the same model generates and judges, one engine handles both calls. In round 1, M0 generates past answers first; a single M1 engine then generates current answers and judges current/past answers. Models, prompts and scores are unchanged. Fused calls and scheduling change RNG consumption, so exact sampled answers can differ.
- Optional first-round M0 generation/judgment data reuse requires matching completed M0 hashes, inputs, generation settings and pair-builder source. No incomplete DPO model or optimizer state is reused.
- Candidate/judgment caches are removed after their training stage; small performance, filtering and source-count metrics remain. Each stage saves only its final model.

The one-hour target covers the four DPO stages and their required generation/self-scoring, reusing completed M0. It excludes scheduling delays and API evaluation. It is a performance target, not a forced early stopping condition; total work is never silently reduced to meet it.
