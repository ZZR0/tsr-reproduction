#!/usr/bin/env bash
# One-time setup:
#   1. Python virtualenv (.venv) with the exact package versions we ran (requirements.txt);
#   2. upstream TSR at commit 020266a in third_party/TSR, plus our patches (patches/*.patch);
#   3. the base model (if BASE_MODEL in config.env is a HF repo id).
# Needs: Python 3.11, git, CUDA driver >= 12.4 (torch 2.5.1 + cu124 wheels), x86_64 Linux.
set -euo pipefail
cd "$(dirname "$0")"
. scripts/lib.sh
load_config

# 1. environment
if [ ! -x .venv/bin/python ]; then
  "${PYTHON:-python3.11}" -m venv .venv
fi
.venv/bin/pip install -q -U pip wheel
.venv/bin/pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# Exact versions, installed without dependency resolution: the environment we ran does not satisfy every declared
# pin (vllm 0.6.4.post1 declares numpy<2, we ran numpy 2.0.2), so a resolver would refuse it.
.venv/bin/pip install -q --no-deps -r requirements.txt
.venv/bin/pip install -q --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# 2. upstream code + patches
if [ ! -d "$TSR_DIR/.git" ]; then
  git clone -q "$UPSTREAM_URL" "$TSR_DIR"
fi
git -C "$TSR_DIR" checkout -q "$UPSTREAM_COMMIT"
for p in patches/*.patch; do
  if git -C "$TSR_DIR" apply --reverse --check "$REPO/$p" 2>/dev/null; then echo "already applied: $p"
  else git -C "$TSR_DIR" apply "$REPO/$p" && echo "applied: $p"; fi
done
.venv/bin/pip install -q --no-deps -e "$TSR_DIR/alignment-handbook"

# 3. base model
if [ ! -f "$BASE/config.json" ]; then
  : "${HF_TOKEN:?set HF_TOKEN in config.env (needs access to $BASE_MODEL)}"
  .venv/bin/huggingface-cli download "$BASE_MODEL" --local-dir "$BASE" --exclude "original/*"
fi

.venv/bin/python - <<'EOF'
import torch, transformers, trl, deepspeed, vllm, alpaca_eval, flash_attn
print("torch", torch.__version__, "| transformers", transformers.__version__, "| trl", trl.__version__,
      "| deepspeed", deepspeed.__version__, "| vllm", vllm.__version__, "| alpaca_eval", alpaca_eval.__version__)
print("CUDA available:", torch.cuda.is_available(), "| GPUs visible:", torch.cuda.device_count())
EOF
echo "setup done. Next: bash run.sh all   (or SMOKE=1 bash run.sh all for a quick plumbing check)"
