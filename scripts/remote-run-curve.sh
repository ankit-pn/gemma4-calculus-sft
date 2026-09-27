#!/usr/bin/env bash
set -euo pipefail
cd /workspace/calculus_sft
PY=${CURVE_PYTHON:-/opt/calculus-venv/bin/python}
mkdir -p output/logs

# Fail fast if the CUDA stack was accidentally replaced by pip.
"$PY" - <<'PY'
import torch
assert torch.__version__.startswith('2.8.0+cu128'), torch.__version__
assert torch.cuda.is_available()
print('Training GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__, flush=True)
PY

if [[ ! -f output/base-validation-summary.json ]]; then
    if [[ ! -f output/probe-validation-summary.json ]]; then
        "$PY" -u train_curve.py --phase probe 2>&1 | tee output/logs/probe-validation.log
    fi
    "$PY" -u train_curve.py --phase baseline 2>&1 | tee output/logs/base-validation.log
fi
for size in 128 512 2048 4096; do
    if [[ ! -f "output/lora-$size-validation-summary.json" ]]; then
        "$PY" -u train_curve.py --phase train --size "$size" 2>&1 | tee "output/logs/train-$size.log"
    fi
done
"$PY" -u report.py validation 2>&1 | tee output/logs/selection.log
