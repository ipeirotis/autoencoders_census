#!/bin/bash
# Per-dataset AE hyperparameter search + final training (paper protocol,
# compute-reduced). Sequential so a single TF process doesn't thrash; coherent-
# battery datasets first. Logs per dataset under /tmp/tune_<ds>.log.
set -u
source venv_ae311/bin/activate
DATASETS="attention_check moral_data racial_data inattentive bot_bot_mturk pennycook_1 public_opinion mturk_ethics"
for ds in $DATASETS; do
  echo "=== tuning $ds  ($(date +%H:%M:%S)) ==="
  python tune_one.py "$ds" config/hp_autoencoder_real.yaml > "/tmp/tune_${ds}.log" 2>&1 \
    && echo "  $ds DONE: $(grep -h 'tuned+scored' /tmp/tune_${ds}.log | tail -1)" \
    || echo "  $ds FAILED: $(tail -2 /tmp/tune_${ds}.log | head -1)"
done
echo "=== ALL TUNING COMPLETE ($(date +%H:%M:%S)) ==="
