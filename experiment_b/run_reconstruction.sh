#!/usr/bin/env bash
# Reproduce the full reconstruction table, one (dataset, method) per process so
# TensorFlow state never accumulates across builds. Resumable: skips cells whose
# CSV already exists. Run from the repo root:  bash experiment_b/run_reconstruction.sh
set -u
PY=./venv_ae/bin/python
DATASETS="sadc_2017 pennycook_1 inattentive attention_check moral_data bot_bot_mturk mturk_ethics public_opinion racial_data"
METHODS="nl100 nl85 lin"
mkdir -p reconstruction_cells
for ds in $DATASETS; do
  for m in $METHODS; do
    out="reconstruction_cells/${ds}__${m}.csv"
    if [ -f "$out" ]; then
      echo "SKIP $ds/$m (exists)"
      continue
    fi
    echo "=== RUN $ds/$m ==="
    $PY -m experiment_b.reconstruction_table "$ds" "$m" 2>&1 \
      | grep -E "^\[$ds" || echo "!! $ds/$m produced no result line"
  done
done
echo "=== COMBINE ==="
$PY -m experiment_b.reconstruction_table combine 2>&1 | grep -vE "importlib|packages_distributions|warnings.warn"
