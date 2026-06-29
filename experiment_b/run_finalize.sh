#!/bin/bash
set -u
source venv_ae311/bin/activate
for ds in attention_check moral_data racial_data inattentive bot_bot_mturk pennycook_1 public_opinion mturk_ethics; do
  echo "=== finalize $ds ($(date +%H:%M:%S)) ==="
  python -m experiment_b.finalize_models "$ds" > "/tmp/final_${ds}.log" 2>&1 && echo "  $ds DONE" || echo "  $ds FAILED: $(tail -2 /tmp/final_${ds}.log | head -1)"
done
echo "=== ALL FINALIZE COMPLETE ($(date +%H:%M:%S)) ==="
