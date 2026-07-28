#!/bin/bash
set -u
source venv_ae311/bin/activate
for ds in attention_check moral_data racial_data inattentive bot_bot_mturk pennycook_1 public_opinion mturk_ethics; do
  python -m experiment_b.multiseed_ae "$ds" 5 >/tmp/ms_${ds}.log 2>&1 && echo "$ds multiseed DONE" || echo "$ds FAILED: $(tail -1 /tmp/ms_${ds}.log)"
done
echo "ALL MULTISEED DONE"
