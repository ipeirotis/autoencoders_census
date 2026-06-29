#!/bin/bash
set -u
source venv_ae311/bin/activate
for ds in attention_check moral_data racial_data inattentive bot_bot_mturk pennycook_1 public_opinion mturk_ethics; do
  python main.py train --model_name PCA --data "$ds" --config config/linear_autoencoder.yaml --output "cache/_lin_${ds}/" --seed 2 >/tmp/lin_${ds}.log 2>&1 \
    && python main.py find_outliers --model_path "cache/_lin_${ds}/autoencoder" --data "$ds" --output "cache/_lin_${ds}/" --seed 2 >>/tmp/lin_${ds}.log 2>&1 \
    && echo "$ds linear DONE" || echo "$ds linear FAILED: $(tail -1 /tmp/lin_${ds}.log)"
done
echo "ALL LINEAR DONE"
