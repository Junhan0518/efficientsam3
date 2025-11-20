#!/bin/bash
set -e

mkdir -p output/efficient_sam3_checkpoints

for cfg in es_rv_s es_rv_m es_rv_l es_tv_s es_tv_m es_tv_l es_ev_s es_ev_m es_ev_l; do
  echo "Converting $cfg..."
  if [ -f "output/stage1/${cfg}/ckpt_epoch_0.pth" ]; then
      python stage1/convert_stage1_weights.py \
        "output/stage1/${cfg}/ckpt_epoch_0.pth" \
        --sam3-ckpt sam3_checkpoints/sam3.pt \
        --output "output/efficient_sam3_checkpoints/${cfg}.pt" \
        --target-prefix "backbone.vision_backbone.student_encoder" \
        --replace-prefix "backbone.vision_backbone"
  else
      echo "Warning: Checkpoint for $cfg not found, skipping."
  fi
done
