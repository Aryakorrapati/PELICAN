#!/bin/bash

MAX_LR=$1

python3 train_pelican_classifier.py \
  --datadir=./data/sample_data/run12 \
  --yaml=./config/48k.yaml \
  --target=is_signal \
  --batch-size=64 \
  --prefix=classifier_lr${MAX_LR} \
  --max_lr=$MAX_LR \
  --mix_mode=mixup
