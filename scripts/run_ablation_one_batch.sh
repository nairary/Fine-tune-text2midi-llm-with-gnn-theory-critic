#!/usr/bin/env bash
set -euo pipefail

python src/training/train_teacher.py -m --config-name ablation_one_batch \
  experiment.epochs=50,500 \
  model.use_hybrid_graph_scorer=false,true \
  model.pooling_mode=mean,mean_max \
  dataloader=graph_ablation,theory_aware_ablation \
  dataloader.batch_size=1,16
