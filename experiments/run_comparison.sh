#!/bin/bash

# Configure experiment parameters
EMBEDDING=128
HEADS=8
DEPTH=6
EPOCHS=20
BATCH=32

# Run standard transformer
python experiments/classify.py \
  --embedding $EMBEDDING --heads $HEADS --depth $DEPTH \
  --num-epochs $EPOCHS --batch-size $BATCH \
  --tb_dir ./runs/standard_transformer

# Run polynomial transformer with CP
python experiments/poly_classify.py \
  --embedding $EMBEDDING --heads $HEADS --depth $DEPTH \
  --num-epochs $EPOCHS --batch-size $BATCH \
  --degree 2 --poly-class CP \
  --tb_dir ./runs/polynomial_CP

# Run polynomial transformer with CP_sparse_LU
python experiments/poly_classify.py \
  --embedding $EMBEDDING --heads $HEADS --depth $DEPTH \
  --num-epochs $EPOCHS --batch-size $BATCH \
  --degree 2 --poly-class CP_sparse_LU \
  --tb_dir ./runs/polynomial_CP_sparse_LU
