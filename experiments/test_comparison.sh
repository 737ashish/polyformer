#!/bin/bash

# Add root directory to Python path (this is the key part)
export PYTHONPATH="/home/ashishkangen/Projects/polyformer:$PYTHONPATH"

# Configuration for quick local testing
EMBEDDING=64  # Smaller embedding size for faster training
HEADS=4       # Fewer attention heads
DEPTH=2       # Just 2 transformer blocks
EPOCHS=1      # Single epoch
BATCH=8       # Small batch size

echo "=== Testing Standard Transformer ==="
python3 -c "import sys; print(sys.path)" # Debug Python path
python3 experiments/classify.py \
  --embedding $EMBEDDING --heads $HEADS --depth $DEPTH \
  --num-epochs $EPOCHS --batch-size $BATCH \
  --tb_dir ./runs/test_standard_transformer

echo -e "\n\n=== Testing Polynomial Transformer with CP ==="
python3 experiments/poly_classify.py \
  --embedding $EMBEDDING --heads $HEADS --depth $DEPTH \
  --num-epochs $EPOCHS --batch-size $BATCH \
  --degree 2 --poly-class CP_sparse_degree_LU \
  --tb_dir ./runs/test_polynomial_CP

echo -e "\n\nTest complete. Check TensorBoard for results:"
echo "tensorboard --logdir=./runs"
