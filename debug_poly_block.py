import torch
from former.modules import PolyTransformerBlock
from former.util import CP

# Configuration parameters - same as in your test.ipynb
emb = 64          # Embedding dimension
heads = 4         # Number of attention heads
seq_length = 32   # Sequence length
batch_size = 8    # Batch size

# Create a PolyTransformerBlock with verbose output during initialization
print("Creating PolyTransformerBlock...")
poly_block = PolyTransformerBlock(
    emb=emb,
    heads=heads,
    mask=True, 
    seq_length=seq_length,
    degree=2,
    poly_class=CP,
    use_relu=True,
    ff_hidden_mult=4  # Explicitly set to default value
)

# Generate random input
print(f"Creating input tensor with shape: {(batch_size, seq_length, emb)}")
input_tensor = torch.randn(batch_size, seq_length, emb)

# Print shape information for debugging
print(f"Input tensor shape: {input_tensor.shape}")
for name, module in poly_block.named_modules():
    if hasattr(module, 'weight'):
        print(f"{name} weight shape: {module.weight.shape}")

# Forward pass (this is where the error occurs)
print("\nRunning forward pass...")
output = poly_block(input_tensor)
print(f"Output shape: {output.shape}")
print("Success!")
