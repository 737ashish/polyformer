from .util import mask_, d, here, contains_nan, tic, toc, \
    slice_diag, compute_compression, LOG2E, estimate_compression, enwik8, sample_batch, sample_sequence, sample, \
    enwik8_string, enwik8_bytes

# Import polynomial network classes
from .polynomial_nets import CP, CP_sparse_LU, CP_sparse_degree, CP_sparse_degree_LU, CP_sparse_LU_sawtooth, CP_sparse_degree_LU_sawtooth

# Import utility functions from poly_utils
from .poly_utils import (
    generate_masks, 
    D_U_d,
    non_zero_count
)
