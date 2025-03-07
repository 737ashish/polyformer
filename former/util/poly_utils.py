import torch

def D_U_d(U, d):
    """
    Extract the d-th column from weight matrix U.
    Used for computing derivatives in polynomial networks.
    
    Args:
        U: Weight matrix
        d: Index of column to extract
        
    Returns:
        Column vector from the weight matrix
    """
    return U[:,d]

def generate_masks(degree, rank, in_dim):
    """
    Generate masks for sparse polynomial networks.
    Used by CP_sparse_degree and CP_sparse_degree_LU classes.
    
    Args:
        degree: Polynomial degree
        rank: Rank of the decomposition
        in_dim: Input dimension
        
    Returns:
        List of mask tensors
    """
    if rank < degree:
        Masks = [torch.ones(rank, in_dim)] * (degree - rank + 1)
        for i in range(1, rank):
            M = torch.ones(rank, in_dim)
            r = torch.arange(rank - i, rank) 
            M[r] = 0
            Masks = Masks + [M]
    else:        
        Masks = [torch.ones(rank, in_dim)]
        steps = []
        for i in range(0, degree - 1):
            M = torch.ones(rank, in_dim)
            r = torch.arange(i, rank, degree) 
            steps = steps + [r]
            M[torch.cat(steps, 0)] = 0
            Masks = Masks + [M]
    return Masks

def non_zero_count(model):
    num_param = []
    for name, param in model.named_parameters():
        num = torch.count_nonzero(param)
        num_param.append(num)

    count = torch.sum(torch.tensor(num_param))    
    return count

