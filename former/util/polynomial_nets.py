import torch
import torch.nn as nn
from . import poly_utils as ut


class CP(nn.Module):
    def __init__(self, degree, d, k, o):
        super(CP, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        for i in range(1, self.degree + 1):
            setattr(self, 'U{}'.format(i), nn.Linear(self.input_dimension, self.rank, bias=False)) 

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 


    def forward(self, z):
        # Store original shape information
        original_shape = z.shape[:-1]  # Store batch and sequence dimensions    
        z = z.view(-1, self.input_dimension)
        out = self.U1(z)
        for i in range(2, self.degree + 1):
            out = getattr(self, 'U{}'.format(i))(z) * out + out
        x = self.layer_C(out)
        x = x.view(*original_shape, self.output_dimension)        
        return x
    
    def derivative(self, z, d):
        z = z.reshape(-1, self.input_dimension)
        out = self.U1(z)
        dout = ut.D_U_d(self.U1.weight.data, d)
        for i in range(2, self.degree + 1):            
            dout = ut.D_U_d(getattr(self, 'U{}'.format(i)).weight.data, d) * out + getattr(self, 'U{}'.format(i))(z) * dout + dout
            out = getattr(self, 'U{}'.format(i))(z) * out + out        
        x = torch.matmul(dout, self.layer_C.weight.data.T)
        return x
    
class CP_sparse_LU(nn.Module):
    def __init__(self, degree, d, k, o, l_offset = 0, u_offset = 0):
        super(CP_sparse_LU, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        if self.input_dimension > self.rank:
            self.register_buffer('mask1', torch.triu(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = u_offset))
            self.register_buffer('mask2', torch.tril(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = l_offset))
            for i in range(1, self.degree + 1, 2):
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.triu(nn.Linear(d, k, bias=False).weight, diagonal = u_offset))) 
            for i in range(2, self.degree + 1, 2):
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.tril(nn.Linear(d, k, bias=False).weight, diagonal = l_offset)))  
        else:
            self.register_buffer('mask1', torch.tril(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = l_offset))
            self.register_buffer('mask2', torch.triu(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = u_offset))

            for i in range(1, self.degree + 1, 2):
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.tril(nn.Linear(d, k, bias=False).weight, diagonal = l_offset))) 
            for i in range(2, self.degree + 1, 2):
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.triu(nn.Linear(d, k, bias=False).weight, diagonal = u_offset)))         

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 


    def forward(self, z):
        original_shape = z.shape[:-1]
        z = z.reshape(-1, self.input_dimension)
        out = torch.matmul(z, (self.mask1 * self.U1).T)
        for i in range(2, self.degree + 1, 2):
            out = torch.matmul(z, (self.mask2 * getattr(self, 'U{}'.format(i))).T) * out + out
            if i == self.degree:
                x = self.layer_C(out)
                x = x.view(*original_shape, self.output_dimension)
                return x
            out = torch.matmul(z, (self.mask1 * getattr(self, 'U{}'.format(i+1))).T) * out + out
        x = self.layer_C(out)
        x = x.view(*original_shape, self.output_dimension)
        return x

class CP_sparse_degree(nn.Module):
    def __init__(self, degree, d, k, o):
        super(CP_sparse_degree, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        masks = ut.generate_masks(self.degree, self.rank, self.input_dimension)

        for i in range(1, self.degree + 1):
            setattr(self, 'U{}'.format(i), nn.Parameter(nn.Linear(d, k, bias=False).weight * masks[i-1]))
            self.register_buffer('mask{}'.format(i), masks[i-1])        

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 


    def forward(self, z):
        original_shape = z.shape[:-1]
        z = z.reshape(-1, self.input_dimension)
        out = torch.matmul(z, (self.mask1 * self.U1).T)
        for i in range(2, self.degree + 1):
            out = torch.matmul(z, (getattr(self, 'mask{}'.format(i)) * getattr(self, 'U{}'.format(i))).T) * out + out
        x = self.layer_C(out)
        x = x.view(*original_shape, self.output_dimension)
        return x
    

class CP_sparse_degree_LU(nn.Module):
    def __init__(self, degree, d, k, o, l_offset = 0, u_offset = 0):
        super(CP_sparse_degree_LU, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        #u_offset = int(d/4)
        #l_offset = int(d/4)
        masks = ut.generate_masks(self.degree, self.rank, self.input_dimension)

        if self.input_dimension > self.rank:
            self.register_buffer('mask_1', torch.triu(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = u_offset))
            self.register_buffer('mask_2', torch.tril(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = l_offset))
            for i in range(1, self.degree + 1, 2):
                self.register_buffer('mask{}'.format(i), masks[i-1] * self.mask_1) 
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.triu(nn.Linear(d, k, bias=False).weight, diagonal = u_offset) * getattr(self, 'mask{}'.format(i)))) 
            for i in range(2, self.degree + 1, 2):
                self.register_buffer('mask{}'.format(i), masks[i-1] * self.mask_2) 
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.tril(nn.Linear(d, k, bias=False).weight, diagonal = l_offset)* getattr(self, 'mask{}'.format(i)))) 
        else:
            self.register_buffer('mask_1', torch.tril(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = l_offset))
            self.register_buffer('mask_2', torch.triu(torch.ones_like(nn.Linear(d, k, bias=False).weight), diagonal = u_offset))
            for i in range(1, self.degree + 1, 2):
                self.register_buffer('mask{}'.format(i), masks[i-1] * self.mask_1) 
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.tril(nn.Linear(d, k, bias=False).weight, diagonal = l_offset) * getattr(self, 'mask{}'.format(i))))
            for i in range(2, self.degree + 1, 2):  
                self.register_buffer('mask{}'.format(i), masks[i-1] * self.mask_2) 
                setattr(self, 'U{}'.format(i), nn.Parameter(torch.triu(nn.Linear(d, k, bias=False).weight, diagonal = u_offset)* getattr(self, 'mask{}'.format(i))))
        

        self.layer_C = nn.Linear(self.rank, self.output_dimension) 
    
    def forward(self, z):
        original_shape = z.shape[:-1]
        z = z.reshape(-1, self.input_dimension)
        out = torch.matmul(z, (self.mask1 * self.U1).T)
        for i in range(2, self.degree + 1, 2):
            out = torch.matmul(z, (getattr(self, 'mask{}'.format(i)) * getattr(self, 'U{}'.format(i))).T) * out + out
            if i == self.degree:
                x = self.layer_C(out)
                x = x.view(*original_shape, self.output_dimension)
                return x
            out = torch.matmul(z, (getattr(self, 'mask{}'.format(i+1)) * getattr(self, 'U{}'.format(i+1))).T) * out + out
        x = self.layer_C(out)
        x = x.view(*original_shape, self.output_dimension)
        return x

class CP_sparse_LU_sawtooth(nn.Module):
    def __init__(self, degree, d, k, o, l_offset=0, u_offset=0):
        super(CP_sparse_LU_sawtooth, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        
        # Create sawtooth masks that repeat the triangular pattern
        mask1, mask2 = self._create_sawtooth_masks_efficient(d, k, l_offset, u_offset)
        self.register_buffer('mask1', mask1)
        self.register_buffer('mask2', mask2)
        
        # Initialize parameters with appropriate masks
        for i in range(1, self.degree + 1, 2):
            weight = nn.Linear(d, k, bias=False).weight
            setattr(self, 'U{}'.format(i), nn.Parameter(weight * mask1))
            
        for i in range(2, self.degree + 1, 2):
            weight = nn.Linear(d, k, bias=False).weight
            setattr(self, 'U{}'.format(i), nn.Parameter(weight * mask2))

        self.layer_C = nn.Linear(self.rank, self.output_dimension)
        
    def _create_sawtooth_masks_efficient(self, d, k, l_offset, u_offset):
        """Efficiently create repeating L/U pattern masks along the elongated dimension"""
        # Determine which dimension is smaller
        min_dim = min(d, k)
        
        # Create base L and U patterns for the square part
        L_base = torch.tril(torch.ones(min_dim, min_dim), diagonal=l_offset)
        U_base = torch.triu(torch.ones(min_dim, min_dim), diagonal=u_offset)
        
        if d >= k:  # Input dimension is longer - repeat patterns horizontally
            # Create masks of the right shape
            mask1 = torch.zeros(k, d)
            mask2 = torch.zeros(k, d)
            
            # Calculate how many complete blocks we need
            num_complete_blocks = d // k
            
            # Handle complete blocks with efficient tensor operations
            if num_complete_blocks > 0:
                # Create repeated U patterns for mask1
                mask1_blocks = U_base.repeat(1, num_complete_blocks)
                mask1[:, :k*num_complete_blocks] = mask1_blocks
                
                # Create repeated L patterns for mask2
                mask2_blocks = L_base.repeat(1, num_complete_blocks)
                mask2[:, :k*num_complete_blocks] = mask2_blocks
            
            # Handle remainder efficiently
            remainder = d % k
            if remainder > 0:
                mask1[:, k*num_complete_blocks:] = U_base[:, :remainder]
                mask2[:, k*num_complete_blocks:] = L_base[:, :remainder]
            
        else:  # Rank dimension is longer - repeat patterns vertically
            # Create masks of the right shape
            mask1 = torch.zeros(k, d)
            mask2 = torch.zeros(k, d)
            
            # Calculate how many complete blocks we need
            num_complete_blocks = k // d
            
            # Handle complete blocks with efficient tensor operations
            if num_complete_blocks > 0:
                # Create repeated U patterns for mask1
                mask1_blocks = U_base.repeat(num_complete_blocks, 1)
                mask1[:d*num_complete_blocks, :] = mask1_blocks
                
                # Create repeated L patterns for mask2
                mask2_blocks = L_base.repeat(num_complete_blocks, 1)
                mask2[:d*num_complete_blocks, :] = mask2_blocks
            
            # Handle remainder efficiently
            remainder = k % d
            if remainder > 0:
                mask1[d*num_complete_blocks:, :] = U_base[:remainder, :]
                mask2[d*num_complete_blocks:, :] = L_base[:remainder, :]
        
        return mask1, mask2
        
    def forward(self, z):
        # ...existing code...
        original_shape = z.shape[:-1]
        z = z.reshape(-1, self.input_dimension)
        out = torch.matmul(z, (self.mask1 * self.U1).T)
        for i in range(2, self.degree + 1, 2):
            out = torch.matmul(z, (self.mask2 * getattr(self, 'U{}'.format(i))).T) * out + out
            if i == self.degree:
                x = self.layer_C(out)
                x = x.view(*original_shape, self.output_dimension)
                return x
            out = torch.matmul(z, (self.mask1 * getattr(self, 'U{}'.format(i+1))).T) * out + out
        x = self.layer_C(out)
        x = x.view(*original_shape, self.output_dimension)
        return x

class CP_sparse_degree_LU_sawtooth(nn.Module):
    def __init__(self, degree, d, k, o, l_offset = 0, u_offset = 0):
        super(CP_sparse_degree_LU_sawtooth, self).__init__()        
     
        self.input_dimension = d 
        self.rank = k
        self.output_dimension = o 
        self.degree = int(degree)
        
        # Generate degree-specific masks
        masks = ut.generate_masks(self.degree, self.rank, self.input_dimension)
        
        # Create sawtooth masks
        mask1, mask2 = self._create_sawtooth_masks_efficient(d, k, l_offset, u_offset)
        self.register_buffer('mask_1', mask1)
        self.register_buffer('mask_2', mask2)
        
        # Apply degree-specific masks combined with sawtooth patterns
        for i in range(1, self.degree + 1, 2):
            self.register_buffer('mask{}'.format(i), masks[i-1] * self.mask_1)
            weight = nn.Linear(d, k, bias=False).weight
            setattr(self, 'U{}'.format(i), nn.Parameter(weight * getattr(self, 'mask{}'.format(i))))
            
        for i in range(2, self.degree + 1, 2):
            self.register_buffer('mask{}'.format(i), masks[i-1] * self.mask_2)
            weight = nn.Linear(d, k, bias=False).weight
            setattr(self, 'U{}'.format(i), nn.Parameter(weight * getattr(self, 'mask{}'.format(i))))

        self.layer_C = nn.Linear(self.rank, self.output_dimension)
        
    def _create_sawtooth_masks_efficient(self, d, k, l_offset, u_offset):
        """Efficiently create repeating L/U pattern masks along the elongated dimension"""
        # Determine which dimension is smaller
        min_dim = min(d, k)
        
        # Create base L and U patterns for the square part
        L_base = torch.tril(torch.ones(min_dim, min_dim), diagonal=l_offset)
        U_base = torch.triu(torch.ones(min_dim, min_dim), diagonal=u_offset)
        
        if d >= k:  # Input dimension is longer - repeat patterns horizontally
            # Create masks of the right shape
            mask1 = torch.zeros(k, d)
            mask2 = torch.zeros(k, d)
            
            # Calculate how many complete blocks we need
            num_complete_blocks = d // k
            
            # Handle complete blocks with efficient tensor operations
            if num_complete_blocks > 0:
                # Create repeated U patterns for mask1
                mask1_blocks = U_base.repeat(1, num_complete_blocks)
                mask1[:, :k*num_complete_blocks] = mask1_blocks
                
                # Create repeated L patterns for mask2
                mask2_blocks = L_base.repeat(1, num_complete_blocks)
                mask2[:, :k*num_complete_blocks] = mask2_blocks
            
            # Handle remainder efficiently
            remainder = d % k
            if remainder > 0:
                mask1[:, k*num_complete_blocks:] = U_base[:, :remainder]
                mask2[:, k*num_complete_blocks:] = L_base[:, :remainder]
            
        else:  # Rank dimension is longer - repeat patterns vertically
            # Create masks of the right shape
            mask1 = torch.zeros(k, d)
            mask2 = torch.zeros(k, d)
            
            # Calculate how many complete blocks we need
            num_complete_blocks = k // d
            
            # Handle complete blocks with efficient tensor operations
            if num_complete_blocks > 0:
                # Create repeated U patterns for mask1
                mask1_blocks = U_base.repeat(num_complete_blocks, 1)
                mask1[:d*num_complete_blocks, :] = mask1_blocks
                
                # Create repeated L patterns for mask2
                mask2_blocks = L_base.repeat(num_complete_blocks, 1)
                mask2[:d*num_complete_blocks, :] = mask2_blocks
            
            # Handle remainder efficiently
            remainder = k % d
            if remainder > 0:
                mask1[d*num_complete_blocks:, :] = U_base[:remainder, :]
                mask2[d*num_complete_blocks:, :] = L_base[:remainder, :]
        
        return mask1, mask2
    
    def forward(self, z):
        original_shape = z.shape[:-1]
        z = z.reshape(-1, self.input_dimension)
        out = torch.matmul(z, (self.mask1 * self.U1).T)
        for i in range(2, self.degree + 1, 2):
            out = torch.matmul(z, (getattr(self, 'mask{}'.format(i)) * getattr(self, 'U{}'.format(i))).T) * out + out
            if i == self.degree:
                x = self.layer_C(out)
                x = x.view(*original_shape, self.output_dimension)
                return x
            out = torch.matmul(z, (getattr(self, 'mask{}'.format(i+1)) * getattr(self, 'U{}'.format(i+1))).T) * out + out
        x = self.layer_C(out)
        x = x.view(*original_shape, self.output_dimension)
        return x