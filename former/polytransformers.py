import torch
from torch import nn
import torch.nn.functional as F

from .modules import PolyTransformerBlock
from .transformers import GTransformer, CTransformer
from .util import d, CP

class PolyGTransformer(GTransformer):
    """
    Transformer for generating text (character by character) using polynomial networks in the feed-forward layers.
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, 
                 attention_type='default', degree=2, poly_class=CP, use_relu=False,
                 ff_hidden_mult=4, dropout=0.0):
        """
        :param emb: Embedding dimension
        :param heads: Number of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Maximum sequence length
        :param num_tokens: Number of tokens in the vocabulary
        :param attention_type: Type of attention to use
        :param degree: Degree of the polynomial networks
        :param poly_class: Which polynomial network class to use (CP, CP_sparse_LU, etc.)
        :param use_relu: Whether to use ReLU activation between polynomial networks
        :param ff_hidden_mult: Multiplier for hidden dimension in feed-forward network
        :param dropout: Dropout rate
        """
        # Initialize everything except transformer blocks
        super(GTransformer, self).__init__()  # Call nn.Module's init

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(
            embedding_dim=emb, 
            num_embeddings=(seq_length * 2 - 1 if attention_type=='relative' else seq_length)
        )

        # Create polynomial transformer blocks instead of regular transformer blocks
        tblocks = []
        for i in range(depth):
            tblocks.append(
                PolyTransformerBlock(
                    emb=emb, 
                    heads=heads, 
                    seq_length=seq_length, 
                    mask=True,
                    attention_type=attention_type,
                    pos_embedding=self.pos_embedding,
                    degree=degree,
                    poly_class=poly_class,
                    use_relu=use_relu,
                    ff_hidden_mult=ff_hidden_mult,
                    dropout=dropout
                )
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(emb, num_tokens)

    # No need to override forward method as it's the same as in GTransformer


class PolyCTransformer(CTransformer):
    """
    Transformer for classifying sequences using polynomial networks in the feed-forward layers.
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, 
                 max_pool=True, dropout=0.0, attention_type='default', 
                 degree=2, poly_class=CP, use_relu=False, ff_hidden_mult=4):
        """
        :param emb: Embedding dimension
        :param heads: Number of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Maximum sequence length
        :param num_tokens: Number of tokens in the vocabulary
        :param num_classes: Number of output classes
        :param max_pool: If true, use global max pooling in the last layer. If false, use mean pooling
        :param dropout: Dropout rate
        :param attention_type: Type of attention to use
        :param degree: Degree of the polynomial networks
        :param poly_class: Which polynomial network class to use (CP, CP_sparse_LU, etc.)
        :param use_relu: Whether to use ReLU activation between polynomial networks
        :param ff_hidden_mult: Multiplier for hidden dimension in feed-forward network
        """
        # Initialize everything except transformer blocks
        super(CTransformer, self).__init__()  # Call nn.Module's init

        self.num_tokens = num_tokens
        self.max_pool = max_pool

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        # Create polynomial transformer blocks instead of regular transformer blocks
        tblocks = []
        for i in range(depth):
            tblocks.append(
                PolyTransformerBlock(
                    emb=emb, 
                    heads=heads, 
                    seq_length=seq_length, 
                    mask=False,
                    attention_type=attention_type,
                    pos_embedding=self.pos_embedding,
                    degree=degree,
                    poly_class=poly_class,
                    use_relu=use_relu,
                    ff_hidden_mult=ff_hidden_mult,
                    dropout=dropout
                )
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(emb, num_classes)
        self.do = nn.Dropout(dropout)

    # No need to override forward method as it's the same as in CTransformer
