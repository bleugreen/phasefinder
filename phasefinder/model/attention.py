import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.pos_encoding import PositionalEncoding

class AttentionModule(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(AttentionModule, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Multi-head attention
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        # Output projection
        self.proj = nn.Linear(input_dim, input_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(input_dim, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Compute Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)
        
        output = self.proj(context)
        return output
