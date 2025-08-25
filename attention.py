import torch
import torch.nn as nn
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) a practical implementation in PyTorch.
    
    This module allows multiple query heads to share the same key and value heads,
    offering a balance between the performance of Multi-Query Attention (MQA)
    and the quality of Multi-Head Attention (MHA).

    Args:
        embed_dim (int): The total embedding dimension of the input.
        num_q_heads (int): The total number of query heads.
        num_kv_heads (int): The total number of key/value heads. 
                           num_q_heads must be divisible by num_kv_heads.
        dropout (float, optional): Dropout probability for the attention scores. Defaults to 0.0.
    """
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, dropout: float = 0.0, device='cpu', dtype=torch.float32, fwd_dropout_pos: int = 0):
        super().__init__()
        
        # --- Parameter Validation ---
        if embed_dim % num_q_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_q_heads ({num_q_heads})")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.fwd_dropout_pos = fwd_dropout_pos
        
        # Head dimension is determined by the query heads
        self.head_dim = embed_dim // num_q_heads
        
        # The number of query groups that share a single K/V head
        self.num_groups = num_q_heads // num_kv_heads

        # --- Linear Projections ---
        # Query projection layer (total dimension is embed_dim)
        # Key and Value projection layers (total dimension is smaller due to fewer heads)
        # Output projection layer

    def forward(self, q, k, v: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for GQA.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, seq_len, seq_len)
                                           or a broadcastable shape. A value of 0 in the mask
                                           means the position is masked out. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = q.shape

        # 1. Project to Q, K, V
        # Q: (batch, seq_len, num_q_heads * head_dim)
        # K, V: (batch, seq_len, num_kv_heads * head_dim)
        assert q.shape == (batch_size, seq_len, self.embed_dim)
        assert k.shape == (batch_size, seq_len, self.embed_dim // self.num_groups)
        assert v.shape == (batch_size, seq_len, self.embed_dim // self.num_groups)
        
        # 2. Reshape Q, K, V for multi-head computation
        # q -> (batch, num_q_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2).to(self.device, dtype=self.dtype)
        
        # k -> (batch, num_kv_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).to(self.device, dtype=self.dtype)
        
        # v -> (batch, num_kv_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).to(self.device, dtype=self.dtype)

        # 3. Repeat K and V to match the number of Q heads (The core of GQA)
        # This is a memory-efficient way to handle the groups. 
        # Instead of creating num_q_heads independent K/V heads, we repeat them.
        # k: (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_q_heads, seq_len, head_dim)
        # v: (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_q_heads, seq_len, head_dim)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # 4. Scaled Dot-Product Attention
        # (batch, num_q_heads, seq_len, head_dim) @ (batch, num_q_heads, head_dim, seq_len)
        # -> (batch, num_q_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # The mask should be broadcastable to the scores tensor.
            # A common mask shape is (1, 1, seq_len, seq_len) for causal masking.
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        if self.fwd_dropout_pos == 0:
            attn_weights /= self.dropout
        # attn_weights = self.attn_dropout(attn_weights)
        
        # Apply weights to Value vectors
        # (batch, num_q_heads, seq_len, seq_len) @ (batch, num_q_heads, seq_len, head_dim)
        # -> (batch, num_q_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)

        if self.fwd_dropout_pos == 1:
            output /= self.dropout

        # 5. Concatenate heads and project output
        # Reshape to (batch, seq_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim).to('cuda', dtype=torch.float32)
        
        # Final linear projection
        # output = self.w_o(output)
        
        return output

### 使用示例 (Demo)

if __name__ == "__main__":
    # --- Configuration ---
    batch_size = 1
    seq_len =8192
    embed_dim = 2048
    num_q_heads = 16
    num_kv_heads = 2
    
    print(f"Initializing GroupedQueryAttention module with:")
    print(f"  - Embedding Dim: {embed_dim}")
    print(f"  - Query Heads: {num_q_heads}")
    print(f"  - Key/Value Heads: {num_kv_heads} (GQA)")
    print("-" * 30)

    attn_cpu = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.1,
        device="cpu",
        fwd_dropout_pos=1
    )

    attn_gpu_0 = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.1,
        device="cuda",
        dtype=torch.bfloat16,
        fwd_dropout_pos=0
    )

    attn_gpu_1 = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.1,
        device="cuda",
        dtype=torch.bfloat16,
        fwd_dropout_pos=1
    )
    
    # Create a dummy input tensor on CPU
    q = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32, device='cuda')
    k = torch.randn(batch_size, seq_len, embed_dim // 8, dtype=torch.float32, device='cuda')
    v = torch.randn(batch_size, seq_len, embed_dim // 8, dtype=torch.float32, device='cuda')
    
    # --- Test Case 1: No Mask (Encoder-style) ---
    print("\n--- Test Case 1: No Mask (Encoder-style attention) ---")
    print(f"Input shape: {q.shape}")
    
    o_cpu = attn_cpu(q, k, v, mask=None)
    o_gpu_0 = attn_gpu_0(q, k, v, mask=None)
    o_gpu_1 = attn_gpu_1(q, k, v, mask=None)
    print(f"Output shape: {o_cpu.shape}")
    assert o_cpu.shape == (batch_size, seq_len, embed_dim)
    assert o_gpu_0.shape == (batch_size, seq_len, embed_dim)
    assert o_gpu_1.shape == (batch_size, seq_len, embed_dim)
    err_gpu_0 = max((abs(o_cpu - o_gpu_0) / (abs(o_cpu) + 1e-5)).view(-1))
    err_gpu_1 = max((abs(o_cpu - o_gpu_1) / (abs(o_cpu) + 1e-5)).view(-1))
    print(err_gpu_0.item())
    print(err_gpu_1.item())
    assert err_gpu_0.item() < 1e-2 and err_gpu_1.item() < 1e-2
    print("Test Case 1 Passed!")
    
    print("\n--- Test Case 2: Causal Mask (Decoder-style attention) ---")
    # Create a causal mask to prevent attention to future tokens
    # Shape: (1, 1, seq_len, seq_len)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
    
    print(f"Input shape: {q.shape}")
    print(f"Causal mask shape: {causal_mask.shape}")
    
    o_cpu = attn_cpu(q, k, v, mask=causal_mask)
    o_gpu_0 = attn_gpu_0(q, k, v, mask=causal_mask)
    o_gpu_1 = attn_gpu_1(q, k, v, mask=causal_mask)

    
    print(f"Output shape: {o_cpu.shape}")
    assert o_cpu.shape == (batch_size, seq_len, embed_dim)
    assert o_gpu_0.shape == (batch_size, seq_len, embed_dim)
    assert o_gpu_1.shape == (batch_size, seq_len, embed_dim)
    err_gpu_0 = max((abs(o_cpu - o_gpu_0) / (abs(o_cpu) + 1e-5)))
    err_gpu_1 = max((abs(o_cpu - o_gpu_1) / (abs(o_cpu) + 1e-5)))
    print(err_gpu_0)
    print(err_gpu_1)
    assert err_gpu_0 < 1e-2 and err_gpu_1 < 1e-2
    print("Test Case 2 Passed!")
