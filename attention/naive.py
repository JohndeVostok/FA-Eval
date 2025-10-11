import torch
from torch.autograd import Function
import math

class NaiveAttn:
    @staticmethod
    def apply(q, k, v, num_q_heads, num_kv_heads, dropout_p, training, mask):
        # --- Pre-computation for backward pass ---
        embed_dim = q.shape[-1]
        head_dim = embed_dim // num_q_heads
        num_groups = num_q_heads // num_kv_heads
        scale = math.sqrt(head_dim)
        batch_size, seq_len, _ = q.shape
        kv_embed_dim = num_kv_heads * head_dim

        # --- Reshape and Repeat ---
        q_reshaped = q.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
        k_reshaped = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v_reshaped = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        if num_groups > 1:
            k_repeated = k_reshaped.repeat_interleave(num_groups, dim=1)
            v_repeated = v_reshaped.repeat_interleave(num_groups, dim=1)
        else:
            k_repeated = k_reshaped
            v_repeated = v_reshaped

        attn_scores = torch.matmul(q_reshaped, k_repeated.transpose(-2, -1)) / scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output_reshaped = torch.matmul(attn_weights, v_repeated)

        if training and dropout_p > 0.0:
            output_reshaped = output_reshaped / (1 - dropout_p)
        
        output = output_reshaped.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return output