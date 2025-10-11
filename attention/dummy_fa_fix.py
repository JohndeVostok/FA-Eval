import torch
from torch.autograd import Function
import math

lp_type = torch.bfloat16
hp_type = torch.float32

class DummyFAFixed(Function):
    hp_o = None
    storage = []

    @staticmethod
    def forward_func(q, k, v, num_group, scale, dropout_p, mask):
        if num_group > 1:
            k = k.repeat_interleave(num_group, dim=1)
            v = v.repeat_interleave(num_group, dim=1)
        s = torch.matmul(q, k.transpose(-2, -1))
        s = s.to(hp_type) / scale
        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))
        m = s.max(dim=-1, keepdim=True)[0]
        p = torch.exp(s - m)
        l = p.sum(dim=-1, keepdim=True)
        p = p.to(lp_type)
        L = m + torch.log(l)
        o = (torch.matmul(p, v) / (l * (1 - dropout_p))).to(lp_type)
        return o, L
    
    @staticmethod
    def backward_func(do, q, k, v, o, L, num_group, scale, dropout_p, mask):
        if num_group > 1:
            k = k.repeat_interleave(num_group, dim=1)
            v = v.repeat_interleave(num_group, dim=1)
        s = torch.matmul(q, k.transpose(-2, -1))
        s = s.to(hp_type) / scale
        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))
        p = torch.exp(s - L).to(lp_type)
        dv = torch.matmul(p.transpose(-2, -1), do) / (1 - dropout_p)
        dp = torch.matmul(do, v.transpose(-2, -1))
        D = torch.sum(do * o, dim=-1, keepdim=True) * (1 - dropout_p)
        DummyFAFixed.storage.append(D)
        ds = (p * (dp - D)).to(torch.bfloat16)
        dq = (torch.matmul(ds, k) / (scale * (1 - dropout_p))).to(lp_type)
        dk = (torch.matmul(ds.transpose(-2, -1), q) / (scale * (1 - dropout_p))).to(lp_type)

        b, n, s, h = q.shape
        if num_group > 1:
            dk = dk.reshape(b, n // num_group, num_group, s, h).sum(dim=2)
            dv = dv.reshape(b, n // num_group, num_group, s, h).sum(dim=2)
        return dq, dk, dv

    @staticmethod
    def forward(ctx, q, k, v, num_q_heads, num_kv_heads, dropout_p, training, mask):
        embed_dim = q.shape[-1]
        head_dim = embed_dim // num_q_heads
        num_groups = num_q_heads // num_kv_heads
        scale = math.sqrt(head_dim)
        batch_size, seq_len, _ = q.shape

        q_reshaped = q.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
        k_reshaped = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v_reshaped = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        output_reshaped, L = DummyFAFixed.forward_func(q_reshaped, k_reshaped, v_reshaped, num_groups, scale, dropout_p, mask)

        output = output_reshaped.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        ctx.save_for_backward(q_reshaped, k_reshaped, v_reshaped, output_reshaped, L)
        ctx.scale = scale
        ctx.num_groups = num_groups
        ctx.dropout_p = dropout_p
        ctx.training = training
        ctx.mask = mask
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # --- Unpack saved tensors and parameters ---
        q_reshaped, k_reshaped, v_reshaped, output_reshaped, L = ctx.saved_tensors
        scale = ctx.scale
        num_groups = ctx.num_groups
        dropout_p = ctx.dropout_p
        training = ctx.training
        mask = ctx.mask
        batch_size, num_q_heads, seq_len, head_dim = q_reshaped.shape
        grad_output = grad_output.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
        if DummyFAFixed.hp_o is not None:
            output_reshaped = DummyFAFixed.hp_o.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
            print("use hp cpu output.")

        dq, dk, dv = DummyFAFixed.backward_func(grad_output, q_reshaped, k_reshaped, v_reshaped, output_reshaped, L, num_groups, scale, dropout_p, mask)

        dq = dq.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        dk = dk.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        dv = dv.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return dq, dk, dv, None, None, None, None, None