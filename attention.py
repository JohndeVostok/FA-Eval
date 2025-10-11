import torch
import torch.nn as nn
import math
from torch.autograd import Function
from attention import DummyFA, DummyFAFixed, DummyFAHP, NaiveAttn

def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits_f = logits.float()
    m = logits_f.max(dim=dim, keepdim=True).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    numer = torch.exp(logits_f - m)
    denom = numer.sum(dim=dim, keepdim=True).clamp_min(1e-20)
    probs_f = numer / denom
    return probs_f.to(logits.dtype)

class AttnFunc(Function):

    storage = []

    @staticmethod
    def forward(ctx, q, k, v, num_q_heads, num_kv_heads, dropout_p, training, mask):
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

        ctx.save_for_backward(q_reshaped, k_repeated, v_repeated, attn_weights)
        ctx.scale = scale
        ctx.num_groups = num_groups
        ctx.dropout_p = dropout_p
        ctx.head_dim = head_dim
        ctx.num_kv_heads = num_kv_heads
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: Gradient of the loss with respect to the output of the forward function.
                         Shape: (batch, seq_len, embed_dim)
        """
        # --- Unpack saved tensors and parameters ---
        q_reshaped, k_repeated, v_repeated, attn_weights = ctx.saved_tensors
        scale = ctx.scale
        num_groups = ctx.num_groups
        dropout_p = ctx.dropout_p
        head_dim = ctx.head_dim
        num_kv_heads = ctx.num_kv_heads

        batch_size, num_q_heads, seq_len, _ = q_reshaped.shape
        kv_embed_dim = num_kv_heads * head_dim

        # Reshape grad_output to match the shape of output_reshaped
        grad_output_reshaped = grad_output.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
        
        # --- 1. Gradient w.r.t. V and attn_weights_dropped ---
        grad_v_repeated = torch.matmul(attn_weights.transpose(-2, -1), grad_output_reshaped)
        tmp0 = attn_weights.transpose(-2, -1)
        tmp1 = grad_output_reshaped
        tmp2 = grad_v_repeated
        AttnFunc.storage.append((tmp0, tmp1, tmp2))

        if dropout_p > 0.0:
            grad_output_reshaped = grad_output_reshaped / (1 - dropout_p)

                # print(attn_weights.flatten()[0], grad_v_repeated.flatten()[0], grad_v_repeated.flatten()[0])
        grad_attn_weights = torch.matmul(grad_output_reshaped, v_repeated.transpose(-2, -1))

        # --- 3. Backward through Softmax ---
        # dL/dS = dL/dA * dA/dS = A * (dL/dA - sum(dL/dA * A))
        sum_grad_x_weights = torch.sum(grad_attn_weights * attn_weights, dim=-1, keepdim=True)
        grad_attn_scores = attn_weights * (grad_attn_weights - sum_grad_x_weights)

        # --- 4. Gradient w.r.t. Q and K ---
        # grad_attn_scores has been scaled by 1/scale in forward,
        # so we need to account for that in the backward pass.
        grad_q_reshaped = torch.matmul(grad_attn_scores, k_repeated) / scale
        grad_k_repeated = torch.matmul(grad_attn_scores.transpose(-2, -1), q_reshaped) / scale

        # --- 5. Propagate gradients back through repeat_interleave ---
        # For k and v, gradients from different query groups for the same kv_head must be summed up.
        if num_groups > 1:
            # Reshape to (batch, num_kv_heads, num_groups, seq_len, head_dim) and sum over the num_groups dim
            grad_k_reshaped = grad_k_repeated.view(batch_size, num_kv_heads, num_groups, seq_len, head_dim).sum(dim=2)
            grad_v_reshaped = grad_v_repeated.view(batch_size, num_kv_heads, num_groups, seq_len, head_dim).sum(dim=2)
        else:
            grad_k_reshaped = grad_k_repeated
            grad_v_reshaped = grad_v_repeated

        # --- 6. Reshape gradients to match original input shapes ---
        dq = grad_q_reshaped.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        dk = grad_k_reshaped.transpose(1, 2).contiguous().view(batch_size, seq_len, kv_embed_dim)
        dv = grad_v_reshaped.transpose(1, 2).contiguous().view(batch_size, seq_len, kv_embed_dim)

        # Return gradients for each input of the forward function
        # The order must match: q, k, v, num_q_heads, num_kv_heads, dropout_p, training, mask
        # Grads for non-Tensor inputs are None
        return dq, dk, dv, None, None, None, None, None


# 步骤 2: 修改 nn.Module to use the custom Function
class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, dropout: float = 0.0, func=AttnFunc):
        super().__init__()
        if embed_dim % num_q_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_q_heads ({num_q_heads})")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.func = func
        
        # No need for nn.Dropout layer anymore as it's handled in the Function
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        The forward pass now simply calls the apply method of our custom Function.
        """
        # Pass self.training to the function to control dropout behavior
        return self.func.apply(q, k, v, self.num_q_heads, self.num_kv_heads, self.dropout, self.training, mask)


if __name__ == "__main__":
    # --- Configuration ---
    batch_size = 1
    seq_len = 8192
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
        func=NaiveAttn,
    )

    attn_gpu_0 = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.1,
        func=DummyFA,
    )

    attn_gpu_1 = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.1,
        func=DummyFAFixed,
    )

    attn_gpu_2 = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.1,
        func=DummyFAHP,
    )

    attn_gpu_3 = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.1,
        func=DummyFAFixed,
    )

    attn_gpu_4 = GroupedQueryAttention(
        embed_dim=embed_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.1,
        func=DummyFAHP,
    )

    attn_cpu.train()
    attn_gpu_0.train()
    attn_gpu_1.train()
    attn_gpu_2.train()
    attn_gpu_3.train()
    attn_gpu_4.train()

    eval_type = torch.bfloat16
    baseline_type = torch.float32
    # Create a dummy input tensor on CPU
    q = torch.rand(batch_size, seq_len, embed_dim, dtype=eval_type, device='cuda', requires_grad=True)
    k = torch.rand(batch_size, seq_len, embed_dim // num_q_heads * num_kv_heads, dtype=eval_type, device='cuda', requires_grad=True)
    v = torch.rand(batch_size, seq_len, embed_dim // num_q_heads * num_kv_heads, dtype=eval_type, device='cuda', requires_grad=True)
    
    # --- Test Case 1: No Mask (Encoder-style) ---
    print("\n--- Test Case 1: No Mask (Encoder-style attention) ---")
    print(f"Input shape: {q.shape}")
    
    attn_cpu.zero_grad()
    attn_gpu_0.zero_grad()
    attn_gpu_1.zero_grad()
    attn_gpu_2.zero_grad()
    attn_gpu_3.zero_grad()
    attn_gpu_4.zero_grad()

    q_cpu, k_cpu, v_cpu = q.to(device='cpu', dtype=baseline_type).clone().detach().requires_grad_(True), k.to(device='cpu', dtype=baseline_type).clone().detach().requires_grad_(True), v.to(device='cpu', dtype=baseline_type).clone().detach().requires_grad_(True)
    q_gpu_0, k_gpu_0, v_gpu_0 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
    q_gpu_1, k_gpu_1, v_gpu_1 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
    q_gpu_2, k_gpu_2, v_gpu_2 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
    q_gpu_3, k_gpu_3, v_gpu_3 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
    q_gpu_4, k_gpu_4, v_gpu_4 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
    

    mask_cpu = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len).to('cpu')
    mask_gpu = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len).to('cuda')
    o_cpu = attn_cpu(q_cpu, k_cpu, v_cpu, mask=mask_cpu)
    o_gpu_0 = attn_gpu_0(q_gpu_0, k_gpu_0, v_gpu_0, mask=mask_gpu)
    o_gpu_1 = attn_gpu_1(q_gpu_1, k_gpu_1, v_gpu_1, mask=mask_gpu)
    o_gpu_2 = attn_gpu_2(q_gpu_2, k_gpu_2, v_gpu_2, mask=mask_gpu)
    o_gpu_3 = attn_gpu_3(q_gpu_3, k_gpu_3, v_gpu_3, mask=mask_gpu)
    o_gpu_4 = attn_gpu_4(q_gpu_4, k_gpu_4, v_gpu_4, mask=mask_gpu)

    do = torch.rand_like(o_cpu, dtype=eval_type)
    do_cpu = do.to(device='cpu', dtype=baseline_type)
    do_gpu_0 = do.to(device='cuda', dtype=eval_type)
    do_gpu_1 = do.to(device='cuda', dtype=eval_type)
    do_gpu_2 = do.to(device='cuda', dtype=eval_type)
    do_gpu_3 = do.to(device='cuda', dtype=eval_type)
    do_gpu_4 = do.to(device='cuda', dtype=eval_type)

    o_cpu.backward(gradient = do_cpu)
    o_gpu_0.backward(gradient = do_gpu_0)
    o_gpu_1.backward(gradient = do_gpu_1)
    o_gpu_2.backward(gradient = do_gpu_2)
    DummyFAFixed.hp_o = o_cpu.cuda().to(eval_type)
    o_gpu_3.backward(gradient = do_gpu_3)
    DummyFAHP.hp_o = o_cpu.cuda().to(baseline_type)
    o_gpu_4.backward(gradient = do_gpu_4)
    

    print(f"Output shape: {o_cpu.shape}")
    err_o_gpu_0 = (abs(o_cpu.cuda() - o_gpu_0)).view(-1).max().item()
    err_o_gpu_1 = (abs(o_cpu.cuda() - o_gpu_1)).view(-1).max().item()
    err_o_gpu_2 = (abs(o_cpu.cuda() - o_gpu_2)).view(-1).max().item()
    err_o_gpu_3 = (abs(o_cpu.cuda() - o_cpu.cuda().to(eval_type))).view(-1).max().item()
    err_o_gpu_4 = (abs(o_cpu.cuda() - o_cpu.cuda().to(baseline_type))).view(-1).max().item()
    err_dq_gpu_0 = (abs(q_cpu.grad.cuda() - q_gpu_0.grad)).view(-1).max().item()
    err_dq_gpu_1 = (abs(q_cpu.grad.cuda() - q_gpu_1.grad)).view(-1).max().item()
    err_dq_gpu_2 = (abs(q_cpu.grad.cuda() - q_gpu_2.grad)).view(-1).max().item()
    err_dq_gpu_3 = (abs(q_cpu.grad.cuda() - q_gpu_3.grad)).view(-1).max().item()
    err_dq_gpu_4 = (abs(q_cpu.grad.cuda() - q_gpu_4.grad)).view(-1).max().item()
    err_dk_gpu_0 = (abs(k_cpu.grad.cuda() - k_gpu_0.grad)).view(-1).max().item()
    err_dk_gpu_1 = (abs(k_cpu.grad.cuda() - k_gpu_1.grad)).view(-1).max().item()
    err_dk_gpu_2 = (abs(k_cpu.grad.cuda() - k_gpu_2.grad)).view(-1).max().item()
    err_dk_gpu_3 = (abs(k_cpu.grad.cuda() - k_gpu_3.grad)).view(-1).max().item()
    err_dk_gpu_4 = (abs(k_cpu.grad.cuda() - k_gpu_4.grad)).view(-1).max().item()
    err_dv_gpu_0 = (abs(v_cpu.grad.cuda() - v_gpu_0.grad)).view(-1).max().item()
    err_dv_gpu_1 = (abs(v_cpu.grad.cuda() - v_gpu_1.grad)).view(-1).max().item()
    err_dv_gpu_2 = (abs(v_cpu.grad.cuda() - v_gpu_2.grad)).view(-1).max().item()
    err_dv_gpu_3 = (abs(v_cpu.grad.cuda() - v_gpu_3.grad)).view(-1).max().item()
    err_dv_gpu_4 = (abs(v_cpu.grad.cuda() - v_gpu_4.grad)).view(-1).max().item()

    print("O mean:", o_cpu.abs().mean().item())
    print("DQ mean:", q_cpu.grad.abs().mean().item())
    print("DK mean:", k_cpu.grad.abs().mean().item())
    print("DV mean:", v_cpu.grad.abs().mean().item())

    print("O error:", err_o_gpu_0, err_o_gpu_1, err_o_gpu_2, err_o_gpu_3, err_o_gpu_4)
    print("DQ error:", err_dq_gpu_0, err_dq_gpu_1, err_dq_gpu_2, err_dq_gpu_3, err_dq_gpu_4)
    print("DK error:", err_dk_gpu_0, err_dk_gpu_1, err_dk_gpu_2, err_dk_gpu_3, err_dk_gpu_4)
    print("DV error:", err_dv_gpu_0, err_dv_gpu_1, err_dv_gpu_2, err_dv_gpu_3, err_dv_gpu_4)

    print((DummyFAFixed.storage[0] - DummyFAFixed.storage[1]).abs().mean())
    print((DummyFAHP.storage[0] - DummyFAHP.storage[1]).abs().mean())

    # print("tmp0: ", torch.max((torch.abs(AttnFunc.storage[0][0] - DummyFA.storage[0][0].cpu())) / (1e-5 + torch.abs(AttnFunc.storage[0][0]))))
    # print("tmp1: ", torch.max((torch.abs(AttnFunc.storage[0][1] - DummyFA.storage[0][1].cpu())) / (1e-5 + torch.abs(AttnFunc.storage[0][1]))))
    # print("tmp2: ", torch.max((torch.abs(AttnFunc.storage[0][2] - DummyFA.storage[0][2].cpu())) / (1e-5 + torch.abs(AttnFunc.storage[0][2]))))    

