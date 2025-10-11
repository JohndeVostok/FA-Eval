import torch
from attention import DummyFA, DummyFAFixed, NaiveAttn

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
    attn_cpu.train()
    attn_gpu_0.train()
    attn_gpu_1.train()

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
    

    q_cpu, k_cpu, v_cpu = q.to(device='cpu', dtype=baseline_type).clone().detach().requires_grad_(True), k.to(device='cpu', dtype=baseline_type).clone().detach().requires_grad_(True), v.to(device='cpu', dtype=baseline_type).clone().detach().requires_grad_(True)
    q_gpu_0, k_gpu_0, v_gpu_0 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
    q_gpu_1, k_gpu_1, v_gpu_1 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)

    mask_cpu = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len).to('cpu')
    mask_gpu = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len).to('cuda')
    o_cpu = attn_cpu(q_cpu, k_cpu, v_cpu, mask=mask_cpu)
    o_gpu_0 = attn_gpu_0(q_gpu_0, k_gpu_0, v_gpu_0, mask=mask_gpu)
    o_gpu_1 = attn_gpu_1(q_gpu_1, k_gpu_1, v_gpu_1, mask=mask_gpu)

    do = torch.rand_like(o_cpu, dtype=eval_type)
    do_cpu = do.to(device='cpu', dtype=baseline_type)
    do_gpu_0 = do.to(device='cuda', dtype=eval_type)
    do_gpu_1 = do.to(device='cuda', dtype=eval_type)
    o_cpu.backward(gradient = do_cpu)
    o_gpu_0.backward(gradient = do_gpu_0)
    print(do_gpu_1)
    o_gpu_1.backward(gradient = do_gpu_1)

    print(f"Output shape: {o_cpu.shape}")
    err_o_gpu_0 = (abs(o_cpu.cuda() - o_gpu_0)).view(-1).max().item()
    err_o_gpu_1 = (abs(o_cpu.cuda() - o_gpu_1)).view(-1).max().item()
    err_dq_gpu_0 = (abs(q_cpu.grad.cuda() - q_gpu_0.grad)).view(-1).max().item()
    err_dq_gpu_1 = (abs(q_cpu.grad.cuda() - q_gpu_1.grad)).view(-1).max().item()
    err_dk_gpu_0 = (abs(k_cpu.grad.cuda() - k_gpu_0.grad)).view(-1).max().item()
    err_dk_gpu_1 = (abs(k_cpu.grad.cuda() - k_gpu_1.grad)).view(-1).max().item()
    err_dv_gpu_0 = (abs(v_cpu.grad.cuda() - v_gpu_0.grad)).view(-1).max().item()
    err_dv_gpu_1 = (abs(v_cpu.grad.cuda() - v_gpu_1.grad)).view(-1).max().item()
    print("O error:", err_o_gpu_0, err_o_gpu_1)
    print("DQ error:", err_dq_gpu_0, err_dq_gpu_1)
    print("DK error:", err_dk_gpu_0, err_dk_gpu_1)
    print("DV error:", err_dv_gpu_0, err_dv_gpu_1)

    # print("tmp0: ", torch.max((torch.abs(AttnFunc.storage[0][0] - DummyFA.storage[0][0].cpu())) / (1e-5 + torch.abs(AttnFunc.storage[0][0]))))
    # print("tmp1: ", torch.max((torch.abs(AttnFunc.storage[0][1] - DummyFA.storage[0][1].cpu())) / (1e-5 + torch.abs(AttnFunc.storage[0][1]))))
    # print("tmp2: ", torch.max((torch.abs(AttnFunc.storage[0][2] - DummyFA.storage[0][2].cpu())) / (1e-5 + torch.abs(AttnFunc.storage[0][2]))))    
