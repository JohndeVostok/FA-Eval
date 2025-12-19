import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np
import time

def torch_fa_fwd(
    q, k, v,
    mask,
    scale,
    LOW_TYPE = torch.bfloat16,
    HIGH_TYPE = torch.float32,
):
    s = torch.matmul(q, k.transpose(-1, -2)).to(HIGH_TYPE) * scale
    s.masked_fill_(mask == 0, float("-inf"))
    p = F.softmax(s - torch.max(s, dim=-1, keepdim=True).values , dim=-1)
    o = torch.matmul(p.to(LOW_TYPE), v)
    return o
    

def torch_fa_bwd(
    q, k, v, o, do,
    mask, scale,
    LOW_TYPE = torch.bfloat16,
    HIGH_TYPE = torch.float32,
):
    s = torch.matmul(q, k.transpose(-1, -2)).to(HIGH_TYPE) * scale
    s.masked_fill_(mask == 0, float("-inf"))
    p = F.softmax(s - torch.max(s, dim=-1, keepdim=True).values, dim=-1)
    dv = torch.matmul(p.transpose(-1, -2).to(LOW_TYPE), do)
    dp = torch.matmul(do, v.transpose(-1, -2))
    ds = p * (dp - torch.sum(do * o, dim=-1, keepdim=True))
    dq = (torch.matmul(ds.to(LOW_TYPE), k) * scale)
    dk = (torch.matmul(ds.to(LOW_TYPE).transpose(-1, -2), q) * scale)
    return dq, dk, dv


configs = [
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=4),
    triton.Config({}, num_warps=4, num_stages=5),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=4),
    triton.Config({}, num_warps=8, num_stages=5),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=4),
    triton.Config({}, num_warps=8, num_stages=5),
]

@triton.autotune(configs=configs, key=['S', 'N', 'H'])
@triton.jit
def kernel_fa_fwd(
    q, k, v, o,
    lse,
    mask,
    scale : tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    num_c = tl.cdiv(S, BLOCK_C)
    for task_id in range(pid, B * N * num_r, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id % (N * num_r) // num_r
        idx_r = task_id % num_r
        offset_h = tl.arange(0, H)

        offset_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * H + offset_h[None, :] + idx_b * N * S * H + idx_n * S * H
        # mask_q = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S)
        mask_q = None
        # block_q = tl.load(q + offset_q, mask=mask_q, other=0.0)
        block_q = tl.load(q + offset_q)
        block_o = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)
        block_l = tl.full([BLOCK_R], 0.0, dtype=HIGH_TYPE)
        block_m = tl.full([BLOCK_R], -float("inf"), dtype=HIGH_TYPE)
        for idx_c in range(0, num_c):
            offset_kv = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * H + offset_h[None, :] + idx_b * N * S * H + idx_n * S * H
            # mask_kv = ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S)
            mask_kv = None
            offset_mask = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
            # mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S)
            mask_mask = None

            # block_mask = tl.load(mask + offset_mask, mask=mask_mask, other=False)
            # block_k = tl.load(k + offset_kv, mask=mask_kv, other=0.0)
            # block_v = tl.load(v + offset_kv, mask=mask_kv, other=0.0)
            block_mask = tl.load(mask + offset_mask)
            block_k = tl.load(k + offset_kv)
            block_v = tl.load(v + offset_kv)

            block_s = tl.dot(block_q, block_k.T).to(HIGH_TYPE) * scale
            block_s += tl.where(block_mask, 0, -float("inf"))
            block_m_1 = tl.maximum(block_m, tl.max(block_s, axis=1))
            block_s = tl.exp(block_s - block_m_1[:, None])
            block_l_1 = tl.exp(block_m - block_m_1) * block_l + tl.sum(block_s, axis=1)
            block_o = tl.exp(block_m - block_m_1)[:, None] * block_o + tl.dot(block_s.to(LOW_TYPE), block_v).to(HIGH_TYPE)
            block_m = block_m_1
            block_l = block_l_1

        block_o = block_o / block_l[:, None]
        block_lse = tl.log(block_l) + block_m
        # tl.store(o + offset_q, block_o.to(LOW_TYPE), mask=mask_q)
        tl.store(o + offset_q, block_o.to(LOW_TYPE))
        offset_lse = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] + idx_b * N * S + idx_n * S
        # mask_lse = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S)
        # tl.store(lse + offset_lse, block_lse, mask=mask_lse)
        tl.store(lse + offset_lse, block_lse)

config_d = [
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
    triton.Config({}, num_warps=16),
    triton.Config({}, num_warps=32),
    triton.Config({}, num_warps=64),
    triton.Config({}, num_warps=128),
]

@triton.autotune(configs=configs, key=['S', 'N', 'H'])
@triton.jit
def kernel_fa_bwd_d(
    o, do, d,
    B: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_block_s = (S - 1) // BLOCK_R + 1
    for task_id in range(pid, num_block_s * B * N, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_block_s) 
        idx_n = task_id // num_block_s % N
        idx_r = task_id % num_block_s * BLOCK_R
        offset_h = tl.arange(0, H)
        offset_o = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * H + offset_h[None, :] + idx_b * N * S * H + idx_n * S * H
        mask_o = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S)
        offset_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] + idx_b * N * S + idx_n * S
        mask_d = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S)

        # block_o = tl.load(o + offset_o, mask=mask_o, other=0.0)
        # block_do = tl.load(do + offset_o, mask=mask_o, other=0.0)
        block_o = tl.load(o + offset_o)
        block_do = tl.load(do + offset_o)
        block_d = tl.sum(block_do.to(HIGH_TYPE) * block_o.to(HIGH_TYPE), axis=1)
        # tl.store(d + offset_d, block_d, mask=mask_d)
        tl.store(d + offset_d, block_d)


@triton.autotune(configs=configs, key=['S', 'N', 'H'])
@triton.jit
def kernel_fa_bwd_kv(
    q, k, v, do, d, dk, dv,
    lse,
    mask,
    scale : tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    num_c = tl.cdiv(S, BLOCK_C)
    for task_id in range(pid, B * N * num_c, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_c)
        idx_n = task_id % (N * num_c) // num_c
        idx_c = task_id % num_c
        offset_h = tl.arange(0, H)
        offset_kv = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * H + offset_h[None, :] + idx_b * N * S * H + idx_n * S * H
        # mask_kv = ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S)
        mask_kv = None
        # block_k = tl.load(k + offset_kv, mask=mask_kv, other=0.0)
        # block_v = tl.load(v + offset_kv, mask=mask_kv, other=0.0)
        block_k = tl.load(k + offset_kv)
        block_v = tl.load(v + offset_kv)
        block_dk = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)
        block_dv = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)

        for idx_r in range(0, num_r):
            offset_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * H + offset_h[None, :] + idx_b * N * S * H + idx_n * S * H
            # mask_q = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S)
            mask_q = None
            offset_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] + idx_b * N * S + idx_n * S
            mask_d = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S)
            offset_mask = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
            # mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S)
            mask_mask = None

            # block_q = tl.load(q + offset_q, mask=mask_q, other=0.0)
            # block_do = tl.load(do + offset_q, mask=mask_q, other=0.0)
            # block_lse = tl.load(lse + offset_d, mask=mask_d, other=0.0)
            # block_d = tl.load(d + offset_d, mask=mask_d, other=0.0)
            # block_mask = tl.load(mask + offset_mask, mask=mask_mask, other=False)
            block_q = tl.load(q + offset_q)
            block_do = tl.load(do + offset_q)
            block_lse = tl.load(lse + offset_d)
            block_d = tl.load(d + offset_d)
            block_mask = tl.load(mask + offset_mask)

            block_s = tl.dot(block_q, block_k.T).to(HIGH_TYPE) * scale
            block_s += tl.where(block_mask, 0, -float("inf"))
            block_p = tl.exp(block_s - block_lse[:, None])
            block_dv += tl.dot(block_p.to(LOW_TYPE).T, block_do).to(HIGH_TYPE)
            block_dp = tl.dot(block_do, block_v.T).to(HIGH_TYPE)
            block_ds = block_p * (block_dp - block_d[:, None])
            block_dk += tl.dot(block_ds.to(LOW_TYPE).T, block_q).to(HIGH_TYPE) * scale

        # tl.store(dk + offset_kv, block_dk.to(LOW_TYPE), mask=mask_kv)
        # tl.store(dv + offset_kv, block_dv.to(LOW_TYPE), mask=mask_kv)
        tl.store(dk + offset_kv, block_dk.to(LOW_TYPE))
        tl.store(dv + offset_kv, block_dv.to(LOW_TYPE))


@triton.autotune(configs=configs, key=['S', 'N', 'H'])
@triton.jit
def kernel_fa_bwd_q(
    q, k, v, do, d, dq,
    lse,
    mask,
    scale : tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    num_c = tl.cdiv(S, BLOCK_C)
    for task_id in range(pid, B * N * num_r, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id % (N * num_r) // num_r
        idx_r = task_id % num_r
        offset_h = tl.arange(0, H)
        offset_q = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * H + offset_h[None, :] + idx_b * N * S * H + idx_n * S * H
        # mask_q = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S)
        mask_q = None
        offset_d = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] + idx_b * N * S + idx_n * S
        mask_d = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S)

        # block_q = tl.load(q + offset_q, mask=mask_q, other=0.0)
        # block_do = tl.load(do + offset_q, mask=mask_q, other=0.0)
        # block_lse = tl.load(lse + offset_d, mask=mask_d, other=0.0)
        # block_d = tl.load(d + offset_d, mask=mask_d, other=0.0)
        block_q = tl.load(q + offset_q)
        block_do = tl.load(do + offset_q)
        block_lse = tl.load(lse + offset_d)
        block_d = tl.load(d + offset_d)
        block_dq = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)

        for idx_c in range(0, num_c):
            offset_kv = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * H + offset_h[None, :] + idx_b * N * S * H + idx_n * S * H
            # mask_kv = ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S)
            mask_kv = None
            offset_mask = (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
            # mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S)
            mask_mask = None
            # block_mask = tl.load(mask + offset_mask, mask=mask_mask, other=False)
            # block_k = tl.load(k + offset_kv, mask=mask_kv, other=0.0)
            # block_v = tl.load(v + offset_kv, mask=mask_kv, other=0.0)
            block_mask = tl.load(mask + offset_mask)
            block_k = tl.load(k + offset_kv)
            block_v = tl.load(v + offset_kv)

            block_s = tl.dot(block_q, block_k.T).to(HIGH_TYPE) * scale
            block_s += tl.where(block_mask, 0, -float("inf"))
            block_p = tl.exp(block_s - block_lse[:, None])
            block_dp = tl.dot(block_do, block_v.T).to(HIGH_TYPE)
            block_ds = block_p * (block_dp - block_d[:, None])
            block_dq += tl.dot(block_ds.to(LOW_TYPE), block_k).to(HIGH_TYPE) * scale

        # tl.store(dq + offset_q, block_dq.to(LOW_TYPE), mask=mask_q)
        tl.store(dq + offset_q, block_dq.to(LOW_TYPE))


if __name__ == "__main__":
    B = 4
    S = 4096
    N = 32
    H = 128
    scale = 1.0 / math.sqrt(H)
    BLOCK_R = 128
    BLOCK_C = 64

    LOW_TYPE = torch.bfloat16
    TRITON_LOW_TYPE = tl.bfloat16
    HIGH_TYPE = torch.float32
    TRITON_HIGH_TYPE = tl.float32

    q = torch.randn(B, N, S, H, dtype=LOW_TYPE, device="cuda")
    k = torch.randn(B, N, S, H, dtype=LOW_TYPE, device="cuda")
    v = torch.randn(B, N, S, H, dtype=LOW_TYPE, device="cuda")
    mask = torch.ones((S, S), dtype=torch.bool, device="cuda")
    o = torch.zeros_like(q)
    o = torch_fa_fwd(q, k, v, mask, scale, LOW_TYPE, HIGH_TYPE)
    do = torch.randn(B, N, S, H, dtype=LOW_TYPE, device="cuda")
    dq, dk, dv = torch_fa_bwd(q, k, v, o, do, mask, scale, LOW_TYPE, HIGH_TYPE)
    d = torch.sum(do.to(HIGH_TYPE) * o.to(HIGH_TYPE), dim=-1)

    triton_lse = torch.zeros((B, N, S), dtype=HIGH_TYPE, device="cuda")
    triton_d = torch.empty((B, N, S), dtype=HIGH_TYPE, device="cuda")
    triton_o = torch.empty_like(o)
    triton_dq = torch.empty_like(dq)
    triton_dk = torch.empty_like(dk)
    triton_dv = torch.empty_like(dv)
    kernel_fa_fwd[[78]](q, k, v, triton_o, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    kernel_fa_bwd_d[[78]](o, do, triton_d, B, N, S, H, BLOCK_R, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    kernel_fa_bwd_kv[[78]](q, k, v, do, d, triton_dk, triton_dv, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    kernel_fa_bwd_q[[78]](q, k, v, do, d, triton_dq, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    print(torch.sum(abs(o - triton_o) / (abs(o) + 0.01)) / o.numel())
    print(torch.sum(abs(dq - triton_dq) / (abs(dq) + 0.01)) / dq.numel())
    print(torch.sum(abs(dk - triton_dk) / (abs(dk) + 0.01)) / dk.numel())
    print(torch.sum(abs(dv - triton_dv) / (abs(dv) + 0.01)) / dv.numel())

    num_eval = 32
    torch.cuda.synchronize()
    for _ in range(num_eval):
        kernel_fa_fwd[[78]](q, k, v, triton_o, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    torch.cuda.synchronize()
    st = time.time()
    for _ in range(num_eval):
        kernel_fa_fwd[[78]](q, k, v, triton_o, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    torch.cuda.synchronize()
    print(f"fa_fwd time: {(time.time() - st) / 1024 * 1000} ms, FLOPS: {2 * B * N * S * H * (S + H) / (time.time() - st) * num_eval / 1e12} TFLOPS")

    torch.cuda.synchronize()
    for _ in range(num_eval):
        kernel_fa_bwd_d[[78]](o, do, triton_d, B, N, S, H, BLOCK_R, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    torch.cuda.synchronize()
    st = time.time()
    for _ in range(num_eval):
        kernel_fa_bwd_d[[78]](o, do, triton_d, B, N, S, H, BLOCK_R, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    torch.cuda.synchronize()
    print(f"fa_bwd_d time: {(time.time() - st) / 1024 * 1000} ms, bandwidth: {2 * B * N * S * H * 2 / (time.time() - st) * num_eval / 1e9} GB/s")

    torch.cuda.synchronize()
    for _ in range(num_eval):
        kernel_fa_bwd_kv[[78]](q, k, v, do, d, triton_dk, triton_dv, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    torch.cuda.synchronize()
    st = time.time()
    for _ in range(num_eval):
        kernel_fa_bwd_kv[[78]](q, k, v, do, d, triton_dk, triton_dv, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    torch.cuda.synchronize()
    print(f"fa_bwd_kv time: {(time.time() - st) / 1024 * 1000} ms, FLOPS: {2 * B * N * S * H * (S + H) * 2 / (time.time() - st) * num_eval / 1e12} TFLOPS")

    torch.cuda.synchronize()
    for _ in range(num_eval):
        kernel_fa_bwd_q[[78]](q, k, v, do, d, triton_dq, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    torch.cuda.synchronize()
    st = time.time()
    for _ in range(num_eval):
        kernel_fa_bwd_q[[78]](q, k, v, do, d, triton_dq, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    torch.cuda.synchronize()
    print(f"fa_bwd_q time: {(time.time() - st) / 1024 * 1000} ms, FLOPS: {2 * B * N * S * H * (S * 2 + H) / (time.time() - st) * num_eval / 1e12} TFLOPS")




    