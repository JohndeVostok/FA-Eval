import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np
import time

def get_device_properties():
    return 78, 0


@triton.autotune(configs=[triton.Config({'BLOCK_R': 64, 'BLOCK_C': 64}, num_warps=4, num_stages=2),], key=['N', 'S', 'H'])
@triton.jit
def kernel_sdpa_fwd(
    q, k, v, o,
    lse,
    mask,
    scale : tl.constexpr,
    num_group : tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    num_c = tl.cdiv(S, BLOCK_C)
    group_size = N // num_group
    for task_id in range(pid, B * N * num_r, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)

        ptr_q = (
            q + 
            idx_b * STRIDE_Q_B + 
            idx_n * STRIDE_Q_N + 
            (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + 
            idx_h[None, :] * STRIDE_Q_H
        )
        ptr_o = (
            o + 
            idx_b * STRIDE_Q_B + 
            idx_n * STRIDE_Q_N + 
            (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + 
            idx_h[None, :] * STRIDE_Q_H
        )
        ptr_lse = (
            lse + 
            idx_b * STRIDE_D_B + 
            idx_n * STRIDE_D_N + 
            (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        )

        mask_q = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S)
        mask_lse = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S)

        block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
        block_o = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)
        block_l = tl.full([BLOCK_R], 0.0, dtype=HIGH_TYPE)
        block_m = tl.full([BLOCK_R], -1e6, dtype=HIGH_TYPE)
        for idx_c in range(0, num_c):
            ptr_k = (
                k + 
                idx_b * STRIDE_K_B + 
                (idx_n // group_size) * STRIDE_K_N + 
                (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S + 
                idx_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v + 
                idx_b * STRIDE_V_B + 
                (idx_n // group_size) * STRIDE_V_N + 
                (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S + 
                idx_h[None, :] * STRIDE_V_H
            )
            ptr_mask = (
                mask + 
                (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S + 
                (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
            )

            mask_kv = ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S)
            mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S)

            block_mask = tl.load(ptr_mask, mask=mask_mask, other=False)
            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)

            block_s = tl.dot(block_q, block_k.T).to(HIGH_TYPE) * scale
            block_s -= (1.0 - block_mask.to(HIGH_TYPE)) * 1e6
            block_m_1 = tl.maximum(block_m, tl.max(block_s, axis=1))
            block_s = tl.exp(block_s - block_m_1[:, None])
            block_l_1 = tl.exp(block_m - block_m_1) * block_l + tl.sum(block_s, axis=1)
            block_o = tl.exp(block_m - block_m_1)[:, None] * block_o + tl.dot(block_s.to(LOW_TYPE), block_v).to(HIGH_TYPE)
            block_m = block_m_1
            block_l = block_l_1

        block_o = block_o / block_l[:, None]
        block_lse = tl.log(block_l) + block_m
        tl.store(ptr_o, block_o.to(LOW_TYPE), mask=mask_q)
        tl.store(ptr_lse, block_lse, mask=mask_lse)


@triton.autotune(configs=[triton.Config({'BLOCK_R': 64}, num_warps=4, num_stages=2),], key=['N', 'S', 'H'])
@triton.jit
def kernel_sdpa_bwd_d(
    o, do, d,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_O_B: tl.constexpr,
    STRIDE_O_N: tl.constexpr,
    STRIDE_O_S: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    for task_id in range(pid, num_r * B * N, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)
        ptr_o = (
            o +
            idx_b * STRIDE_O_B + 
            idx_n * STRIDE_O_N + 
            (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S + 
            idx_h[None, :] * STRIDE_O_H
        )
        ptr_do = (
            do +
            idx_b * STRIDE_O_B + 
            idx_n * STRIDE_O_N + 
            (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_O_S + 
            idx_h[None, :] * STRIDE_O_H
        )
        ptr_d = (
            d +
            idx_b * STRIDE_D_B + 
            idx_n * STRIDE_D_N + 
            (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        )
        mask_o = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S)
        mask_d = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S)

        block_o = tl.load(ptr_o, mask=mask_o, other=0.0)
        block_do = tl.load(ptr_do, mask=mask_o, other=0.0)
        block_d = tl.sum(block_do.to(HIGH_TYPE) * block_o.to(HIGH_TYPE), axis=1)
        tl.store(ptr_d, block_d, mask=mask_d)


@triton.autotune(configs=[triton.Config({'BLOCK_R': 64, 'BLOCK_C': 64}, num_warps=4, num_stages=2),], key=['N', 'S', 'H'])
@triton.jit
def kernel_sdpa_bwd_q(
    q, k, v, do, d, dq,
    lse,
    mask,
    scale : tl.constexpr,
    num_group : tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    num_c = tl.cdiv(S, BLOCK_C)
    group_size = N // num_group
    for task_id in range(pid, B * N * num_r, tl.num_programs(axis=0)):
        idx_b = task_id // (N * num_r)
        idx_n = task_id // num_r % N
        idx_r = task_id % num_r
        idx_h = tl.arange(0, H)

        ptr_q = (
            q +
            idx_b * STRIDE_Q_B + 
            idx_n * STRIDE_Q_N + 
            (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + 
            idx_h[None, :] * STRIDE_Q_H
        )
        ptr_do = (
            do +
            idx_b * STRIDE_Q_B + 
            idx_n * STRIDE_Q_N + 
            (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + 
            idx_h[None, :] * STRIDE_Q_H
        )
        ptr_dq = (
            dq +
            idx_b * STRIDE_Q_B + 
            idx_n * STRIDE_Q_N + 
            (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + 
            idx_h[None, :] * STRIDE_Q_H
        )
        ptr_d = (
            d +
            idx_b * STRIDE_D_B + 
            idx_n * STRIDE_D_N + 
            (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        )
        ptr_lse = (
            lse +
            idx_b * STRIDE_D_B + 
            idx_n * STRIDE_D_N + 
            (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
        )

        mask_q = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S)
        mask_d = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S)
        block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
        block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
        block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
        block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
        block_dq = tl.full([BLOCK_R, H], 0.0, dtype=HIGH_TYPE)

        for idx_c in range(0, num_c):
            ptr_k = (
                k +
                idx_b * STRIDE_K_B + 
                (idx_n // group_size) * STRIDE_K_N + 
                (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S + 
                idx_h[None, :] * STRIDE_K_H
            )
            ptr_v = (
                v +
                idx_b * STRIDE_V_B + 
                (idx_n // group_size) * STRIDE_V_N + 
                (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S + 
                idx_h[None, :] * STRIDE_V_H
            )
            ptr_mask = (
                mask + 
                (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S + 
                (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
            )

            mask_kv = ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S)
            mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S)

            block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
            block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
            block_mask = tl.load(ptr_mask, mask=mask_mask, other=False)

            block_s = tl.dot(block_q, block_k.T).to(HIGH_TYPE) * scale
            block_s -= (1.0 - block_mask.to(HIGH_TYPE)) * 1e6
            block_p = tl.exp(block_s - block_lse[:, None])
            block_dp = tl.dot(block_do, block_v.T).to(HIGH_TYPE)
            block_ds = block_p * (block_dp - block_d[:, None])
            block_dq += tl.dot(block_ds.to(LOW_TYPE), block_k).to(HIGH_TYPE) * scale

        tl.store(ptr_dq, block_dq.to(LOW_TYPE), mask=mask_q)

@triton.autotune(configs=[triton.Config({'BLOCK_R': 64, 'BLOCK_C': 64}, num_warps=4, num_stages=2),], key=['N', 'S', 'H'])
@triton.jit
def kernel_sdpa_bwd_kv(
    q, k, v, do, d, dk, dv,
    lse,
    mask,
    scale : tl.constexpr,
    num_group : tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    num_r = tl.cdiv(S, BLOCK_R)
    num_c = tl.cdiv(S, BLOCK_C)
    group_size = N // num_group
    for task_id in range(pid, B * num_group * num_c, tl.num_programs(axis=0)):
        idx_b = task_id // (num_group * num_c)
        idx_group = task_id // num_c % num_group
        idx_c = task_id % num_c
        idx_h = tl.arange(0, H)

        ptr_k = (
            k + 
            idx_b * STRIDE_K_B + 
            idx_group * STRIDE_K_N + 
            (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S + 
            idx_h[None, :] * STRIDE_K_H
        )
        ptr_v = (
            v + 
            idx_b * STRIDE_V_B + 
            idx_group * STRIDE_V_N + 
            (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S + 
            idx_h[None, :] * STRIDE_V_H
        )
        ptr_dk = (
            dk + 
            idx_b * STRIDE_K_B + 
            idx_group * STRIDE_K_N + 
            (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S + 
            idx_h[None, :] * STRIDE_K_H
        )
        ptr_dv = (
            dv + 
            idx_b * STRIDE_V_B + 
            idx_group * STRIDE_V_N + 
            (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S + 
            idx_h[None, :] * STRIDE_V_H
        )

        mask_kv = ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S)
        block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
        block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
        block_dk = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)
        block_dv = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)

        for idx_ingroup in range(group_size):
            idx_n = idx_group * group_size + idx_ingroup
            for idx_r in range(0, num_r):
                ptr_q = (
                    q + 
                    idx_b * STRIDE_Q_B + 
                    idx_n * STRIDE_Q_N + 
                    (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + 
                    idx_h[None, :] * STRIDE_Q_H
                )
                ptr_do = (
                    do + 
                    idx_b * STRIDE_Q_B + 
                    idx_n * STRIDE_Q_N + 
                    (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * STRIDE_Q_S + 
                    idx_h[None, :] * STRIDE_Q_H
                )
                ptr_d = (
                    d + 
                    idx_b * STRIDE_D_B + 
                    idx_n * STRIDE_D_N + 
                    (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
                )
                ptr_lse = (
                    lse + 
                    idx_b * STRIDE_D_B + 
                    idx_n * STRIDE_D_N + 
                    (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
                )
                ptr_mask = (
                    mask + 
                    (idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * S + 
                    (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :]
                )

                mask_q = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S)
                mask_d = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < S)
                mask_mask = ((idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] < S) & ((idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[None, :] < S)

                block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
                block_do = tl.load(ptr_do, mask=mask_q, other=0.0)
                block_lse = tl.load(ptr_lse, mask=mask_d, other=0.0)
                block_d = tl.load(ptr_d, mask=mask_d, other=0.0)
                block_mask = tl.load(ptr_mask, mask=mask_mask, other=False)

                block_s = tl.dot(block_q, block_k.T).to(HIGH_TYPE) * scale
                block_s -= (1.0 - block_mask.to(HIGH_TYPE)) * 1e6
                block_p = tl.exp(block_s - block_lse[:, None])
                block_dv += tl.dot(block_p.to(LOW_TYPE).T, block_do).to(HIGH_TYPE)
                block_dp = tl.dot(block_do, block_v.T).to(HIGH_TYPE)
                block_ds = block_p * (block_dp - block_d[:, None])
                block_dk += tl.dot(block_ds.to(LOW_TYPE).T, block_q).to(HIGH_TYPE) * scale

        tl.store(ptr_dk, block_dk.to(LOW_TYPE), mask=mask_kv)
        tl.store(ptr_dv, block_dv.to(LOW_TYPE), mask=mask_kv)


def sdpa_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None,
    scale: float = 1.0,
    gqa_enabled: bool = False,
):
    """
    Forward computation interface:
    Args:
        q: Query tensor (Q), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        k: Key tensor (K), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        v: Value tensor (V), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        mask: Attention mask, shape [SEQ, SEQ]
        scale: Scaling factor for QK product
    Returns:
        o: Attention output tensor, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        lse: LogSumExp tensor, shape [BSZ, Q_HEAD_NUM, SEQ]
    """
    # shape constraints
    assert len(q.shape) == 4 and len(k.shape) == 4 and len(v.shape) == 4
    assert len(mask.shape) == 2 and mask.dtype == torch.bool and mask.shape[0] == mask.shape[1]

    if gqa_enabled:
        assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    else:
        assert q.shape[1] == k.shape[1] and q.shape[1] == v.shape[1]
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] == mask.shape[0]
    assert q.shape[3] == k.shape[3] and k.shape[3] == v.shape[3] and q.shape[3] in {64, 128}

    o = torch.empty_like(q)
    lse = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    num_cores, _ = get_device_properties()

    kernel_sdpa_fwd[(num_cores,)](
        q, k, v, o,
        lse,
        mask,
        scale,
        k.shape[1],
        q.shape[0], q.shape[1], q.shape[2], q.shape[3],
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
    )

    return o, lse


def sdpa_bwd_impl(
    o: torch.Tensor,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    mask: torch.Tensor = None,
    scale: float = 1.0,
    gqa_enabled: bool = False,
):
    """
    Backward computation interface:
    Args:
        o: Attention output tensor, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        do: Gradient tensor for o, shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        q: Query tensor (Q), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        k: Key tensor (K), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        v: Value tensor (V), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        lse: Logsumexp tensor, shape [BSZ, Q_HEAD_NUM, SEQ]
        mask: Attention mask, shape [SEQ, SEQ]
        scale: Scaling factor for QK product
    Returns:
        dq: Gradient tensor for Query tensor (Q), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        dk: Gradient tensor for Key tensor (K), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
        dv: Gradient tensor for Value tensor (V), shape [BSZ, Q_HEAD_NUM, SEQ, HEAD_DIM]
    """
    # shape constraints
    assert len(q.shape) == 4 and len(k.shape) == 4 and len(v.shape) == 4 and len(lse.shape) == 3
    assert q.shape == o.shape and o.shape == do.shape
    assert len(mask.shape) == 2 and mask.dtype == torch.bool and mask.shape[0] == mask.shape[1]
    if gqa_enabled:
        assert k.shape[1] == v.shape[1] and q.shape[1] % k.shape[1] == 0
    else:
        assert q.shape[1] == k.shape[1] and q.shape[1] == v.shape[1]
    assert q.shape[2] == k.shape[2] and k.shape[2] == v.shape[2] and q.shape[2] == mask.shape[0]
    assert q.shape[3] == k.shape[3] and k.shape[3] == v.shape[3] and q.shape[3] in {64, 128}
    assert q.shape[0] == lse.shape[0] and q.shape[1] == lse.shape[1] and q.shape[2] == lse.shape[2]

    num_cores, _ = get_device_properties()
    d = torch.empty_like(lse)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    kernel_sdpa_bwd_d[(num_cores,)](
        o, do, d,
        o.shape[0], o.shape[1], o.shape[2], o.shape[3],
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        d.stride(0), d.stride(1), d.stride(2),
    )
    kernel_sdpa_bwd_q[(num_cores,)](
        q, k, v, do, d, dq,
        lse,
        mask,
        scale,
        k.shape[1],
        q.shape[0], q.shape[1], q.shape[2], q.shape[3],
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        d.stride(0), d.stride(1), d.stride(2),
    )
    kernel_sdpa_bwd_kv[(num_cores,)](
        q, k, v, do, d, dk, dv,
        lse,
        mask,
        scale,
        k.shape[1],
        q.shape[0], q.shape[1], q.shape[2], q.shape[3],
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        d.stride(0), d.stride(1), d.stride(2),
    )
    
    return dq, dk, dv


def torch_fa_fwd(
    q, k, v,
    mask,
    scale,
    LOW_TYPE = torch.bfloat16,
    HIGH_TYPE = torch.float32,
):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    s = torch.matmul(q, k.transpose(-1, -2)).to(HIGH_TYPE) * scale
    s.masked_fill_(mask == 0, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = F.softmax(s - torch.max(s, dim=-1, keepdim=True).values , dim=-1)
    o = torch.matmul(p.to(LOW_TYPE), v)
    return o, lse
    

def torch_fa_bwd(
    q, k, v, o, do,
    mask, scale,
    LOW_TYPE = torch.bfloat16,
    HIGH_TYPE = torch.float32,
):
    num_group = k.shape[1]
    group_size = q.shape[1] // num_group
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    s = torch.matmul(q, k.transpose(-1, -2)).to(HIGH_TYPE) * scale
    s.masked_fill_(mask == 0, float("-inf"))
    p = F.softmax(s - torch.max(s, dim=-1, keepdim=True).values, dim=-1)
    dv = torch.matmul(p.transpose(-1, -2).to(LOW_TYPE), do)
    dp = torch.matmul(do, v.transpose(-1, -2))
    ds = p * (dp - torch.sum(do * o, dim=-1, keepdim=True))
    dq = (torch.matmul(ds.to(LOW_TYPE), k) * scale)
    dk = (torch.matmul(ds.to(LOW_TYPE).transpose(-1, -2), q) * scale)
    if group_size > 1:
        dk = dk.reshape(q.shape[0], num_group, group_size, dk.shape[2], dk.shape[3]).sum(dim=2)
        dv = dv.reshape(q.shape[0], num_group, group_size, dv.shape[2], dv.shape[3]).sum(dim=2)
    return dq, dk, dv


if __name__ == "__main__":
    B = 4
    S = 4096
    N = 16
    H = 128
    scale = 1.0 / math.sqrt(H)
    num_group = 4
    group_size = N // num_group

    LOW_TYPE = torch.bfloat16
    TRITON_LOW_TYPE = tl.bfloat16
    HIGH_TYPE = torch.float32
    TRITON_HIGH_TYPE = tl.float32

    q = torch.randn(B, N, S, H, dtype=LOW_TYPE, device="cuda")
    k = torch.randn(B, num_group, S, H, dtype=LOW_TYPE, device="cuda")
    v = torch.randn(B, num_group, S, H, dtype=LOW_TYPE, device="cuda")
    do = torch.randn(B, N, S, H, dtype=LOW_TYPE, device="cuda")
    mask = torch.ones((S, S), dtype=torch.bool, device="cuda")

    torch_o, torch_lse = torch_fa_fwd(q, k, v, mask, scale)
    torch_dq, torch_dk, torch_dv = torch_fa_bwd(q, k, v, torch_o, do, mask, scale)

    triton_o, triton_lse = sdpa_fwd_impl(q, k, v, mask, scale, group_size > 1)
    triton_dq, triton_dk, triton_dv = sdpa_bwd_impl(torch_o, do, q, k, v, triton_lse, mask, scale, group_size > 1)

    print(torch.sum(abs(torch_o - triton_o) / (abs(torch_o) + 0.01)) / torch_o.numel())
    print(torch.sum(abs(torch_dq - triton_dq) / (abs(torch_dq) + 0.01)) / torch_dq.numel())
    print(torch.sum(abs(torch_dk - triton_dk) / (abs(torch_dk) + 0.01)) / torch_dk.numel())
    print(torch.sum(abs(torch_dv - triton_dv) / (abs(torch_dv) + 0.01)) / torch_dv.numel())

    # num_eval = 32
    # torch.cuda.synchronize()
    # for _ in range(num_eval):
    #     kernel_fa_fwd[[78]](q, k, v, triton_o, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    # torch.cuda.synchronize()
    # st = time.time()
    # for _ in range(num_eval):
    #     kernel_fa_fwd[[78]](q, k, v, triton_o, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    # torch.cuda.synchronize()
    # print(f"fa_fwd time: {(time.time() - st) / 1024 * 1000} ms, FLOPS: {2 * B * N * S * H * (S + H) / (time.time() - st) * num_eval / 1e12} TFLOPS")

    # torch.cuda.synchronize()
    # for _ in range(num_eval):
    #     kernel_fa_bwd_d[[78]](o, do, triton_d, B, N, S, H, BLOCK_R, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    # torch.cuda.synchronize()
    # st = time.time()
    # for _ in range(num_eval):
    #     kernel_fa_bwd_d[[78]](o, do, triton_d, B, N, S, H, BLOCK_R, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    # torch.cuda.synchronize()
    # print(f"fa_bwd_d time: {(time.time() - st) / 1024 * 1000} ms, bandwidth: {2 * B * N * S * H * 2 / (time.time() - st) * num_eval / 1e9} GB/s")

    # torch.cuda.synchronize()
    # for _ in range(num_eval):
    #     kernel_fa_bwd_kv[[78]](q, k, v, do, d, triton_dk, triton_dv, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    # torch.cuda.synchronize()
    # st = time.time()
    # for _ in range(num_eval):
    #     kernel_fa_bwd_kv[[78]](q, k, v, do, d, triton_dk, triton_dv, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    # torch.cuda.synchronize()
    # print(f"fa_bwd_kv time: {(time.time() - st) / 1024 * 1000} ms, FLOPS: {2 * B * N * S * H * (S + H) * 2 / (time.time() - st) * num_eval / 1e12} TFLOPS")

    # torch.cuda.synchronize()
    # for _ in range(num_eval):
    #     kernel_fa_bwd_q[[78]](q, k, v, do, d, triton_dq, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    # torch.cuda.synchronize()
    # st = time.time()
    # for _ in range(num_eval):
    #     kernel_fa_bwd_q[[78]](q, k, v, do, d, triton_dq, triton_lse, mask, scale, B, S, N, H, BLOCK_R, BLOCK_C, TRITON_LOW_TYPE, TRITON_HIGH_TYPE)
    # torch.cuda.synchronize()
    # print(f"fa_bwd_q time: {(time.time() - st) / 1024 * 1000} ms, FLOPS: {2 * B * N * S * H * (S * 2 + H) / (time.time() - st) * num_eval / 1e12} TFLOPS")




    