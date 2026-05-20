import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import pytest
from core_modules.normalization import SyncBatchNorm

_SINGLE_PORT = "29500"
_MULTI_PORT = "29502"


def _init_single():
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = _SINGLE_PORT
        dist.init_process_group("gloo", rank=0, world_size=1)


# ── single-process tests ──────────────────────────────────────────────────────

def test_syncbn_2d():
    _init_single()
    torch.manual_seed(0)
    x = torch.randn(8, 4, requires_grad=True)
    # SyncBN requires dim >= 3; 2D input should raise
    with pytest.raises(ValueError):
        SyncBatchNorm(4, affine=False)(x)


def test_syncbn_4d():
    _init_single()
    torch.manual_seed(0)
    x = torch.randn(4, 8, 6, 6, requires_grad=True)
    out = SyncBatchNorm(8, affine=False)(x)
    ref = F.batch_norm(x.detach(), None, None, training=True, eps=1e-5)
    assert torch.allclose(out.detach(), ref, atol=1e-5)


def test_syncbn_affine():
    _init_single()
    torch.manual_seed(0)
    x = torch.randn(4, 8, 6, 6, requires_grad=True)
    sbn = SyncBatchNorm(8, affine=True, bias=True)
    w, b = sbn.weight.detach(), sbn.bias.detach()
    out = sbn(x)
    ref = F.batch_norm(x.detach(), None, None, weight=w, bias=b, training=True, eps=1e-5)
    assert torch.allclose(out.detach(), ref, atol=1e-5)


def test_syncbn_eval_uses_running_stats():
    _init_single()
    torch.manual_seed(0)
    sbn = SyncBatchNorm(8)
    sbn.train()
    sbn(torch.randn(4, 8, 6, 6))  # populate running stats
    sbn.eval()
    x = torch.randn(4, 8, 6, 6)
    ref = F.batch_norm(
        x, sbn.running_mean, sbn.running_var,
        sbn.weight, sbn.bias, training=False, eps=1e-5,
    )
    assert torch.allclose(sbn(x), ref, atol=1e-5)


# ── multi-process test (true sync) ───────────────────────────────────────────

def _run_multiprocess(rank, world_size, result_queue):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = _MULTI_PORT
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(rank)
    x_local = torch.randn(4, 8, 6, 6)

    all_x = [torch.zeros_like(x_local) for _ in range(world_size)]
    dist.all_gather(all_x, x_local)
    global_x = torch.cat(all_x, dim=0)

    sbn = SyncBatchNorm(8, affine=False)
    out = sbn(x_local)

    ref_global = F.batch_norm(global_x, None, None, training=True, eps=1e-5)
    ref_local = ref_global[rank * 4: (rank + 1) * 4]

    passed = torch.allclose(out.detach(), ref_local, atol=1e-5)
    result_queue.put((rank, passed, (out.detach() - ref_local).abs().max().item()))
    dist.destroy_process_group()


def test_syncbn_multiprocess():
    world_size = 2
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(target=_run_multiprocess, args=(r, world_size, result_queue))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"rank process exited with code {p.exitcode}"

    results = [result_queue.get_nowait() for _ in range(world_size)]
    for rank, passed, max_diff in results:
        assert passed, f"rank {rank} FAIL max_diff={max_diff:.6f}"
