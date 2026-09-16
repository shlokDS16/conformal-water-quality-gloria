"""Mixture density network (MDN) reimplemented in PyTorch after MDN-STREAM defaults.

Reference behaviour (read, not vendored; GPL-3.0) from https://github.com/STREAM-RS/MDN-STREAM
commit 79d1b47a992d25b134bdfe4d10fe7f4e7e30e6d1:
  parameters.py: n_mix 5, n_layers 5, n_hidden 100, batch 128, lr 1e-3, l2 1e-3, epsilon 1e-3,
                 n_iter 10000, n_rounds 10, bagging on (75 % of rows per round).
  transformers/__init__.py: x -> RobustScaler on raw Rrs; y -> natural log then MinMaxScaler(-1, 1).
  model/MDN.py: Dense ReLU hidden layers, glorot_uniform kernels, zero biases, L2(l2) on kernels and
                biases of every Dense (incl. output); output prior softmax (+1e-9), mu, scale;
                sigma = scale^2 + epsilon (1 target); loss mean NLL + L2; Adam(lr, Keras eps 1e-7);
                epochs = max(1, int(n_iter / (N / batch))) with no early stopping.
  product_estimation.py: estimate = mean of the max-prior component ("top"), median over rounds.

Deviations (documented in research/EXPERIMENT_BUILD_LOG.md):
  * mini-batches are drawn i.i.d. uniformly per step from a counter-based hash of
    (member seed, step, slot) instead of Keras per-epoch shuffling; this makes each member's
    batch stream independent of how members are grouped on the GPU.
  * total number of optimiser steps = epochs * ceil(N_bag / batch), as in Keras.
  * the target passed in is log10(y); MDN-STREAM's natural log is a linear rescaling of it and
    MinMax(-1, 1) removes the difference, so the fitted model is equivalent.

Vectorised ensemble: all members (possibly from several splits) share one training loop via
torch.func.functional_call + vmap over stacked parameters (stack_module_state layout).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap
from scipy.special import ndtr

MDN_DEFAULTS = dict(n_mix=5, n_layers=5, n_hidden=100, batch=128, lr=1e-3, l2=1e-3,
                    epsilon=1e-3, n_iter=10_000, n_rounds=10, bag_frac=0.75)
RECAL_SEED_OFFSET = 500_000  # member seeds of the 80 %-train models used by recal_train20 (= config.MDN_RECAL_OFFSET)


# ---------------------------------------------------------------- scalers (numpy, fit on member bag)
def robust_fit(X: np.ndarray):
    med = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0  # sklearn RobustScaler behaviour for constant columns
    return med, iqr


def minmax_fit(y: np.ndarray):
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo if hi > lo else 1.0
    return lo, span  # y_t = 2 (y - lo) / span - 1


class _Net(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_layers: int, n_mix: int):
        super().__init__()
        sizes = [n_in] + [n_hidden] * n_layers
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.out = nn.Linear(n_hidden, 3 * n_mix)
        self.n_mix = n_mix

    def forward(self, x):
        for lin in self.hidden:
            x = torch.relu(lin(x))
        o = self.out(x)
        k = self.n_mix
        return o[..., :k], o[..., k:2 * k], o[..., 2 * k:]


def _glorot_init(net: _Net, rng: np.random.Generator):
    with torch.no_grad():
        for lin in list(net.hidden) + [net.out]:
            fan_out, fan_in = lin.weight.shape
            lim = math.sqrt(6.0 / (fan_in + fan_out))
            lin.weight.copy_(torch.from_numpy(rng.uniform(-lim, lim, size=(fan_out, fan_in)).astype(np.float32)))
            lin.bias.zero_()


def _hash32(x: torch.Tensor) -> torch.Tensor:
    """32-bit integer hash (no int64 overflow: products stay below 2^59)."""
    m = 0xFFFFFFFF
    x = x & m
    x = ((x >> 16) ^ x) * 0x45D9F3B & m
    x = ((x >> 16) ^ x) * 0x45D9F3B & m
    x = (x >> 16) ^ x
    return x


@dataclass
class MemberTask:
    X: np.ndarray            # raw features, training rows available to the member
    y_log10: np.ndarray      # log10 target
    seed: int                # member seed
    bag: bool = True         # draw 75 % of rows (MDN-STREAM bagging)
    meta: dict = field(default_factory=dict)


@dataclass
class FittedMember:
    state: dict              # parameter tensors on CPU
    x_med: np.ndarray
    x_iqr: np.ndarray
    y_lo: float
    y_span: float
    n_bag: int


def _prepare(task: MemberTask, cfg: dict):
    rng = np.random.default_rng(task.seed)
    n = len(task.X)
    if task.bag:
        idx = rng.permutation(n)[: int(cfg["bag_frac"] * n)]
    else:
        idx = np.arange(n)
    Xb, yb = task.X[idx], task.y_log10[idx]
    med, iqr = robust_fit(Xb)
    lo, span = minmax_fit(yb)
    Xt = ((Xb - med) / iqr).astype(np.float32)
    yt = (2.0 * (yb - lo) / span - 1.0).astype(np.float32)
    net = _Net(Xb.shape[1], cfg["n_hidden"], cfg["n_layers"], cfg["n_mix"])
    _glorot_init(net, rng)
    return net, Xt, yt, (med, iqr, lo, span)


def fit_members(tasks: list[MemberTask], device: str = "cuda", cfg: dict | None = None,
                chunk: int = 400, verbose: bool = False) -> list[FittedMember]:
    """Train many independent MDN members in vectorised chunks. Order of outputs = order of tasks."""
    cfg = {**MDN_DEFAULTS, **(cfg or {})}
    out: list[FittedMember] = []
    for c0 in range(0, len(tasks), chunk):
        out.extend(_fit_chunk(tasks[c0:c0 + chunk], device, cfg, verbose))
    return out


def _fit_chunk(tasks, device, cfg, verbose):
    torch.use_deterministic_algorithms(True, warn_only=False)
    prepared = [_prepare(t, cfg) for t in tasks]
    M = len(prepared)
    d = prepared[0][1].shape[1]
    nmax = max(p[1].shape[0] for p in prepared)
    Xall = np.zeros((M, nmax, d), np.float32)
    yall = np.zeros((M, nmax), np.float32)
    nrows = np.zeros(M, np.int64)
    steps_per_model = np.zeros(M, np.int64)
    B = cfg["batch"]
    for i, (_, Xt, yt, _) in enumerate(prepared):
        n = len(Xt)
        Xall[i, :n], yall[i, :n], nrows[i] = Xt, yt, n
        epochs = max(1, int(cfg["n_iter"] / max(1.0, n / B)))
        steps_per_model[i] = epochs * math.ceil(n / B)
    n_steps = int(steps_per_model.max())

    nets = [p[0] for p in prepared]
    params, buffers = stack_module_state(nets)
    params = {k: v.detach().to(device).requires_grad_(True) for k, v in params.items()}
    buffers = {k: v.to(device) for k, v in buffers.items()}
    template = _Net(d, cfg["n_hidden"], cfg["n_layers"], cfg["n_mix"]).to("meta")

    Xg = torch.from_numpy(Xall).to(device)
    yg = torch.from_numpy(yall).to(device)
    ng = torch.from_numpy(nrows).to(device)
    seeds = torch.tensor([int(t.seed) & 0xFFFFFFFF for t in tasks], dtype=torch.int64, device=device)
    active_until = torch.from_numpy(steps_per_model).to(device)
    slots = torch.arange(B, dtype=torch.int64, device=device)
    ar = torch.arange(M, device=device)[:, None]
    eps = cfg["epsilon"]
    l2 = cfg["l2"]
    log2pi = math.log(2 * math.pi)

    def nll(p, b, xb, yb):
        logits, mu, s = functional_call(template, (p, b), (xb,))
        prior = torch.softmax(logits, -1) + 1e-9
        var = s * s + eps
        comp = -0.5 * (log2pi + torch.log(var) + (yb[:, None] - mu) ** 2 / var)
        ll = torch.logsumexp(torch.log(prior) + comp, -1)
        reg = sum((w * w).sum() for w in p.values())
        return -ll.mean() + l2 * reg

    batched = vmap(nll, in_dims=(0, 0, 0, 0))
    opt = torch.optim.Adam(list(params.values()), lr=cfg["lr"], eps=1e-7)
    finish = {}
    for i, k in enumerate(steps_per_model.tolist()):
        finish.setdefault(int(k), []).append(i)
    snap: dict[int, dict] = {}
    for step in range(n_steps):
        h = _hash32(_hash32(seeds * 0x9E3779B1 ^ step)[:, None] ^ (slots[None, :] * 0x85EBCA6B))
        idx = h % ng[:, None]
        xb = Xg[ar, idx]
        yb = yg[ar, idx]
        losses = batched(params, buffers, xb, yb)
        opt.zero_grad(set_to_none=True)
        losses.sum().backward()
        opt.step()
        if step + 1 in finish:  # members whose Keras-equivalent step budget ends here: exact snapshot
            for i in finish[step + 1]:
                snap[i] = {k: v[i].detach().cpu().clone() for k, v in params.items()}
        if verbose and step % 2000 == 0:
            print(f"  step {step}/{n_steps} mean loss {float(losses.mean()):.4f}", flush=True)

    fitted = []
    for i, (_, Xt, _, (med, iqr, lo, span)) in enumerate(prepared):
        fitted.append(FittedMember(snap[i], med, iqr, lo, span, len(Xt)))
    return fitted


# ---------------------------------------------------------------- prediction
@torch.no_grad()
def member_coefs(member: FittedMember, X: np.ndarray, cfg: dict | None = None):
    """Mixture coefficients in log10 target space: prior (n, K), mu (n, K), sd (n, K)."""
    cfg = {**MDN_DEFAULTS, **(cfg or {})}
    net = _Net(X.shape[1], cfg["n_hidden"], cfg["n_layers"], cfg["n_mix"])
    net.load_state_dict(member.state)
    xt = torch.from_numpy(((X - member.x_med) / member.x_iqr).astype(np.float32))
    logits, mu, s = net(xt)
    prior = (torch.softmax(logits, -1) + 1e-9).double().numpy()
    prior /= prior.sum(1, keepdims=True)
    var_t = (s * s + cfg["epsilon"]).double().numpy()
    a = member.y_span / 2.0  # y_log10 = (y_t + 1) * span / 2 + lo
    mu10 = (mu.double().numpy() + 1.0) * a + member.y_lo
    sd10 = np.sqrt(var_t) * a
    return prior, mu10, sd10


def ensemble_summary(members: list[FittedMember], X: np.ndarray, alphas=(0.05, 0.10, 0.20), cfg=None,
                     quantiles: bool = True):
    """Point (median over members of top-component mean), mixture mean, total SD, native quantiles.

    Native interval = central quantiles of the equal-weight mixture of all members' mixtures.
    """
    P, MU, SD, TOP = [], [], [], []
    for m in members:
        p, mu, sd = member_coefs(m, X, cfg)
        P.append(p); MU.append(mu); SD.append(sd)
        TOP.append(mu[np.arange(len(X)), p.argmax(1)])
    P = np.concatenate(P, 1) / len(members)
    MU = np.concatenate(MU, 1)
    SD = np.concatenate(SD, 1)
    point = np.median(np.stack(TOP, 0), 0)
    mean = (P * MU).sum(1)
    var = (P * (SD ** 2 + MU ** 2)).sum(1) - mean ** 2
    out = {"point": point, "mix_mean": mean, "mix_sd": np.sqrt(np.maximum(var, 1e-12))}
    if not quantiles or not alphas:
        return out
    levels = sorted({a / 2 for a in alphas} | {1 - a / 2 for a in alphas})
    qs = mixture_quantiles(P, MU, SD, levels)
    for lv, q in zip(levels, qs):
        out[f"q{lv:.3f}"] = q
    return out


def mixture_quantiles(P, MU, SD, levels, iters: int = 60):
    """Quantiles of per-row Gaussian mixtures by vectorised bisection."""
    lo = (MU - 12 * SD).min(1)
    hi = (MU + 12 * SD).max(1)
    res = []
    for lv in levels:
        a, b = lo.copy(), hi.copy()
        for _ in range(iters):
            mid = 0.5 * (a + b)
            cdf = (P * ndtr((mid[:, None] - MU) / SD)).sum(1)
            left = cdf < lv
            a = np.where(left, mid, a)
            b = np.where(left, b, mid)
        res.append(0.5 * (a + b))
    return res
