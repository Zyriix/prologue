import copy
import glob as glob_module
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
import itertools
from typing import Iterator, Iterable, List, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from accelerate import Accelerator
from omegaconf import OmegaConf


def build_ar_logit_mask(vis_pos_mask, sem_pos_mask, vis_cb_size, sem_cb_size):
    """Merge visual/semantic per-position masks into a single ``[T, vis_cb+sem_cb]`` AR logit mask."""
    if vis_pos_mask is None and sem_pos_mask is None:
        return None
    ar_vocab = vis_cb_size + sem_cb_size
    parts = []
    if sem_pos_mask is not None:
        sem_full = torch.full((sem_pos_mask.shape[0], ar_vocab), float('-inf'))
        sem_full[:, vis_cb_size:vis_cb_size + sem_cb_size] = sem_pos_mask
        parts.append(sem_full)
    if vis_pos_mask is not None:
        vis_full = torch.full((vis_pos_mask.shape[0], ar_vocab), float('-inf'))
        vis_full[:, :vis_cb_size] = vis_pos_mask
        parts.append(vis_full)
    return torch.cat(parts, dim=0) if parts else None


def load_config():
    """OmegaConf merge of ``--config`` / ``--configs`` (comma list, left-to-right) plus CLI ``key=value`` overrides."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cli = OmegaConf.from_cli()
    paths_str = cli.pop("--configs", None) or cli.pop("--config", None)
    if paths_str is None:
        raise ValueError("Must provide --config or --configs")
    paths = [p.strip() for p in str(paths_str).split(",") if p.strip()]
    conf = OmegaConf.merge(*[OmegaConf.load(p) for p in paths])
    for k, v in cli.items():
        OmegaConf.update(conf, k, v)
    return conf


def print0(*args, **kwargs):
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        print(*args, **kwargs)


# ============================================================================
# Phase / Target Training System
# ============================================================================

class Target(NamedTuple):
    DO_AE: bool = False
    DO_L2: bool = False
    DO_L1: bool = False
    DO_LPIPS: bool = False
    DO_GAN_G: bool = False
    DO_GAN_D: bool = False
    DO_PRIOR_AR: bool = False
    DO_PRIOR_ENC: bool = False

class Phase(NamedTuple):
    num_steps: int
    targets: List[Target]
    internal_steps: List[int]

def parse_phases(phases_str):
    phases = []
    for phase_str in phases_str.split(' '):
        num_steps, targets_str, internal_steps_str = phase_str.split(':')
        num_steps = int(num_steps)
        targets = [Target(**{k: True for obj in target_str.split(',') for k in obj.split('-') }) for target_str in targets_str.split(',')]
        internal_steps = [int(step) for step in internal_steps_str.split(',')]
        phases.append(Phase(num_steps, targets, internal_steps))
    return phases

def parse_training_config_from_phases(phases):
    train_ae = False
    train_ar = False
    use_lpips_loss = False
    use_gan_loss = False
    train_prior_enc = False
    for phase in phases:
        for target in phase.targets:
            if target.DO_L1 or target.DO_L2 or target.DO_LPIPS or target.DO_GAN_G :
                train_ae = True
            if target.DO_PRIOR_AR or target.DO_PRIOR_ENC:
                train_ar = True
            if target.DO_LPIPS:
                use_lpips_loss = True
            if target.DO_GAN_G or target.DO_GAN_D:
                use_gan_loss = True
            if target.DO_PRIOR_ENC:
                train_prior_enc = True
    return train_ae, train_ar, use_lpips_loss, use_gan_loss, train_prior_enc

def get_phase(global_step, phases, phase_step_accum, gan_start=0):
    target = None
    for phase_idx, phase_step in enumerate(phase_step_accum):
        if global_step <= phase_step:
            internel_step = (global_step - phase_step_accum[phase_idx-1]) if phase_idx > 0 else global_step
            internel_accumulate = list(itertools.accumulate(phases[phase_idx].internal_steps))
            internel_step = internel_step % internel_accumulate[-1]
            for inner_idx in range(len(internel_accumulate)):
                if internel_step < internel_accumulate[inner_idx]:
                    target = phases[phase_idx].targets[inner_idx]
                    break
            if target is not None:
                break

    DO_AE = any([target.DO_L2, target.DO_L1, target.DO_LPIPS, target.DO_GAN_G])
    target = Target(DO_L1=target.DO_L1,
                    DO_L2=target.DO_L2,
                    DO_LPIPS=target.DO_LPIPS,
                    DO_GAN_G=target.DO_GAN_G and (global_step >= gan_start),
                    DO_GAN_D=target.DO_GAN_D and (global_step >= gan_start),
                    DO_PRIOR_AR=target.DO_PRIOR_AR,
                    DO_PRIOR_ENC=target.DO_PRIOR_ENC,
                    DO_AE=DO_AE)
    return phase_idx, inner_idx, target, internel_step


# ============================================================================
# Learning Rate Schedulers
# ============================================================================

def get_linear_schedule_with_warmup_peak(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_peak_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """Linear warmup -> flat peak -> linear decay (``base_lr`` -> ``end_lr``)."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + num_peak_steps:
            return 1.0
        else:
            decay_steps = num_training_steps - num_warmup_steps - num_peak_steps
            progress = float(current_step - num_warmup_steps - num_peak_steps) / float(max(1, decay_steps))
            progress = min(progress, 1.0)
            ratio = 1.0 - progress
            return (end_lr + (base_lr - end_lr) * ratio) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# Trie structure for computing data conditional entropy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class TrieNode:
    count: int = 0
    children: Dict[int, "TrieNode"] = field(default_factory=dict)

def trie_insert(root: TrieNode, seq: List[int], max_depth: int) -> None:
    """Insert a sequence into the Trie up to max_depth."""
    node = root
    for tok in seq[:max_depth]:
        nxt = node.children.get(tok)
        if nxt is None:
            nxt = TrieNode()
            node.children[tok] = nxt
        nxt.count += 1
        node = nxt

def entropy_from_counts(counts: List[int], log_base: float = 2.0) -> float:
    """Compute entropy from a list of counts."""
    if log_base <= 0:
        raise ValueError("log_base must be > 0")
    T = int(sum(counts))
    if T <= 0:
        return float("nan")
    s = 0.0
    for c in counts:
        if c > 0:
            s += c * math.log(c)
    denom = math.log(log_base) if log_base != math.e else 1.0
    return (math.log(T) - (s / float(T))) / denom

def trie_conditional_entropy_all_positions(root: TrieNode, max_depth: int, log_base: float = 2.0) -> Tuple[List[float], List[int]]:
    """Per-position conditional entropy H(X_d | X_<d); returns (H_cond[d], num_contexts[d])."""
    if max_depth <= 0:
        return [], []
    
    H_cond: List[float] = []
    num_contexts: List[int] = []
    
    # Position 0: H(X_0)
    root_child_counts = [int(ch.count) for ch in root.children.values()]
    H_cond.append(float(entropy_from_counts(root_child_counts, log_base=log_base)))
    num_contexts.append(1)
    
    # Position 1 to max_depth-1: H(X_d | X_<d)
    for d in range(1, max_depth):
        target_depth = d
        total_T = 0
        weighted_sum = 0.0
        ctx_cnt = 0
        stack: List[Tuple[TrieNode, int]] = [(root, 0)]
        
        while stack:
            node, depth = stack.pop()
            if depth == target_depth:
                ctx_T = int(node.count)
                if ctx_T > 0:
                    child_counts = [int(ch.count) for ch in node.children.values()]
                    if len(child_counts) == 0:
                        continue
                    Hc = entropy_from_counts(child_counts, log_base=log_base)
                    total_T += ctx_T
                    weighted_sum += ctx_T * Hc
                    ctx_cnt += 1
                continue
            if depth > target_depth:
                continue
            for child in node.children.values():
                stack.append((child, depth + 1))
        
        H = weighted_sum / float(total_T) if total_T > 0 else float("nan")
        H_cond.append(float(H))
        num_contexts.append(int(ctx_cnt))
    
    return H_cond, num_contexts

def _entropy_from_logits(logits: torch.Tensor, log_base: float = 2.0) -> torch.Tensor:
    """Masked categorical entropy from logits (bits when ``log_base == 2``)."""
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ent_nats = -torch.nan_to_num(probs * log_probs, nan=0.0).sum(dim=-1)  # 0*log0 -> 0
    if log_base == math.e:
        return ent_nats
    return ent_nats / math.log(log_base)


def plot_data_conditional_entropy(
    out_path: str,
    H_cond: list | None,
    log_base: float = 2.0,
    codebook_size: int | None = None,
    title_prefix: str = "Data conditional entropy",
) -> bool:
    """Save bar plot of H(X_d | X_<d) to ``out_path``; optional ``codebook_size`` adds reference lines."""
    if plt is None:
        return False
    if H_cond is None or len(H_cond) == 0:
        return False

    N = len(H_cond)
    xs = list(range(N))

    fig = plt.figure(figsize=(max(10, N * 0.05), 4))
    plt.bar(xs, H_cond, width=1.0, color="#E45756", label="H(X_d | X_<d)", edgecolor='none')

    plt.grid(True, linestyle="--", alpha=0.3, axis='y')

    # Theoretical reference lines (fixed codebook)
    if codebook_size is not None and codebook_size > 0:
        max_entropy = math.log(codebook_size) / math.log(log_base)
        plt.axhline(y=max_entropy, color='red', linestyle='--', linewidth=2.0,
                   label=f'Max H (Independent): {max_entropy:.2f}', alpha=0.8, zorder=10)
        if N > 0:
            codes_per_pos = codebook_size / N
            if codes_per_pos >= 1.0:
                split_entropy = math.log(codes_per_pos) / math.log(log_base)
                plt.axhline(y=split_entropy, color='orange', linestyle='-.', linewidth=2.0,
                           label=f'Split Codebook ({codebook_size}/{N}={codes_per_pos:.1f}): {split_entropy:.2f}',
                           alpha=0.8, zorder=10)

    plt.xlim(-0.5, max(0, N - 0.5))

    step = max(1, N // 16)
    xticks = list(range(0, N, step))
    if (N - 1) not in xticks:
        xticks.append(N - 1)
    plt.xticks(xticks)

    # Compute the y-axis range from data + reference lines
    valid_vals = [v for v in H_cond if isinstance(v, (int, float)) and not math.isnan(v)]
    if codebook_size is not None and codebook_size > 0:
        valid_vals.append(math.log(codebook_size) / math.log(log_base))
        if N > 0 and codebook_size / N >= 1.0:
            valid_vals.append(math.log(codebook_size / N) / math.log(log_base))

    if valid_vals:
        ymax = max(valid_vals)
        ymax = max(ymax, 0.0)
        y_max_tick = int(math.ceil(ymax * 1.1))  # leave 10% headroom
        plt.yticks(list(range(0, y_max_tick + 1, max(1, y_max_tick // 5))))
        plt.ylim(0, y_max_tick)
    else:
        plt.ylim(bottom=0)

    plt.xlabel("position d (0-based)")
    plt.ylabel(f"conditional entropy (log_base={log_base})")
    plt.title(f"{title_prefix}: H(X_d | X_<d)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return True

def plot_ar_prefix_conditional_entropy(
    out_path: str,
    H: list | None,
    log_base: float = 2.0,
    codebook_size: int | None = None,
    title_prefix: str = "AR predictive conditional entropy",
) -> bool:
    """Save per-position entropy curve to ``out_path``; optional ``codebook_size`` adds reference lines.
    """
    if plt is None:
        return False
    if H is None or len(H) <= 0:
        return False

    plot_len = len(H)
    xs = list(range(plot_len))
    fig = plt.figure(figsize=(max(10, plot_len * 0.05), 4))

    plt.bar(xs, H, width=1.0, color="#4C78A8", label="H_ar(X_d | X_<d)", edgecolor='none')

    plt.grid(True, linestyle="--", alpha=0.3, axis='y')

    # Reference line (fixed codebook)
    if codebook_size is not None and codebook_size > 0:
        max_entropy = math.log(codebook_size) / math.log(log_base)
        plt.axhline(y=max_entropy, color='red', linestyle='--', linewidth=2.0,
                   label=f'Max H (K={codebook_size}): {max_entropy:.2f}', alpha=0.8, zorder=10)

    plt.xlim(-0.5, max(0, plot_len - 0.5))

    step = max(1, plot_len // 16)
    xticks = list(range(0, plot_len, step))
    if (plot_len - 1) not in xticks:
        xticks.append(plot_len - 1)
    plt.xticks(xticks)

    valid_vals = [v for v in H if isinstance(v, (int, float)) and not math.isnan(v)]
    if codebook_size is not None and codebook_size > 0:
        valid_vals.append(math.log(codebook_size) / math.log(log_base))

    if valid_vals:
        ymax = max(max(valid_vals), 0.0)
        y_max_tick = int(math.ceil(ymax * 1.1))
        plt.yticks(list(range(0, y_max_tick + 1, max(1, y_max_tick // 5))))
        plt.ylim(0, y_max_tick)
    else:
        plt.ylim(bottom=0)

    plt.xlabel("position d (0-based)")
    plt.ylabel(f"conditional entropy (log_base={log_base})")
    plt.title(f"{title_prefix}, d=0..{plot_len-1} (log_base={log_base})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def compute_posterior_entropy_from_logits(logits: torch.Tensor, log_base: float = 2.0) -> torch.Tensor:
    """Posterior entropy ``-E_q log q`` from ``[B, L, K]`` logits."""
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ent_nats = -torch.nan_to_num(probs * log_probs, nan=0.0).sum(dim=-1)  # 0*log0 -> 0
    if log_base == math.e:
        return ent_nats
    return ent_nats / math.log(log_base)


def compute_aggregated_entropy_from_counts(count_matrix: torch.Tensor, log_base: float = 2.0) -> torch.Tensor:
    """Aggregated-posterior entropy ``-E_z log q(z)`` from ``[L, K]`` counts."""
    probs = count_matrix.float() / count_matrix.sum(dim=-1, keepdim=True).clamp(min=1.0)
    log_probs = torch.log(probs.clamp(min=1e-10))
    ent_nats = -(probs * log_probs).sum(dim=-1)
    if log_base == math.e:
        return ent_nats
    return ent_nats / math.log(log_base)


def plot_posterior_entropy(
    sample_entropy: list | None,
    aggregated_entropy: list | None,
    *,
    accelerator: Accelerator,
    save_dir: str,
    global_step: int,
    rFID: float = 0.0,
    gFID: float = 0.0,
    log_base: float = 2.0,
    codebook_size: int | None = None,
    out_name: str = "ae_pos_posterior_entropy.png",
) -> None:
    """Save bar plot of per-position sample/aggregated entropy."""
    if plt is None or not accelerator.is_main_process:
        return
    if sample_entropy is None and aggregated_entropy is None:
        return

    len_sample = len(sample_entropy) if sample_entropy is not None else 0
    len_agg = len(aggregated_entropy) if aggregated_entropy is not None else 0
    L = max(len_sample, len_agg)
    if L <= 0:
        return

    out_dir = Path(save_dir) / "analysis_ae" / f"Step={global_step+1}-rFID={rFID:.4f}-gFID={gFID:.4f}"
    out_dir.mkdir(exist_ok=True, parents=True)
    fig_path = out_dir / out_name

    # Thin bars
    fig = plt.figure(figsize=(max(10, L * 0.05), 4))
    xs = list(range(L))

    # Pick bar offsets/widths based on which series are present
    if sample_entropy is not None and aggregated_entropy is not None:
        # Both series: side-by-side with offsets
        if len(sample_entropy) > 0:
            plt.bar([x - 0.2 for x in xs[:len_sample]], sample_entropy,
                    width=0.4, color="#F58518", label="Sample Entropy", edgecolor='none')
        if len(aggregated_entropy) > 0:
            plt.bar([x + 0.2 for x in xs[:len_agg]], aggregated_entropy,
                    width=0.4, color="#4C78A8", label="Aggregated Entropy", edgecolor='none')
    else:
        # Only one series: centered bars
        if sample_entropy is not None and len(sample_entropy) > 0:
            plt.bar(xs[:len_sample], sample_entropy,
                    width=1.0, color="#F58518", label="Sample Entropy", edgecolor='none')
        if aggregated_entropy is not None and len(aggregated_entropy) > 0:
            plt.bar(xs[:len_agg], aggregated_entropy,
                    width=1.0, color="#4C78A8", label="Aggregated Entropy", edgecolor='none')

    plt.grid(True, linestyle="--", alpha=0.3, axis='y')

    # Theoretical reference lines (fixed codebook)
    if codebook_size is not None and codebook_size > 0:
        max_entropy = math.log(codebook_size) / math.log(log_base)
        plt.axhline(y=max_entropy, color='red', linestyle='--', linewidth=2.0,
                   label=f'Max Entropy (Uniform over {codebook_size}): {max_entropy:.2f}', alpha=0.8, zorder=10)
        if L > 0:
            codes_per_pos = codebook_size / L
            if codes_per_pos >= 1.0:
                split_entropy = math.log(codes_per_pos) / math.log(log_base)
                plt.axhline(y=split_entropy, color='orange', linestyle='-.', linewidth=2.0,
                           label=f'Split Codebook ({codebook_size}/{L}={codes_per_pos:.1f}): {split_entropy:.2f}',
                           alpha=0.8, zorder=10)
    plt.xlim(-0.5, max(0, L - 0.5))

    # Thin out x-axis ticks
    step = max(1, L // 16)
    xticks = list(range(0, L, step))
    if (L - 1) not in xticks:
        xticks.append(L - 1)
    plt.xticks(xticks)

    # Compute the y-axis range
    valid_vals = []
    if sample_entropy is not None:
        valid_vals += [v for v in sample_entropy if isinstance(v, (int, float)) and not math.isnan(v)]
    if aggregated_entropy is not None:
        valid_vals += [v for v in aggregated_entropy if isinstance(v, (int, float)) and not math.isnan(v)]
    if codebook_size is not None and codebook_size > 0:
        valid_vals.append(math.log(codebook_size) / math.log(log_base))
        if L > 0 and codebook_size / L >= 1.0:
            valid_vals.append(math.log(codebook_size / L) / math.log(log_base))

    if valid_vals:
        ymax = max(valid_vals)
        ymax = max(ymax, 0.0)
        y_max_tick = int(math.ceil(ymax * 1.1))  # leave 10% headroom
        plt.yticks(list(range(0, y_max_tick + 1, max(1, y_max_tick // 5))))
        plt.ylim(0, y_max_tick)
    else:
        plt.ylim(bottom=0)
    
    plt.xlabel("position d (0-based)")
    plt.ylabel(f"entropy (log_base={log_base})")
    plt.title(f"Aggregated Posterior Entropy per Position")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=200)
    plt.close(fig)
    
    accelerator.log({"analysis/ae_pos_posterior_entropy": wandb.Image(str(fig_path)), "global_step": global_step + 1}, step=global_step+1)


def plot_codebook_usage(
    codebook_usage: torch.Tensor | None,
    *,
    accelerator: Accelerator,
    save_dir: str,
    global_step: int,
    rFID: float = 0.0,
    gFID: float = 0.0,
    out_name: str = "ae_pos_code_usage_rate.png",
) -> None:
    """Save per-position unique-code / K usage from ``codebook_usage[L, K]`` counts."""
    if plt is None or not accelerator.is_main_process:
        return
    if codebook_usage is None or codebook_usage.dim() != 2:
        return

    # Per-position codebook utilization
    used_per_pos = (codebook_usage > 0).sum(dim=1).float()  # [L]
    usage = (used_per_pos / float(codebook_usage.shape[1])).detach().cpu().tolist()

    L = int(len(usage))
    if L <= 0:
        return

    out_dir = Path(save_dir) / "analysis_ae" / f"Step={global_step+1}-rFID={rFID:.4f}-gFID={gFID:.4f}"
    out_dir.mkdir(exist_ok=True, parents=True)
    fig_path = out_dir / out_name

    # Bar plot
    fig = plt.figure(figsize=(max(10, L * 0.05), 4))
    plt.bar(list(range(L)), usage, color="#54A24B", width=1.0, edgecolor='none')
    plt.grid(True, linestyle="--", alpha=0.3, axis='y')
    plt.xlim(-0.5, max(0, L - 0.5))
    plt.ylim(0.0, 1.05)
    # Thin out x-axis ticks
    step = max(1, L // 16)
    xticks = list(range(0, L, step))
    if (L - 1) not in xticks:
        xticks.append(L - 1)
    plt.xticks(xticks)
    plt.xlabel("position d (0-based)")
    plt.ylabel(f"unique / K (K={codebook_usage.shape[1]})")

    plt.title("Codebook Usage Rate per Position")
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=200)
    plt.close(fig)
    accelerator.log({"analysis/ae_pos_code_usage_rate": wandb.Image(str(fig_path)), "global_step": global_step + 1}, step=global_step+1)

def seed_everything(seed):
    """Set python/numpy/torch (CPU+CUDA)/hash seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_worker_init_fn(base_seed: int):
    base_seed = int(base_seed) % (2**32)

    def _init(worker_id: int):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        worker_seed = (base_seed + worker_id + 1000 * rank) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init


def load_accelerate_weights_only(
    *,
    accelerator: Accelerator,
    input_dir: str,
    strict: bool = True,
    map_location: str | torch.device | None = "cpu",
) -> None:
    """Load only ``model*.safetensors`` from an ``accelerator.save_state()`` dir (no optim/RNG/dl state)."""
    input_dir = os.path.expanduser(str(input_dir))
    if not os.path.isdir(input_dir):
        raise ValueError(f"Tried to load weights from {input_dir} but folder does not exist")

    from accelerate.state import DistributedType
    from accelerate.utils import load as accelerate_load, load_fsdp_model
    from accelerate.checkpointing import SAFE_MODEL_NAME, MODEL_NAME, load_model

    device_str = "cpu" if map_location in (None, "cpu") else str(map_location)
    input_path = Path(input_dir)

    # Iterate over accelerator._models to preserve save_state ordering.
    models = getattr(accelerator, "_models", None) or []
    if len(models) == 0:
        print0("[warn] No models registered in accelerator; skip loading weights.")
        return

    for i, model in enumerate(models):
        if accelerator.distributed_type == DistributedType.FSDP:
            load_fsdp_model(accelerator.state.fsdp_plugin, accelerator, model, input_dir, i)
            continue

        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
            model.load_checkpoint(
                input_dir,
                ckpt_id,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
                load_module_strict=bool(strict),
            )
            continue

        if accelerator.distributed_type == DistributedType.MEGATRON_LM:
            raise NotImplementedError(
                "resume_train=False (weights-only) is not supported for Megatron-LM checkpoints in this script."
            )

        ending = f"_{i}" if i > 0 else ""
        safe_file = input_path / f"{SAFE_MODEL_NAME}{ending}.safetensors"
        if safe_file.exists():
            load_model(model, safe_file, strict=bool(strict), device=device_str)
            continue

        bin_file = input_path / f"{MODEL_NAME}{ending}.bin"
        if bin_file.exists():
            state_dict = accelerate_load(bin_file, map_location=map_location)
            model.load_state_dict(state_dict, strict=bool(strict))
            continue

        raise FileNotFoundError(
            f"Could not find model weights for model index {i} under {input_dir}. "
            f"Tried: {safe_file.name} and {bin_file.name}"
        )


@torch.no_grad()
def draw_data_conditional_entropy(
    trie_root: TrieNode | None,
    *,
    idx: torch.Tensor | None = None,
    accelerator: Accelerator,
    save_dir: str,
    global_step: int,
    log_base: float = 2.0,
    finalize: bool = False,
    rFID: float = 0.0,
    gFID: float = 0.0,
    max_depth: int = 0,
    codebook_size: int | None = None,
) -> TrieNode | None:
    """Trie builder for data conditional entropy: idx chunks until ``finalize=True``, then plot + wandb log."""
    if not finalize:
        if idx is None:
            return trie_root
        idx_all = accelerator.gather(idx.detach())  # [B_total, L]
        if accelerator.is_main_process:
            if trie_root is None:
                trie_root = TrieNode()
            L = int(idx_all.shape[1])
            if max_depth <= 0:
                max_depth = L
            idx_cpu = idx_all.cpu().tolist()
            for seq in idx_cpu:
                trie_insert(trie_root, seq, max_depth=max_depth)
        return trie_root

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process or trie_root is None:
        return trie_root
    if max_depth <= 0:
        max_depth = 256

    H_cond, num_contexts = trie_conditional_entropy_all_positions(
        trie_root, max_depth=max_depth, log_base=log_base
    )
    
    if len(H_cond) == 0:
        return trie_root

    out_dir = Path(save_dir) / "analysis_ae" / f"Step={global_step+1}-rFID={rFID:.4f}-gFID={gFID:.4f}"
    out_dir.mkdir(exist_ok=True, parents=True)
    fig_path = out_dir / "ae_data_conditional_entropy.png"

    saved = plot_data_conditional_entropy(
        out_path=str(fig_path),
        H_cond=H_cond,
        log_base=log_base,
        codebook_size=codebook_size,
        title_prefix="AE Data Conditional Entropy",
    )

    if saved and wandb is not None:
        accelerator.log(
            {"analysis/ae_data_conditional_entropy": wandb.Image(str(fig_path)), "global_step": global_step + 1},
            step=global_step+1
        )
    if len(H_cond) > 0:
        mean_H = float(np.nanmean(np.array(H_cond, dtype=np.float64)))
        accelerator.log(
            {"analysis/ae_data_cond_entropy_mean": mean_H, "global_step": global_step + 1},
            step=global_step+1
        )
    return trie_root

@torch.no_grad()
def draw_conditional_entropy(
    acc: dict,
    *,
    logits: torch.Tensor | None = None,
    accelerator: Accelerator,
    save_dir: str,
    global_step: int,
    log_base: float = 2.0,
    finalize: bool = False,
    rFID: float = 0.0,
    gFID: float = 0.0,
    codebook_size: int | None = None,
) -> None:
    """Accumulate logit entropy from existing logits; ``finalize=True`` reduces + plots + wandb-logs."""
    device = accelerator.device
    if not finalize:
        if logits is None:
            return

        ent = _entropy_from_logits(logits, log_base=log_base)  # [B, L]
        L = int(ent.shape[1])
        if acc.get("ent_sum") is None:
            acc["ent_sum"] = torch.zeros(L, dtype=ent.dtype, device=device)
            acc["ent_cnt"] = torch.zeros(L, dtype=torch.long, device=device)
        elif int(acc["ent_sum"].shape[0]) < L:
            pad = L - int(acc["ent_sum"].shape[0])
            acc["ent_sum"] = torch.cat(
                [acc["ent_sum"], torch.zeros(pad, dtype=acc["ent_sum"].dtype, device=device)], dim=0,
            )
            acc["ent_cnt"] = torch.cat(
                [acc["ent_cnt"], torch.zeros(pad, dtype=torch.long, device=device)], dim=0,
            )
        acc["ent_sum"][:L] += ent.sum(dim=0)
        acc["ent_cnt"][:L] += int(ent.shape[0])
        return

    # finalize mode
    accelerator.wait_for_everyone()
    if acc.get("ent_sum") is not None and acc.get("ent_cnt") is not None:
        acc["ent_sum"] = accelerator.reduce(acc["ent_sum"], reduction='sum')
        acc["ent_cnt"] = accelerator.reduce(acc["ent_cnt"], reduction='sum')

    if not accelerator.is_main_process:
        return

    H = None
    if acc.get("ent_sum") is not None and acc.get("ent_cnt") is not None:
        denom = acc["ent_cnt"].clamp(min=1).to(acc["ent_sum"].dtype)
        H = (acc["ent_sum"] / denom).detach().cpu().tolist()

    out_dir = Path(save_dir) / "analysis_ar" / f"Step={global_step+1}-rFID={rFID:.4f}-gFID={gFID:.4f}"
    out_dir.mkdir(exist_ok=True, parents=True)
    fig_path = out_dir / "ar_prefix_conditional_entropy.png"
    saved = plot_ar_prefix_conditional_entropy(
        out_path=str(fig_path),
        H=H,
        log_base=log_base,
        codebook_size=codebook_size,
    )
    if saved:
        accelerator.log({"analysis/ar_prefix_conditional_entropy": wandb.Image(str(fig_path)), "global_step": global_step + 1}, step=global_step+1)
    if H is not None and len(H) > 0:
        accelerator.log(
            {
                "analysis/ar_entropy_mean_per_pos": float(np.nanmean(np.array(H, dtype=np.float64))),
                "global_step": global_step + 1,
            },
            step=global_step+1,
        )

def generate_uniform_labels(
    *,
    num_samples: int,
    num_classes: int,
    accelerator: Accelerator,
    exclude_uncond: bool = True,
) -> torch.Tensor:
    """Uniform class label indices for this rank (excludes uncond class by default)."""
    num_valid_classes = num_classes - 1 if exclude_uncond else num_classes
    all_classes = list(range(num_valid_classes)) * (num_samples // num_valid_classes + 1)
    all_classes = all_classes[:num_samples]
    all_classes_tensor = torch.tensor(all_classes, dtype=torch.long)

    rank = accelerator.process_index
    num_devices = accelerator.num_processes
    samples_per_rank = num_samples // num_devices
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank < num_devices - 1 else num_samples

    return all_classes_tensor[start_idx:end_idx].to(accelerator.device)


class InfiniteIterator(Iterator):
    def __init__(self, iterable: Iterable, dl_generator: torch.Generator = None):
        self.iterable = iterable
        self.dl_generator = dl_generator
        self._pre_epoch_gen_state = (
            dl_generator.get_state().clone() if dl_generator is not None else None
        )
        self._it = iter(iterable)
        self.total_yielded = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self._it)
        except StopIteration:
            if self.dl_generator is not None:
                self._pre_epoch_gen_state = self.dl_generator.get_state().clone()
            self._it = iter(self.iterable)
            item = next(self._it)
        self.total_yielded += 1
        return item

# ============================================================================
# File / Checkpoint Utils
# ============================================================================

def safe_remove_file(path: str):
    """Remove a single file safely: warn on failure but never raise."""
    try:
        if path is not None and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Warning: Failed to remove {path}: {e}")


def save_training_state(accelerator, save_path, extra_state, **updates):
    """Save accelerator state + extra state for fully consistent resume."""
    if updates:
        extra_state.update(updates)
    accelerator.save_state(save_path)
    if accelerator.is_main_process:
        torch.save(extra_state, os.path.join(save_path, "extra_state.pt"))
    accelerator.wait_for_everyone()


def remove_old_best_checkpoints(ckpt_dir: str, metric_type: str = "gFID"):
    """Delete older ``best-*-{metric_type}=*`` checkpoints under ``ckpt_dir``."""
    pattern = os.path.join(ckpt_dir, f"best-*-{metric_type}=*")
    old_best_ckpts = glob_module.glob(pattern)

    for old_ckpt in old_best_ckpts:
        try:
            if os.path.isdir(old_ckpt):
                shutil.rmtree(old_ckpt)
                print(f"Removed old best checkpoint: {os.path.basename(old_ckpt)}")
        except Exception as e:
            print(f"Warning: Failed to remove {old_ckpt}: {e}")


# ============================================================================
# Image Processing Utils
# ============================================================================

def patchify(x, patch_size):
    x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
    return x


def img_uint8_to_norm(x):
    return x.float() / 127.5 - 1.0


def unpatchify(x, image_size, patch_size):
    x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size, w=image_size//patch_size)
    return x


def img_denormalize(x):
    return x.clamp(-1, 1) * 0.5 + 0.5


def img_norm_to_uint8(x):
    return torch.clamp(127.5 * x + 128.0, 0, 255).byte()


# ============================================================================
# FID Utils
# ============================================================================

def adm_fid_evaluator(sample_cached_path, gt_cache_path, config, accelerator: Accelerator, compute_is=False):
    if not os.path.exists(gt_cache_path):
        raise FileNotFoundError(f"Ground-truth cache not found: {gt_cache_path}")
    if not os.path.exists(sample_cached_path):
        raise FileNotFoundError(f"Sample cache not found: {sample_cached_path}")

    fid_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_fid.py')
    env = os.environ.copy()
    cmd = [sys.executable, fid_script, "--ref_batch", gt_cache_path, "--sample_batch", sample_cached_path, "--batch_size", str(config.eval_batch_size)]
    if compute_is:
        cmd.append("--compute_is")
    print0(f"Running FID evaluation via {fid_script}..." + (" (with IS)" if compute_is else ""))

    FID = 0.0
    IS = 0.0
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        if line:
            print0(line, flush=True)
        if line.startswith("FID_RESULT:"):
            try:
                FID = float(line.split("FID_RESULT:")[1].strip())
            except ValueError:
                pass
        elif line.startswith("IS_RESULT:"):
            try:
                IS = float(line.split("IS_RESULT:")[1].strip())
            except ValueError:
                pass
    retcode = process.wait()
    if retcode != 0 and FID == 0.:
        print0(f"eval_fid.py exited with code {retcode} and no FID_RESULT was parsed.")

    if compute_is:
        return FID, IS
    return FID


# ============================================================================
# Training Utils
# ============================================================================

@torch.no_grad()
@torch._dynamo.disable
def _unwrap(model):
    """Unwrap torch.compile / DDP wrappers to access the raw nn.Module."""
    while hasattr(model, '_orig_mod'):
        model = model._orig_mod
    while hasattr(model, 'module'):
        model = model.module
    return model


@torch.no_grad()
@torch._dynamo.disable
def ema_update(model, ema_model, ema_rate):
    if model is None or ema_model is None:
        return
    for p, ema_p in zip(model.parameters(), ema_model.parameters()):
        ema_p.copy_(p.detach().lerp(ema_p, ema_rate))


def sync_gradients(model, sub_modules=None):
    import torch.distributed as dist
    if not dist.is_initialized():
        return
    params = []
    if sub_modules is not None:
        for name in sub_modules:
            params.extend(getattr(model, name).parameters())
    else:
        params = list(model.parameters())
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def toggle_require_grad(model, grads=True, accelerator=None, sub_modules=None):
    if model is None:
        return
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    elif hasattr(model, "_orig_mod"):
        model = model._orig_mod

    if sub_modules is not None:
        for name in sub_modules:
            getattr(model, name).requires_grad_(grads)
    elif hasattr(model, "requires_grad_"):
        model.requires_grad_(grads)
    else:
        for p in model.parameters():
            p.requires_grad_(grads)


def toggle_train_eval(model, train=True, accelerator=None, sub_modules=None):
    if model is None: return
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    elif hasattr(model, "_orig_mod"):
        model = model._orig_mod

    if sub_modules is not None:
        for name in sub_modules:
            getattr(model, name).train(mode=train)
    elif hasattr(model, "train"):
        model.train(mode=train)


def zero_nan_gradients(model, accelerator=None):
    if model is None: return
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    elif hasattr(model, "_orig_mod"):
        model = model._orig_mod
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.nan_to_num_(nan=0.0, posinf=1e5, neginf=-1e5)


def calc_grad_norm(
    named_models: dict,
    global_step: int,
    grad_norm_freq: int,
    accelerator=None,
) -> dict:
    """Per-parameter grad L2 norms as a flat dict for wandb (only on ``(step+1) % freq == 0``)."""
    if grad_norm_freq <= 0 or (global_step + 1) % grad_norm_freq != 0:
        return {}

    result = {}
    for group, model in named_models.items():
        if model is None:
            continue
        # unwrap DDP wrapper, then strip torch.compile's OptimizedModule
        raw = accelerator.unwrap_model(model) if accelerator is not None else model
        while hasattr(raw, "_orig_mod"):
            raw = raw._orig_mod
        sq_sum = 0.0
        for name, param in raw.named_parameters():
            if param.grad is not None:
                pnorm = param.grad.norm().item()
                result[f"Gradient_Norm/{group}/{name}"] = pnorm
                sq_sum += pnorm ** 2
        if sq_sum > 0.0:
            result[f"Gradient_Norm/{group}/_total"] = sq_sum ** 0.5
    return result


def save_tensor_image_png_pdf(tensor, png_path: str, dpi: float = 300.0) -> None:
    """Save ``[N, C, H, W]`` in [0, 1] to ``png_path`` and a sibling PDF (figure-friendly).
    """
    import torchvision.utils

    torchvision.utils.save_image(tensor, png_path)
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    from PIL import Image

    im = Image.open(png_path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im.close()
        im = bg
    else:
        im = im.convert("RGB")
    im.save(pdf_path, "PDF", resolution=dpi)
    im.close()
