import copy
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils
try:
    import wandb
except ImportError:                                              # inference-only envs
    wandb = None
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace

from models import ARModel, Decoder, PrologueQuantizer, insert_eos_token
from utils import (
    InfiniteIterator,
    adm_fid_evaluator,
    build_ar_logit_mask,
    ema_update,
    img_denormalize,
    img_norm_to_uint8,
    make_worker_init_fn,
    remove_old_best_checkpoints,
    save_training_state,
    toggle_train_eval,
    unpatchify,
    zero_nan_gradients,
    calc_grad_norm,
    safe_remove_file,
    generate_uniform_labels,
    seed_everything,
    get_linear_schedule_with_warmup_peak,
    draw_conditional_entropy,
    print0,
    _unwrap,
)
from util_model_profile import print_model_stats

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False




def _default_pretoken_dir(tokenizer_ckpt_path: str) -> str:
    """Map ``.../experiments/<exp>/<run>/ckpts/<name>`` to ``$PRETOKEN_DIR/<exp>/<run>/<name>`` (default ``pretoken``)."""
    dst = os.environ.get("PRETOKEN_DIR", "pretoken")
    parts = tokenizer_ckpt_path.replace("\\", "/").split("/")
    try:
        idx = len(parts) - 1 - parts[::-1].index("experiments")
        exp_name = parts[idx + 1] if len(parts) > idx + 1 else None
        run_id = parts[idx + 2] if len(parts) > idx + 2 else None
        ckpt_name = parts[idx + 4] if len(parts) > idx + 4 else None
        if exp_name:
            result = os.path.join(dst, exp_name)
            if run_id:
                result = os.path.join(result, run_id)
            if ckpt_name:
                result = os.path.join(result, ckpt_name)
            return result
    except (ValueError, IndexError):
        pass
    return dst


class PretokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens: torch.Tensor, z_len: int, crop_type: str = None):
        self.tokens = tokens
        self.z_len = int(z_len)
        self.crop_type = crop_type
        row_len = self.tokens.shape[1]
        self.is_ten_crop = (self.crop_type == 'ten_crop' and row_len == 10 * z_len + 1)
        print0(f"is_ten_crop: {self.is_ten_crop}")

    def __len__(self):
        return int(self.tokens.shape[0])

    def __getitem__(self, idx: int):
        row = self.tokens[idx]
        if self.is_ten_crop:
            crop_idx = torch.randint(0, 10, (1,)).item()
            start_idx = crop_idx * self.z_len
            tokens = row[start_idx : start_idx + self.z_len]
            label = row[10 * self.z_len]
            return tokens, label
        return row[: self.z_len], row[self.z_len]


def _labels_from_label_idx(label_idx: torch.Tensor, *, num_classes: int, uncond_idx: int) -> torch.Tensor:
    # cond uses first num_classes-1 dims; uncond uses the last dim.
    B = int(label_idx.shape[0])
    labels = torch.zeros((B, num_classes), device=label_idx.device, dtype=torch.float32)
    cond = label_idx != uncond_idx
    if cond.any():
        labels[cond, : num_classes - 1] = F.one_hot(label_idx[cond].to(torch.long), num_classes=num_classes - 1).float()
    if (~cond).any():
        labels[~cond, uncond_idx] = 1.0
    return labels


def _log_load_verify_strict(tag: str, path: str, n_keys_in_sd: int, module: torch.nn.Module) -> None:
    n_mod = len(module.state_dict())
    print0(f"[load verify] {tag}: strict=True path={path} keys_in_ckpt={n_keys_in_sd} keys_in_module={n_mod}")


def _load_quantizer(config, *, ckpt_dir: str):
    q = PrologueQuantizer(**config["Quantizer"])

    enc_file = "model_2.safetensors" if config.ema_reconstruction else "model.safetensors"
    enc_path = os.path.join(ckpt_dir, enc_file)
    if config.ema_reconstruction and not os.path.exists(enc_path):
        raise FileNotFoundError(f"ema_reconstruction=True but EMA encoder weights not found: {enc_path}")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Encoder weights not found: {enc_path}")
    sd = load_file(enc_path)
    q_sd = {k.split("quantizer.", 1)[1]: v for k, v in sd.items() if k.startswith("quantizer.")}
    q_sd.pop("codebook_size_per_pos", None)  # legacy per-pos buffer

    q.load_state_dict(q_sd, strict=True)
    _log_load_verify_strict("visual_quantizer", enc_path, len(q_sd), q)
    return q.eval().requires_grad_(False)


def _load_semantic_quantizer(config, *, ckpt_dir: str):
    """Load the semantic quantizer (prologue prefix codebook) from encoder checkpoint."""
    if config.get("share_semantic_codebook", False):
        return _load_quantizer(config, ckpt_dir=ckpt_dir)
    q = PrologueQuantizer(**config["SemanticQuantizer"])

    enc_file = "model_2.safetensors" if config.ema_reconstruction else "model.safetensors"
    enc_path = os.path.join(ckpt_dir, enc_file)
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Encoder weights not found: {enc_path}")
    sd = load_file(enc_path)
    q_sd = {k.split("semantic_quantizer.", 1)[1]: v for k, v in sd.items() if k.startswith("semantic_quantizer.")}
    q_sd.pop("codebook_size_per_pos", None)
    q.load_state_dict(q_sd, strict=True)
    _log_load_verify_strict("semantic_quantizer", enc_path, len(q_sd), q)
    return q.eval().requires_grad_(False)


def _load_decoder(config, *, ckpt_dir: str):
    dec = Decoder(config).eval().requires_grad_(False)
    dec_file = "model_3.safetensors" if config.ema_reconstruction else "model_1.safetensors"
    dec_path = os.path.join(ckpt_dir, dec_file)
    if config.ema_reconstruction and not os.path.exists(dec_path):
        raise FileNotFoundError(f"ema_reconstruction=True but EMA decoder weights not found: {dec_path}")
    if not os.path.exists(dec_path):
        raise FileNotFoundError(f"Decoder weights not found: {dec_path}")
    sd = load_file(dec_path)
    dec.load_state_dict(sd, strict=True)
    _log_load_verify_strict("decoder", dec_path, len(sd), dec)
    return dec


def _codes_from_indices(quantizer, indices: torch.Tensor, labels: torch.Tensor):
    try:
        return quantizer.get_codes_w_indices(indices, labels)
    except TypeError:
        return quantizer.get_codes_w_indices(indices)


@torch.no_grad()
def sampling(quantizer, dec, ar_model, *, bz: int, class_label: torch.Tensor, config,  ae_label=None):
    ar_raw = _unwrap(ar_model)
    token_ids = ar_raw.sampling(
        bz,
        class_label,
        config.temperature,
        config.topK,
        config.topP,
        config.cfg,
        config.cfg_schedule,
        config.cfg_power,
        config.cache_kv,
        semantic_cfg_schedule=config.get("semantic_cfg_schedule", None),
        semantic_cfg_scale=config.get("semantic_cfg_scale", None),
        semantic_cfg_power=config.get("semantic_cfg_power", None),
        semantic_cfg_start=float(config.get("semantic_cfg_start", 0.0)),
        visual_cfg_schedule=config.get("visual_cfg_schedule", None),
        visual_cfg_scale=config.get("visual_cfg_scale", None),
        visual_cfg_power=config.get("visual_cfg_power", None),
        visual_cfg_start=float(config.get("visual_cfg_start", 1.0)),
        semantic_temperature=float(config.get("semantic_temperature")) if config.get("semantic_temperature") is not None else None,
    )
    prologue = config.get("Prologue", False) and not config.get("share_semantic_codebook", False)
    if prologue:
        eos_len = 1 if bool(config.get("use_eos", False)) and int(config.get("z_len", 0)) > 0 else 0
        visual_ids = token_ids[:, int(config.z_len) + eos_len:]
        quant = _codes_from_indices(quantizer, visual_ids, ae_label)
    else:
        quant = _codes_from_indices(quantizer, token_ids, ae_label)
    output = dec(quant, ae_label)
    return output


# ---------------------------------------------------------------------------
# Extracted sub-routines operating on shared ARContext (SimpleNamespace)
# ---------------------------------------------------------------------------

def _save_checkpoint(ctx, train_loader, extra_training_states, global_step, ckpt_name,
                     remove_old_metric=None, **extra_save_kwargs):
    """Unified checkpoint saving: best / periodic / last."""
    acc = ctx.accelerator
    if remove_old_metric is not None:
        if acc.is_main_process:
            remove_old_best_checkpoints(f"{ctx.save_dir}/ckpts", metric_type=remove_old_metric)
        acc.wait_for_everyone()
    save_path = f'{ctx.save_dir}/ckpts/{ckpt_name}'
    dl_kwargs = {}
    if train_loader is not None:
        dl_kwargs = dict(
            dl_generator=train_loader._pre_epoch_gen_state,
            total_yielded=train_loader.total_yielded,
        )
    save_training_state(acc, save_path, extra_training_states,
                        global_step=global_step, **dl_kwargs, **extra_save_kwargs)
    if remove_old_metric is not None and acc.is_main_process:
        print0(f"Saved new best {remove_old_metric} checkpoint: {ckpt_name}")


def _train_step(ctx, train_loader, opt_ar, scheduler_ar, ar_ema_rate, global_step):
    """One training step. Returns (loss, logits, idx, raw_labels, grad_norms)."""
    config, acc = ctx.config, ctx.accelerator

    idx_raw, label_idx_raw = next(train_loader)
    idx = idx_raw.to(device=acc.device, dtype=torch.long, non_blocking=True)
    label_idx = label_idx_raw.to(device=acc.device, dtype=torch.long, non_blocking=True)

    if ctx.semantic_offset > 0:
        idx = idx.clone()
        idx[:, :ctx.z_len] += ctx.semantic_offset

    if ctx.use_eos:
        idx = insert_eos_token(idx, ctx.z_len, ctx.eos_token_id)

    if config.use_label:
        raw_labels = _labels_from_label_idx(label_idx, num_classes=int(config.num_classes), uncond_idx=ctx.uncond_idx)
    else:
        raw_labels = ctx.uncond_labels.expand(idx.shape[0], -1)
    labels = raw_labels


    if config.label_drop_prob > 0:
        drop = (torch.rand((labels.shape[0], 1), device=acc.device) < config.label_drop_prob)
        labels = torch.where(drop, ctx.uncond_labels.expand_as(raw_labels), raw_labels)

    ar_out = ctx.ar_model(idx, labels)
    logits = ar_out.logits
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1))

    acc.backward(loss)
    grad_norms = calc_grad_norm(
        {"AR": ctx.ar_model},
        global_step, int(getattr(config, "grad_norm_freq", 0)),
        accelerator=acc,
    )
    cur_lr = max(pg['lr'] for pg in opt_ar.param_groups)
    if cur_lr > 0:
        if config.grad_clip > 0:
            acc.clip_grad_norm_(ctx.ar_model.parameters(), max_norm=config.grad_clip)
        opt_ar.step()
    opt_ar.zero_grad(set_to_none=True)
    if scheduler_ar is not None:
        scheduler_ar.step()
    if ctx.ar_model_ema is not None and cur_lr > 0:
        ema_update(ctx.ar_model, ctx.ar_model_ema, ar_ema_rate)

    return loss, logits, idx, raw_labels, grad_norms


def _visualize(ctx, idx, raw_labels, global_step):
    """Decode GT tokens + generate AR samples, save visualization grid."""
    config, acc = ctx.config, ctx.accelerator
    acc.wait_for_everyone()
    if acc.is_main_process:
        vis_n = max(1, min(config.visualize_img_num, int(idx.shape[0])))

        if not config.ema_sampling:
            toggle_train_eval(ctx.ar_model, train=False, accelerator=acc)
            eval_ar = ctx.ar_model
        else:
            eval_ar = ctx.ar_model_ema

        with torch.no_grad():
            vis_labels = raw_labels[:vis_n]
            vis_idx = idx[:vis_n]
            vis_ae_labels = ctx.uncond_labels.expand(vis_n, -1) if ctx.ae_no_label else vis_labels
            gt_visual_idx = vis_idx[:, ctx.z_len + ctx.eos_len:] if ctx.prologue else vis_idx
            gt_quant = _codes_from_indices(ctx.quantizer, gt_visual_idx, vis_ae_labels)
            gt_patches = ctx.dec(gt_quant, vis_ae_labels)
            gt_imgs = img_denormalize(unpatchify(gt_patches, config.image_size, config.patch_size))
            sample_patches = sampling(ctx.quantizer, ctx.dec, eval_ar, bz=vis_n,
                                      class_label=vis_labels, config=config,
                                      ae_label=vis_ae_labels)
            sample_imgs = img_denormalize(unpatchify(sample_patches, config.image_size, config.patch_size))

            grid_imgs = torch.cat([gt_imgs, sample_imgs], dim=0)
            grid = torchvision.utils.make_grid(grid_imgs, nrow=vis_n, normalize=False)
            out_path = os.path.join(ctx.save_dir, "images", f"Step={global_step+1}-ar_vis.png")
            torchvision.utils.save_image(grid, out_path)
            acc.log({"visualization/imgs": wandb.Image(grid), "global_step": global_step + 1}, step=global_step + 1)

        if not config.ema_sampling:
            toggle_train_eval(ctx.ar_model, train=True, accelerator=acc)

    acc.wait_for_everyone()


def _evaluate(ctx, eval_loader, global_step):
    """Run gFID evaluation + optional conditional entropy. Returns (gFID, gFID_nocfg, eval_ar_loss, IS)."""
    config, acc = ctx.config, ctx.accelerator

    sample_cached_path = os.path.join(config.tmp_dir, "sample_images.npz")
    gt_cache_path = config.eval_fid_ref_path
    num_fid_samples = config.num_fid_samples
    num_classes = config.num_classes

    label_indices_local = generate_uniform_labels(
        num_samples=num_fid_samples, num_classes=num_classes,
        accelerator=acc, exclude_uncond=True,
    )
    local_bz = config.eval_batch_size // acc.num_processes
    num_batches = (len(label_indices_local) + local_bz - 1) // local_bz
    samples_buf, nocfg_samples_buf = [], []

    if not config.ema_sampling:
        toggle_train_eval(ctx.ar_model, train=False, accelerator=acc)
        eval_ar = ctx.ar_model
    else:
        eval_ar = ctx.ar_model_ema

    gFID, gFID_nocfg, eval_ar_loss, IS = 0., 0., 0., 0.

    cuda_rng_state = torch.cuda.get_rng_state()
    eval_seed = int(config.seed) + acc.process_index
    torch.cuda.manual_seed(eval_seed)
    print(f"[Eval] Per-rank CUDA seed: base={config.seed}, rank={acc.process_index}, effective={eval_seed}")

    with torch.no_grad():
        acc.wait_for_everyone()

        for i in tqdm(range(num_batches), disable=not acc.is_main_process, dynamic_ncols=True, file=sys.stdout, desc="Evaluating(gFID)"):
            batch_start = i * local_bz
            batch_end = min(batch_start + local_bz, len(label_indices_local))
            batch_label_idx = label_indices_local[batch_start:batch_end]

            bz = len(batch_label_idx)
            lbl_full = _labels_from_label_idx(batch_label_idx, num_classes=num_classes, uncond_idx=ctx.uncond_idx)
            ae_lbl = ctx.uncond_labels.expand(bz, -1) if ctx.ae_no_label else lbl_full

            sample_patches = sampling(ctx.quantizer, ctx.dec, eval_ar, bz=bz,
                                      class_label=lbl_full, config=config,
                                      ae_label=ae_lbl)
            sample_u8 = img_norm_to_uint8(unpatchify(sample_patches, config.image_size, config.patch_size))
            samples_buf.append(sample_u8.permute(0, 2, 3, 1).cpu().numpy())

            if config.nocfg_sample:
                nocfg_config = config.copy()
                nocfg_config.cfg = 0.0
                nocfg_config.semantic_cfg_schedule = None
                nocfg_config.visual_cfg_schedule = None
                nocfg_config.semantic_cfg_scale = None
                nocfg_config.visual_cfg_scale = None
                nocfg_patches = sampling(ctx.quantizer, ctx.dec, eval_ar, bz=bz,
                                         class_label=lbl_full, config=nocfg_config,
                                         ae_label=ae_lbl)
                nocfg_u8 = img_norm_to_uint8(unpatchify(nocfg_patches, config.image_size, config.patch_size))
                nocfg_samples_buf.append(nocfg_u8.permute(0, 2, 3, 1).cpu().numpy())

        all_samples = np.concatenate(samples_buf, axis=0)
        gathered_samples = acc.gather(torch.from_numpy(all_samples).to(acc.device))
        acc.wait_for_everyone()
        if acc.is_main_process:
            gathered_np = gathered_samples.cpu().numpy()
            print0(f"[Eval] Gathered samples: {gathered_np.shape[0]}, num_fid_samples: {num_fid_samples}")
            if gathered_np.shape[0] != num_fid_samples:
                print0(f"WARNING: Gathered samples count ({gathered_np.shape[0]}) != num_fid_samples ({num_fid_samples})")
            np.savez(sample_cached_path, gathered_np)
            _compute_is = getattr(config, 'compute_is', False)
            result = adm_fid_evaluator(sample_cached_path, gt_cache_path, config, acc, compute_is=_compute_is)
            if _compute_is:
                gFID, IS = result
            else:
                gFID = result
        acc.wait_for_everyone()
        gFID = acc.reduce(torch.tensor(gFID, device=acc.device), reduction="sum").item()
        IS = acc.reduce(torch.tensor(IS, device=acc.device), reduction="sum").item()
        print0(f"gFID: {gFID}" + (f", IS: {IS}" if IS > 0 else ""))

        if config.nocfg_sample:
            all_nocfg = np.concatenate(nocfg_samples_buf, axis=0)
            gathered_nocfg = acc.gather(torch.from_numpy(all_nocfg).to(acc.device))
            acc.wait_for_everyone()
            nocfg_cached_path = os.path.join(config.tmp_dir, "nocfg_sample_images.npz")
            if acc.is_main_process:
                gathered_nocfg_np = gathered_nocfg.cpu().numpy()
                print0(f"[Eval] Gathered nocfg_samples: {gathered_nocfg_np.shape[0]}, num_fid_samples: {num_fid_samples}")
                if gathered_nocfg_np.shape[0] != num_fid_samples:
                    print0(f"WARNING: Gathered nocfg_samples count ({gathered_nocfg_np.shape[0]}) != num_fid_samples ({num_fid_samples})")
                np.savez(nocfg_cached_path, gathered_nocfg_np)
                gFID_nocfg = adm_fid_evaluator(nocfg_cached_path, gt_cache_path, config, acc)
            acc.wait_for_everyone()
            gFID_nocfg = acc.reduce(torch.tensor(gFID_nocfg, device=acc.device), reduction="sum").item()
            print0(f"gFID_nocfg: {gFID_nocfg}")

    if not config.ema_sampling:
        toggle_train_eval(ctx.ar_model, train=True, accelerator=acc)

    # Conditional entropy on eval_loader (optional)
    if getattr(config, "do_ar_eval_loader", False):
        acc.wait_for_everyone()
        if acc.is_main_process:
            print0("Evaluating AR model conditional entropy and loss on eval_loader...")

        if not config.ema_sampling:
            toggle_train_eval(ctx.ar_model, train=False, accelerator=acc)
            eval_ar = ctx.ar_model
        else:
            eval_ar = ctx.ar_model_ema

        entropy_acc_dict = {"ent_sum": None, "ent_cnt": None}
        entropy_log_base = float(getattr(config, "prefix_entropy_log_base", 2.0))
        eval_loss_sum = torch.zeros((), device=acc.device, dtype=torch.float32)
        eval_loss_count = torch.zeros((), device=acc.device, dtype=torch.long)

        with torch.no_grad():
            for _, (idx_tokens, label_idx) in enumerate(tqdm(eval_loader, disable=not acc.is_main_process, dynamic_ncols=True, file=sys.stdout, desc="Evaluating(Cond Entropy)")):
                idx = idx_tokens.to(device=acc.device, dtype=torch.long, non_blocking=True)
                label_idx_batch = label_idx.to(device=acc.device, dtype=torch.long, non_blocking=True)

                if ctx.semantic_offset > 0:
                    idx = idx.clone()
                    idx[:, :ctx.z_len] += ctx.semantic_offset

                if ctx.use_eos:
                    idx = insert_eos_token(idx, ctx.z_len, ctx.eos_token_id)

                if config.use_label:
                    labels = _labels_from_label_idx(label_idx_batch, num_classes=int(config.num_classes), uncond_idx=ctx.uncond_idx)
                else:
                    labels = ctx.uncond_labels.expand(idx.shape[0], -1)

                ar_out = eval_ar(idx, labels)
                logits = ar_out.logits
                batch_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1))
                eval_loss_sum += batch_loss * idx.shape[0]
                eval_loss_count += idx.shape[0]

                draw_conditional_entropy(
                    entropy_acc_dict, logits=logits, accelerator=acc,
                    save_dir=ctx.save_dir, global_step=global_step,
                    log_base=entropy_log_base, finalize=False,
                    codebook_size=config.codebook_size,
                )

        acc.wait_for_everyone()
        eval_loss_sum = acc.reduce(eval_loss_sum, reduction='sum')
        eval_loss_count = acc.reduce(eval_loss_count, reduction='sum')
        eval_ar_loss = (eval_loss_sum / eval_loss_count.clamp(min=1)).item()
        print0(f"eval_ar_loss: {eval_ar_loss}")

        draw_conditional_entropy(
            entropy_acc_dict, accelerator=acc,
            save_dir=ctx.save_dir, global_step=global_step,
            log_base=entropy_log_base, finalize=True,
            rFID=0.0, gFID=gFID,
            codebook_size=config.codebook_size,
        )

        if not config.ema_sampling:
            toggle_train_eval(ctx.ar_model, train=True, accelerator=acc)

    acc.wait_for_everyone()

    if acc.is_main_process:
        if not getattr(config, "keep_sample_npz", False):
            safe_remove_file(sample_cached_path)
            if config.nocfg_sample:
                safe_remove_file(os.path.join(config.tmp_dir, "nocfg_sample_images.npz"))
        else:
            print0(f"[Eval] keep_sample_npz=True: retaining {sample_cached_path}"
                   + (f" and {os.path.join(config.tmp_dir, 'nocfg_sample_images.npz')}"
                      if config.nocfg_sample else ""))
    acc.wait_for_everyone()
    torch.cuda.set_rng_state(cuda_rng_state)

    return gFID, gFID_nocfg, eval_ar_loss, IS


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_ar(config):

    if config.get("use_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if "EXPERIMENT_SAVE_DIR" in os.environ:
        save_dir = os.environ["EXPERIMENT_SAVE_DIR"]
    else:
        experiment_index = len(list(Path(str(config.save_dir)).glob("*")))
        save_dir = str(config.save_dir) + f"/{experiment_index:03d}"
        os.environ["EXPERIMENT_SAVE_DIR"] = save_dir

    project_config = ProjectConfiguration(project_dir=save_dir, logging_dir=save_dir)
    accelerator = Accelerator(
        mixed_precision=config.precision if config.precision in ["fp16", "bf16"] else "no",
        log_with="wandb",
        project_config=project_config,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )

    # prepare seed
    if config.seed is not None:
        seed = int(config.seed) if config.seed is not None else 0
    set_seed(seed)
    seed_everything(seed)
    
    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)
    worker_init = make_worker_init_fn(seed)

    print0(config)
    print0(f"Global seed set to {config.seed}")

    prologue = config.get("Prologue", False) and not config.get("share_semantic_codebook", False)
    z_len = int(config.z_len) if prologue else 0
    quantizer = _load_quantizer(config, ckpt_dir=config.tokenizer_ckpt_path).to(accelerator.device)
    dec = _load_decoder(config, ckpt_dir=config.tokenizer_ckpt_path).to(accelerator.device)

    semantic_quantizer = None
    if prologue:
        semantic_quantizer = _load_semantic_quantizer(config, ckpt_dir=config.tokenizer_ckpt_path).to(accelerator.device)
        print0(f"Prologue enabled: z_len={z_len}")

    semantic_offset = int(config["Quantizer"]["codebook_size"]) if z_len > 0 else 0
    print0(f"semantic_offset={semantic_offset}")

    ar_model = ARModel(config)
    _logit_mask = build_ar_logit_mask(
        getattr(quantizer, 'pos_select_mask', None),
        getattr(semantic_quantizer, 'pos_select_mask', None) if prologue else None,
        vis_cb_size=int(config["Quantizer"]["codebook_size"]),
        sem_cb_size=int(config["SemanticQuantizer"]["codebook_size"]) if prologue else 0,
    )
    ar_model.set_logit_mask(_logit_mask)
    ar_ema_rate = math.pow(0.5, 1 / config.ar_ema_halflife) if config.ar_ema_halflife > 0 else 0.0
    ar_model_ema = copy.deepcopy(ar_model).requires_grad_(False).eval() if ar_ema_rate > 0 else None
    if ar_model_ema is not None:
        ar_model_ema.set_logit_mask(_logit_mask)
    print0(f"logit_mask: {'set' if ar_model.logit_mask is not None else 'None'}")

    if config.get("continuous_training", False) and config.tokenizer_ckpt_path:
        tok_ckpt = config.tokenizer_ckpt_path
        ar_weights_path = os.path.join(tok_ckpt, "model_5.safetensors")
        print0(f"[Continuous training] Loading AR weights from tokenizer ckpt: {ar_weights_path}")
        _ar_sd = load_file(ar_weights_path); _ar_sd.pop("logit_mask", None)  # old ckpts
        ar_model.load_state_dict(_ar_sd, strict=True)
        _log_load_verify_strict("AR_model_5", ar_weights_path, len(_ar_sd), ar_model)
        if ar_model_ema is not None:
            ar_ema_path = os.path.join(tok_ckpt, "model_6.safetensors")
            print0(f"[Continuous training] Loading AR EMA from tokenizer ckpt: {ar_ema_path}")
            _ar_ema_sd = load_file(ar_ema_path); _ar_ema_sd.pop("logit_mask", None)
            ar_model_ema.load_state_dict(_ar_ema_sd, strict=True)
            _log_load_verify_strict("AR_ema_model_6", ar_ema_path, len(_ar_ema_sd), ar_model_ema)

    if config.resume_ckpt_path != "" and not config.resume_train:
        print0(f"Loading weights from {config.resume_ckpt_path} before prepare/compile...")
        sd = load_file(config.resume_ckpt_path + "/model.safetensors")
        sd.pop("logit_mask", None)  # old ckpts
        missing, unexpected = ar_model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected key(s) in checkpoint: {unexpected}")
        non_buffer_missing = [k for k in missing if k not in dict(ar_model.named_buffers())]
        if non_buffer_missing:
            raise RuntimeError(f"Missing non-buffer key(s) in checkpoint: {non_buffer_missing}")
        if missing:
            print0(f"  (skipped {len(missing)} buffer(s) not in checkpoint: {missing})")
        if ar_model_ema is not None:
            print0(f"AR EMA weights from {config.resume_ckpt_path} before prepare/compile...")
            sd_ema = load_file(config.resume_ckpt_path + "/model_1.safetensors")
            sd_ema.pop("logit_mask", None)
            missing_ema, unexpected_ema = ar_model_ema.load_state_dict(sd_ema, strict=False)
            if unexpected_ema:
                raise RuntimeError(f"Unexpected key(s) in EMA checkpoint: {unexpected_ema}")
            non_buffer_missing_ema = [k for k in missing_ema if k not in dict(ar_model_ema.named_buffers())]
            if non_buffer_missing_ema:
                raise RuntimeError(f"Missing non-buffer key(s) in EMA checkpoint: {non_buffer_missing_ema}")
            if missing_ema:
                print0(f"  (skipped {len(missing_ema)} EMA buffer(s) not in checkpoint: {missing_ema})")

    if config.ema_sampling and ar_model_ema is None:
        raise ValueError("ema_sampling=True but ar_model_ema is None (set ar_ema_halflife>0)")

    eval_only = bool(getattr(config, 'eval_only', False))
    if eval_only:
        print0("=== EVAL ONLY MODE: skipping optimizer, scheduler, and train data ===")

    if not eval_only:
        no_decay_keywords = ['bias', 'norm', 'adaln']
        if getattr(config, 'embedding_nowd', False):
            no_decay_keywords = no_decay_keywords +  ['emb.','ar_model.out.']
        decay_params, nodecay_params = [], []
        for n, p in ar_model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or any(kw in n for kw in no_decay_keywords):
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        print0(f"Weight decay groups: {len(decay_params)} decay ({sum(p.numel() for p in decay_params):,} params), "
                          f"{len(nodecay_params)} no-decay ({sum(p.numel() for p in nodecay_params):,} params)")
        opt_ar = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay_ar},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=config.lr_ar,
            betas=(config.ar_beta1, config.ar_beta2),
        )
    else:
        opt_ar = None

    total_steps = sum(int(p.split(":")[0]) for p in str(config.phases).split())
    if not eval_only:
        if config.lr_scheduler == 'linear':
            scheduler_ar = get_linear_schedule_with_warmup_peak(
                opt_ar,
                num_warmup_steps=config.warmup_steps,
                num_peak_steps=config.peak_steps,
                num_training_steps=total_steps,
                base_lr=config.lr_ar,
                end_lr=config.lr_ar_min,
            )
        else:
            scheduler_ar = None
    else:
        scheduler_ar = None

    # Pretoken (load into memory) + DataLoaders (train/eval)
    pretoken_dir = str(getattr(config, "pretoken_dir", "") or "")
    if pretoken_dir == "":
        pretoken_dir = _default_pretoken_dir(config.tokenizer_ckpt_path)
        print0(f"Auto-inferred pretoken_dir: {pretoken_dir}")
    
    def get_pretoken_filename(base_name: str, crop_type: str) -> str:
        if crop_type == 'ten_crop':
            suffix = '_tencrop'
        elif crop_type == 'center':
            suffix = '_centercrop'
        elif crop_type == 'random':
            suffix = '_randomcrop'
        elif crop_type is None or crop_type == 'None':
            suffix = ''
        else:
            suffix = f'_{crop_type}'
        name_without_ext = base_name.replace('.npz', '')
        return f"{name_without_ext}{suffix}.npz"
    
    prologue = config.get("Prologue", False) and not config.get("share_semantic_codebook", False)
    token_len = int(config.z_len) + int(config.x_len) if prologue else int(config.z_len)
    crop_type = config.get('crop_type', None)
    eval_crop_type = config.get('eval_crop_type', None)
    
    train_filename = get_pretoken_filename("train_pretoken.npz", crop_type)
    eval_filename = get_pretoken_filename("eval_pretoken.npz", eval_crop_type)
    
    if not eval_only:
        print0(f"Loading train tokens from: {train_filename}")
        train_tokens = torch.from_numpy(np.load(os.path.join(pretoken_dir, train_filename))["data"])
        train_dataset = PretokenDataset(train_tokens, token_len, crop_type=crop_type)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size // accelerator.num_processes,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            worker_init_fn=worker_init,
            generator=dl_generator,
        )
    else:
        train_loader = None
    
    # Create eval dataloader for conditional entropy calculation
    if getattr(config, "do_ar_eval_loader", False):
        print0(f"Loading eval tokens from: {eval_filename}")
        eval_tokens = torch.from_numpy(np.load(os.path.join(pretoken_dir, eval_filename))["data"])
        eval_dataset = PretokenDataset(eval_tokens, token_len, crop_type=eval_crop_type)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size // accelerator.num_processes,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
            worker_init_fn=worker_init,
            generator=dl_generator,
        )
    else:
        eval_loader = None

    # Prepare models, optimizers and dataloaders
    objs = [ar_model]
    if opt_ar is not None:
        objs.append(opt_ar)
    if ar_model_ema is not None:
        objs.append(ar_model_ema)
    if train_loader is not None:
        objs.append(train_loader)
    if eval_loader is not None:
        objs.append(eval_loader)
    if scheduler_ar is not None:
        accelerator.register_for_checkpointing(scheduler_ar)
    prepared = accelerator.prepare(*objs)
    idx_ptr = 0
    ar_model = prepared[idx_ptr]; idx_ptr += 1
    if opt_ar is not None:
        opt_ar = prepared[idx_ptr]; idx_ptr += 1
    if ar_model_ema is not None:
        ar_model_ema = prepared[idx_ptr]; idx_ptr += 1
    if train_loader is not None:
        train_loader = prepared[idx_ptr]; idx_ptr += 1
    if eval_loader is not None:
        eval_loader = prepared[idx_ptr]; idx_ptr += 1

    global_step = 0
    if not eval_only:
        pbar = tqdm(total=total_steps, disable=not accelerator.is_main_process, dynamic_ncols=True, file=sys.stdout, desc="Training AR")
        pbar.set_description(f"Total Steps {total_steps}")

    extra_training_states = {"global_step": 0, "total_yielded": 0, "best_gfid": float("inf"), "best_gfid_nocfg": float("inf")}
    if config.resume_ckpt_path != "" and config.resume_train and not eval_only:
        print0(f"Resuming training from {config.resume_ckpt_path}")
        accelerator.load_state(config.resume_ckpt_path)
        _extra_path = os.path.join(config.resume_ckpt_path, "extra_state.pt")
        if os.path.exists(_extra_path):
            _saved = torch.load(_extra_path, weights_only=False)
            if "dl_generator" in _saved:
                dl_generator.set_state(_saved["dl_generator"])
            extra_training_states.update({k: _saved[k] for k in extra_training_states if k in _saved})
            print0(f"Restored extra state: {extra_training_states}")
        global_step = extra_training_states["global_step"]
        if global_step == 0:
            global_step = int(config.resume_ckpt_path.split("Step=")[-1].split('-')[0])
        pbar.update(global_step)

    accelerator.wait_for_everyone()
    _is_resume = (config.resume_ckpt_path != "" and config.resume_train)
    if accelerator.is_main_process:
        if not _is_resume:
            print0("Calculating model stats...")
            print_model_stats(config, accelerator.device, None, dec, accelerator.unwrap_model(ar_model))
        root_dir = Path(save_dir)
        root_dir.mkdir(exist_ok=True, parents=True)
        img_dir = Path(save_dir + "/images")
        img_dir.mkdir(exist_ok=True, parents=True)
        ckpt_dir = Path(save_dir + "/ckpts")
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        tmp_dir = Path(config.tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
    accelerator.wait_for_everyone()
    if config.torch_compile:
        ar_model = torch.compile(ar_model)
        dec = torch.compile(dec)
        quantizer = torch.compile(quantizer)
        if ar_model_ema is not None:
            ar_model_ema = torch.compile(ar_model_ema)

    if not eval_only:
        # Fast-forward DataLoader to resume position
        _resume_total_yielded = extra_training_states["total_yielded"]
        if _resume_total_yielded > 0:
            _bpe = len(train_loader)
            if _bpe > 0:
                _epoch = _resume_total_yielded // _bpe
                _skip = _resume_total_yielded % _bpe
                if hasattr(train_loader, 'iteration'):
                    train_loader.iteration = _epoch
                print0(f"DataLoader fast-forward: epoch={_epoch}, skip={_skip}/{_bpe}")

        # Infinite iterator wrapper
        train_loader = InfiniteIterator(train_loader, dl_generator=dl_generator)

        if _resume_total_yielded > 0 and _bpe > 0 and _skip > 0:
            for _ in range(_skip):
                next(train_loader)
        train_loader.total_yielded = _resume_total_yielded

    wandb_run_dir = os.path.join(str(config.wandb_dir), str(config.wandb_name))
    os.makedirs(wandb_run_dir, exist_ok=True)

    accelerator.init_trackers(
        project_name=config.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
        init_kwargs={"wandb": {"name": config.wandb_name, "dir": wandb_run_dir}},
    )
    
    uncond_idx = int(config.num_classes) - 1
    uncond_labels = F.one_hot(
        torch.full((1,), uncond_idx, device=accelerator.device, dtype=torch.long),
        num_classes=int(config.num_classes),
    ).float()
    _ae_no_label = bool(getattr(config, 'ae_no_label', False))

    _ar_raw = _unwrap(ar_model)
    _use_eos = bool(_ar_raw.eos_len > 0)
    ctx = SimpleNamespace(
        ar_model=ar_model, ar_model_ema=ar_model_ema,
        quantizer=quantizer, dec=dec,
        config=config, accelerator=accelerator,
        uncond_labels=uncond_labels, uncond_idx=uncond_idx,
        ae_no_label=_ae_no_label,
        semantic_offset=semantic_offset, z_len=z_len, prologue=prologue,
        save_dir=save_dir,
        use_eos=_use_eos,
        eos_token_id=int(_ar_raw.eos_token_id) if _use_eos else -1,
        eos_len=int(_ar_raw.eos_len),
    )

    best_gfid = extra_training_states["best_gfid"]
    best_gfid_nocfg = extra_training_states["best_gfid_nocfg"]

    exp_total_steps = int(getattr(config, 'exp_total_phase_steps', 0)) or total_steps
    if eval_only:
        exp_total_steps = 1
        config.eval_freq = 1

    while global_step < exp_total_steps:
        # --- Training step ---
        if not eval_only:
            loss, logits, idx, raw_labels, grad_norms = _train_step(ctx, train_loader, opt_ar, scheduler_ar, ar_ema_rate, global_step)

            if global_step == 0 or ((global_step + 1) % config.visualize_freq == 0):
                _visualize(ctx, idx, raw_labels, global_step)

        # --- Evaluation ---
        gFID, gFID_nocfg, eval_ar_loss, IS = 0., 0., 0., 0.

        if (global_step + 1) % config.eval_freq == 0:
            if train_loader is not None:
                extra_training_states["global_step"] = global_step + 1
                extra_training_states["dl_generator"] = train_loader._pre_epoch_gen_state
                extra_training_states["total_yielded"] = train_loader.total_yielded

            gFID, gFID_nocfg, eval_ar_loss, IS = _evaluate(ctx, eval_loader, global_step)

            if config.save_best and config.save_ckpt:
                if gFID < best_gfid:
                    best_gfid = float(gFID)
                    _save_checkpoint(ctx, train_loader, extra_training_states, global_step + 1,
                        f'best-Step={global_step+1}-gFID={best_gfid:.4f}',
                        remove_old_metric="gFID", best_gfid=best_gfid)
                if config.nocfg_sample and gFID_nocfg > 0. and gFID_nocfg < best_gfid_nocfg:
                    best_gfid_nocfg = float(gFID_nocfg)
                    _save_checkpoint(ctx, train_loader, extra_training_states, global_step + 1,
                        f'best-Step={global_step+1}-gFIDwoCFG={best_gfid_nocfg:.4f}',
                        remove_old_metric="gFIDwoCFG", best_gfid_nocfg=best_gfid_nocfg)

        # --- Eval-only logging ---
        if eval_only and gFID > 0.:
            eval_metrics = {
                "eval/gFID": gFID,
                "eval/gFID_nocfg": gFID_nocfg,
                "eval/ar_loss": eval_ar_loss,
                "eval/IS": IS,
                "global_step": global_step + 1,
            }
            eval_metrics_logger = {k: v for k, v in eval_metrics.items() if v != 0.}
            accelerator.log(eval_metrics_logger, step=global_step + 1)

        # --- Logging ---
        if not eval_only and (global_step + 1) % config.log_freq == 0:
            with torch.no_grad():
                acc_rate = (logits.argmax(dim=-1) == idx).float().mean()
            loss_val = accelerator.reduce(loss.detach(), reduction="mean").item()
            acc_val = accelerator.reduce(acc_rate, reduction="mean").item()
            lr_val = float(opt_ar.param_groups[0]["lr"])

            metrics = {
                "prior_loss/ar_loss": loss_val,
                "prior_loss/correct_token_rate": acc_val,
                "lr/lr_ar": lr_val,
                "eval/gFID": gFID,
                "eval/gFID_nocfg": gFID_nocfg,
                "eval/ar_loss": eval_ar_loss,
                "eval/IS": IS,
                "global_step": global_step + 1,
            }
            metrics_logger = {k: v for k, v in metrics.items() if v != 0.}
            metrics_logger.update(grad_norms)
            metrics_4f = {k: f"{v:.4f}" if k not in ["global_step"] else int(v) for k, v in metrics.items()}

            accelerator.log(metrics_logger, step=global_step + 1)
            if accelerator.is_main_process:
                pbar.set_postfix(metrics_4f, refresh=False)
                pbar.update(config.log_freq)

        # --- Periodic checkpoint ---
        if not eval_only and config.save_ckpt and ((global_step + 1) % config.ckpt_freq == 0):
            _save_checkpoint(ctx, train_loader, extra_training_states, global_step + 1,
                f'Step={global_step+1}-gFID={gFID:.4f}')

        global_step += 1

    # --- Cleanup ---
    if not eval_only and config.save_ckpt:
        _save_checkpoint(ctx, train_loader, extra_training_states, global_step,
            f'last-Step={global_step}')
    accelerator.wait_for_everyone()
    if not eval_only:
        pbar.close()

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            tracker.finish()
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    from utils import load_config
    train_ar(load_config())
