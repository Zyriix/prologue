import sys
import os
from safetensors.torch import load_file, save_file
import argparse
import itertools
import math
import random
import subprocess
import zipfile
import io
from pathlib import Path
from typing import Iterable, Iterator
try:
    import wandb
except ImportError:                                              # inference-only envs
    wandb = None
import numpy as np
import numpy.lib.format as np_format
import torch
import torch.nn.functional as F
import torchvision.utils
import yaml
from einops import rearrange
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from accelerate.utils import set_seed, ProjectConfiguration
from dataset import ImageFolderDataset
from model_gan import GANLoss
from model_lpips import LPIPS, BothPerceptualLoss, build_perceptual_loss
from models import ARModel, AROutput, Encoder, Decoder, Linear, EncoderOutput, VQLossDetail, insert_eos_token
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import copy
import torch.distributed as dist
from glob import glob
from utils import (
    seed_everything,
    build_ar_logit_mask,
    make_worker_init_fn,
    load_accelerate_weights_only,
    draw_conditional_entropy,
    draw_data_conditional_entropy,
    plot_codebook_usage,
    plot_posterior_entropy,
    compute_posterior_entropy_from_logits,
    compute_aggregated_entropy_from_counts,
    InfiniteIterator,
    get_linear_schedule_with_warmup_peak,
    safe_remove_file,
    save_training_state,
    remove_old_best_checkpoints,
    adm_fid_evaluator,
    ema_update,
    img_denormalize,
    img_norm_to_uint8,
    img_uint8_to_norm,
    patchify,
    unpatchify,
    toggle_require_grad,
    toggle_train_eval,
    zero_nan_gradients,
    calc_grad_norm,
    print0,
    _unwrap,
    Target,
    Phase,
    parse_phases,
    parse_training_config_from_phases,
    get_phase,
)
from util_model_profile import print_model_stats
import gc
from torchmetrics.functional.image import (
    peak_signal_noise_ratio as calc_psnr, 
     structural_similarity_index_measure as calc_ssim)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch._dynamo.config.recompile_limit = 64

@torch.no_grad()
@torch._dynamo.disable
def sampling(enc, dec, ar_model, bz, class_label=None, temperature=1.0, topK=None, topP=None, cfg=1.0, cfg_schedule=None, cfg_power=None, cache_kv=False, ae_label=None,
             semantic_cfg_schedule=None, semantic_cfg_scale=None, semantic_cfg_power=None, semantic_cfg_start=0.0,
             visual_cfg_schedule=None, visual_cfg_scale=None, visual_cfg_power=None, visual_cfg_start=1.0):
    enc_raw = _unwrap(enc)
    ar_raw = _unwrap(ar_model)
    token_ids = ar_raw.sampling(bz, class_label, temperature, topK, topP, cfg, cfg_schedule, cfg_power, cache_kv,
                                semantic_cfg_schedule=semantic_cfg_schedule,
                                semantic_cfg_scale=semantic_cfg_scale,
                                semantic_cfg_power=semantic_cfg_power,
                                semantic_cfg_start=semantic_cfg_start,
                                visual_cfg_schedule=visual_cfg_schedule,
                                visual_cfg_scale=visual_cfg_scale,
                                visual_cfg_power=visual_cfg_power,
                                visual_cfg_start=visual_cfg_start)
    quant = enc_raw.get_visual_codes(token_ids, ae_label)
    x_hat = dec(quant, ae_label)
    return x_hat

@torch.no_grad()
@torch._dynamo.disable
def reconstruction(enc, dec, x, labels):
    out = enc(x, labels, training=False)
    x_hat = dec(out.visual_quant, labels)
    return x_hat, out.indices

def calc_per_sample_reconstruction_metrics(img, img_hat, lpips):
    img = img.to(dtype=torch.float32)
    img_hat = img_hat.to(dtype=torch.float32)
    img01 = ((img + 1.0) * 0.5).clamp(0.0, 1.0)
    img_hat01 = ((img_hat + 1.0) * 0.5).clamp(0.0, 1.0)
    psnr_vals = calc_psnr(img_hat01, img01, data_range=1.0).to(dtype=torch.float32)
    ssim_vals = calc_ssim(img_hat01, img01, data_range=1.0,).to(dtype=torch.float32)
    out = lpips(img, img_hat)
    lpips_vals = out[0].mean() if isinstance(out, tuple) else out.mean()
    return lpips_vals, ssim_vals, psnr_vals

def calculate_adaptive_weight_acc(nll_loss, g_loss, last_layer, accelerator, dec):
    # Avoid gradient all-reduce on decoder params during these probe grads.
    with accelerator.no_sync(dec):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight

def make_cache_kwargs(cached_enc_out, cached_imgs_hat):
    detach_cpu = lambda t: t.detach().cpu() if isinstance(t, torch.Tensor) else None
    if cached_enc_out is not None:
        return dict(
            cached_quant=detach_cpu(cached_enc_out.quant),
            cached_idx=detach_cpu(cached_enc_out.indices),
            cached_visual_quant=detach_cpu(cached_enc_out.visual_quant),
            cached_semantic_quant=detach_cpu(cached_enc_out.semantic_quant),
            cached_visual_indices=detach_cpu(cached_enc_out.visual_indices),
            cached_semantic_indices=detach_cpu(cached_enc_out.semantic_indices),
            cached_one_hot=detach_cpu(cached_enc_out.one_hot),
            cached_imgs_hat=detach_cpu(cached_imgs_hat),
        )
    return dict(cached_quant=None, cached_idx=None, cached_imgs_hat=None)

def train(config):
    if config.get("use_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if "EXPERIMENT_SAVE_DIR" in os.environ:
        save_dir = os.environ["EXPERIMENT_SAVE_DIR"]
    else:
        experiment_index = len(glob(f"{config.save_dir}/*"))
        save_dir = config.save_dir+f"/{experiment_index:03d}"
        os.environ["EXPERIMENT_SAVE_DIR"] = save_dir
    
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(config.get("find_unused_parameters", False))
    )
    accelerator = Accelerator(
        mixed_precision=config.precision if config.precision in ["fp16", "bf16"] else "no",
        log_with="wandb",
        project_config=ProjectConfiguration(project_dir=save_dir, logging_dir=save_dir),
        dataloader_config=DataLoaderConfiguration(even_batches=False),
        kwargs_handlers=[ddp_kwargs],
    )

    phases = parse_phases(config.phases)
    train_ae, train_ar, use_lpips_loss, use_gan_loss, train_prior_enc = parse_training_config_from_phases(phases)
    total_phase_steps = sum(phase.num_steps for phase in phases)
    total_phase = len(phases)
    phase_step_accum = list(itertools.accumulate([phase.num_steps for phase in phases]))
    print0("Training Phases: ", phases)


    if config.torch_compile and use_gan_loss and config.disc_adaptive_weight:
        try:
            import torch._functorch.config as _functorch_config  
            _functorch_config.donated_buffer = False
            if accelerator.is_main_process:
                print0("[warn] disabled torch._functorch.config.donated_buffer due to disc_adaptive_weight + torch_compile")
        except Exception:
            pass
    
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

    # Models
    enc = Encoder(config)
    dec = Decoder(config)
    config["train_prior_enc"] = train_prior_enc
    ar_model = ARModel(config) if train_ar else None

    # Cache encoder properties before DDP / torch.compile wrapping
    _has_separate_semantic = enc.has_separate_semantic
    _visual_modules = enc.visual_modules
    _semantic_modules = enc.semantic_modules
    _prologue = config.get("Prologue", False) and not config.get("share_semantic_codebook", False)
    _ste_ar_embedding = config.get("ARModel", {}).get("ste_ar_embedding", False)
    _semantic_offset = int(config["Quantizer"]["codebook_size"]) if _prologue else 0
    _use_eos = bool(config.get("use_eos", False)) and _prologue and int(config.get("z_len", 0)) > 0
    _eos_offset = 1 if _use_eos else 0
    _ae_no_label = bool(getattr(config, 'ae_no_label', False))
    _prior_visual_dropout = float(config.get("prior_visual_dropout", 0.0))
    if _prior_visual_dropout > 0 and not _use_eos:
        print0("WARNING: prior_visual_dropout requires use_eos=True and Prologue; forcing to 0.")
        _prior_visual_dropout = 0.0
    _label_drop_always = float(getattr(config, "label_drop_prob", 0.0)) >= 1.0  # uncond viz/eval when always-drop

    if not train_ae:
        toggle_require_grad(enc, False, sub_modules=_visual_modules if _has_separate_semantic else None)
        toggle_train_eval(enc, train=False, sub_modules=_visual_modules if _has_separate_semantic else None)
        toggle_require_grad(dec, False)
        toggle_train_eval(dec, train=False)

    gan_loss = GANLoss(**config.GANLoss) if use_gan_loss else None
    perceptual_loss = build_perceptual_loss(config.get("perceptual_network", "vgg")).to(accelerator.device).eval().requires_grad_(False) if use_lpips_loss else None

    # AR logit mask: visual/semantic segments + optional EOS row.
    if ar_model is not None:
        _logit_mask = build_ar_logit_mask(
            getattr(enc.quantizer, 'pos_select_mask', None),
            getattr(getattr(enc, 'semantic_quantizer', None), 'pos_select_mask', None),
            vis_cb_size=int(config["Quantizer"]["codebook_size"]),
            sem_cb_size=int(config["SemanticQuantizer"]["codebook_size"]) if _prologue else 0,
        )
        ar_model.set_logit_mask(_logit_mask)
        print0(f"logit_mask: {'set' if ar_model.logit_mask is not None else 'None'}")

    # EMA models
    ae_ema_rate = math.pow(0.5,1/config.ae_ema_halflife) if config.ae_ema_halflife>0 else 0.
    ar_ema_rate = math.pow(0.5,1/config.ar_ema_halflife) if config.ar_ema_halflife>0 else 0.
    enc_ema = copy.deepcopy(enc).requires_grad_(False).eval() if ae_ema_rate>0 else None
    dec_ema = copy.deepcopy(dec).requires_grad_(False).eval() if ae_ema_rate>0 else None
    ar_model_ema = copy.deepcopy(ar_model).requires_grad_(False).eval() if train_ar and ar_ema_rate>0 else None
    fixed_ar_model = copy.deepcopy(ar_model).requires_grad_(False).eval() if train_prior_enc else None

    if config.resume_ckpt_path != "" and not config.resume_train:
        ckpt_path = config.resume_ckpt_path
        if config.resume_enc:
            sd = load_file(os.path.join(ckpt_path, "model.safetensors"))
            load_visual_only = _has_separate_semantic and not any(k.startswith("semantic_enc.") for k in sd)
            if load_visual_only:
                use_ema_as_visual = not train_ae
                visual_sd = load_file(os.path.join(ckpt_path, "model_2.safetensors")) if use_ema_as_visual else sd
                enc.enc.load_state_dict({k.removeprefix("enc."): v for k, v in visual_sd.items() if k.startswith("enc.")}, strict=True)
                enc.quantizer.load_state_dict({k.removeprefix("quantizer."): v for k, v in visual_sd.items() if k.startswith("quantizer.")}, strict=True)
            else:
                enc.load_state_dict(sd, strict=True)
            if enc_ema is not None:
                ema_sd = load_file(os.path.join(ckpt_path, "model_2.safetensors")) if not load_visual_only or not use_ema_as_visual else visual_sd
                if load_visual_only:
                    enc_ema.enc.load_state_dict({k.removeprefix("enc."): v for k, v in ema_sd.items() if k.startswith("enc.")}, strict=True)
                    enc_ema.quantizer.load_state_dict({k.removeprefix("quantizer."): v for k, v in ema_sd.items() if k.startswith("quantizer.")}, strict=True)
                else:
                    enc_ema.load_state_dict(ema_sd, strict=True)
            print0(f"Loaded encoder from {ckpt_path}" + (" (visual only, ema)" if load_visual_only and use_ema_as_visual else " (visual only)" if load_visual_only else ""))
        if config.resume_dec:
            use_ema_as_dec = not train_ae
            dec_file = "model_3.safetensors" if use_ema_as_dec else "model_1.safetensors"
            dec.load_state_dict(load_file(os.path.join(ckpt_path, dec_file)), strict=True)
            if dec_ema is not None:
                dec_ema.load_state_dict(load_file(os.path.join(ckpt_path, "model_3.safetensors")), strict=True)
            print0(f"Loaded decoder from {ckpt_path}" + (f" (ema as init)" if use_ema_as_dec else ""))
        if config.resume_ar:
            ar_model.load_state_dict(load_file(os.path.join(ckpt_path, "model_5.safetensors")), strict=True)
            if ar_model_ema is not None:
                ar_model_ema.load_state_dict(load_file(os.path.join(ckpt_path, "model_6.safetensors")), strict=True)
            if train_prior_enc:
                fixed_ar_model.load_state_dict(load_file(os.path.join(ckpt_path, "model_7.safetensors")), strict=True)
            print0(f"Loaded AR model from {ckpt_path}" + (f" (+ema)" if ar_model_ema is not None else "") + (f" (+fixed)" if train_prior_enc else ""))
        if config.resume_gan:
            gan_loss.load_state_dict(load_file(os.path.join(ckpt_path, "model_4.safetensors")), strict=True)
            print0(f"Loaded GAN loss from {ckpt_path}")


    opt_enc = torch.optim.AdamW( enc.semantic_parameters() if (_has_separate_semantic and not train_ae) else  enc.parameters(), lr=config.lr_enc, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay_enc) if (train_ae or (_has_separate_semantic and train_ar)) else None
    if train_ar:
        no_decay_keywords = ['bias', 'norm', 'adaln']
        decay_params, nodecay_params = [], []
        for n, p in ar_model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or any(kw in n for kw in no_decay_keywords):
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        print0(f"AR weight decay groups: {len(decay_params)} decay ({sum(p.numel() for p in decay_params):,} params), "
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
    opt_dec = torch.optim.AdamW(dec.parameters(), lr=config.lr_dec, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay_dec) if train_ae else None
    opt_gan_loss = torch.optim.AdamW(gan_loss.parameters(), lr=config.lr_gan_loss, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay_gan) if use_gan_loss else None

    _gc_fallback = float(getattr(config, 'grad_clip', 0.0))
    grad_clip_enc = float(getattr(config, 'grad_clip_enc', _gc_fallback))
    grad_clip_dec = float(getattr(config, 'grad_clip_dec', _gc_fallback))
    grad_clip_ar = float(getattr(config, 'grad_clip_ar', _gc_fallback))
    grad_clip_gan = float(getattr(config, 'grad_clip_gan', _gc_fallback))
    print0(f"Grad clip: enc={grad_clip_enc}, dec={grad_clip_dec}, ar={grad_clip_ar}, gan={grad_clip_gan}")

    dataset = ImageFolderDataset(path=config['data_dir'],
        resolution=config.image_size,
        use_label=config.use_label,
        max_size=None,
        xflip=config.xflip,
        crop_type=config.crop_type,
        deterministic_crop=bool(getattr(config, "deterministic_crop", False)),
        crop_seed=int(config.seed) if config.seed is not None else 0,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size // accelerator.num_processes,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        worker_init_fn=worker_init,
        generator=dl_generator,
    )
    eval_loader=None
    if config.eval_data_dir is not None:
        eval_dataset = ImageFolderDataset(path=config['eval_data_dir'],
            resolution=config.image_size,
            use_label=config.use_label,
            max_size=None,
            xflip=False,
            crop_type=config.eval_crop_type,
            deterministic_crop=bool(getattr(config, "deterministic_crop", False)),
            crop_seed=int(config.seed) if config.seed is not None else 0,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size // accelerator.num_processes,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init,
            generator=dl_generator,
        )
        eval_dataset_size = len(eval_dataset)

    _is_resume = (config.resume_ckpt_path != "" and config.resume_train)
    if accelerator.is_main_process and not _is_resume:
        print0("Calculating model stats...")
        print_model_stats(config, accelerator.device, enc, dec, ar_model)

    # Prepare with Accelerator
    all_training_components = [enc, dec, enc_ema, dec_ema, gan_loss, ar_model, ar_model_ema, fixed_ar_model,
                            opt_enc, opt_dec, opt_ar, opt_gan_loss,
                            train_loader, eval_loader]
    prepared_components = iter(accelerator.prepare(*filter(lambda x: x is not None, all_training_components)))
    enc = next(prepared_components) if enc is not None else None
    dec = next(prepared_components) if dec is not None else None
    enc_ema = next(prepared_components) if enc_ema is not None else None
    dec_ema = next(prepared_components) if dec_ema is not None else None
    gan_loss = next(prepared_components) if gan_loss is not None else None
    ar_model = next(prepared_components) if ar_model is not None else None
    ar_model_ema = next(prepared_components) if ar_model_ema is not None else None
    fixed_ar_model = next(prepared_components) if fixed_ar_model is not None else None
    opt_enc = next(prepared_components) if opt_enc is not None else None
    opt_dec = next(prepared_components) if opt_dec is not None else None
    opt_ar = next(prepared_components) if opt_ar is not None else None
    opt_gan_loss = next(prepared_components) if opt_gan_loss is not None else None
    train_loader = next(prepared_components) if train_loader is not None else None
    eval_loader = next(prepared_components) if eval_loader is not None else None

    global_step = 0
    pbar = tqdm(total=total_phase_steps, disable=not accelerator.is_main_process, dynamic_ncols=True, file=sys.stdout)
    pbar.set_description(f"Total Steps {total_phase_steps}")

    # LR schedulers (must be created and registered BEFORE load_state for resume compatibility)
    if config.lr_scheduler == 'linear':
        scheduler_enc = get_linear_schedule_with_warmup_peak(opt_enc, num_warmup_steps=config.warmup_steps, num_peak_steps=config.peak_steps, num_training_steps=total_phase_steps/2,base_lr=config.lr_enc,end_lr=config.lr_enc_min) if opt_enc is not None else None
        scheduler_dec = get_linear_schedule_with_warmup_peak(opt_dec, num_warmup_steps=config.warmup_steps, num_peak_steps=config.peak_steps, num_training_steps=total_phase_steps/2,base_lr=config.lr_dec,end_lr=config.lr_dec_min) if opt_dec is not None else None
        scheduler_ar = get_linear_schedule_with_warmup_peak(opt_ar, num_warmup_steps=config.warmup_steps, num_peak_steps=config.peak_steps, num_training_steps=total_phase_steps/2,base_lr=config.lr_ar,end_lr=config.lr_ar_min) if train_ar else None
        scheduler_gan_loss = get_linear_schedule_with_warmup_peak(opt_gan_loss, num_warmup_steps=config.warmup_steps, num_peak_steps=config.peak_steps, num_training_steps=total_phase_steps/2,base_lr=config.lr_gan_loss,end_lr=config.lr_gan_loss_min) if use_gan_loss else None
    else:
        scheduler_enc = None
        scheduler_dec = None
        scheduler_ar = None
        scheduler_gan_loss = None

    for sched in (scheduler_enc, scheduler_dec, scheduler_ar, scheduler_gan_loss):
        if sched is not None:
            accelerator.register_for_checkpointing(sched)

    extra_training_states = {
        "global_step": 0, "total_yielded": 0,
        "best_rfid": float("inf"), "best_gfid": float("inf"),
        "prev_phase_idx": -1, "prev_inner_idx": -1,
        "data_buffer": [], "cached_quant": None, "cached_idx": None, "cached_imgs_hat": None,
        "cached_visual_quant": None, "cached_semantic_quant": None,
        "cached_visual_indices": None, "cached_semantic_indices": None,
        "cached_one_hot": None,
    }
    if config.resume_ckpt_path != "" and config.resume_train:
        print0(f"Resuming training from {config.resume_ckpt_path}")
        accelerator.load_state(config.resume_ckpt_path)
        _extra_path = os.path.join(config.resume_ckpt_path, "extra_state.pt")
        if os.path.exists(_extra_path):
            _saved = torch.load(_extra_path, weights_only=False)
            if "dl_generator" in _saved:
                dl_generator.set_state(_saved["dl_generator"])
            extra_training_states.update({k: _saved[k] for k in extra_training_states if k in _saved})
            print0(f"Restored extra state: step={extra_training_states['global_step']}, "
                   f"yielded={extra_training_states['total_yielded']}, "
                   f"prev_phase={extra_training_states['prev_phase_idx']}, "
                   f"prev_inner={extra_training_states['prev_inner_idx']}, "
                   f"buf_len={len(extra_training_states['data_buffer'])}, "
                   f"has_cache={extra_training_states.get('cached_quant') is not None}")
        global_step = extra_training_states["global_step"]
        if global_step == 0:
            global_step = int(config.resume_ckpt_path.split("Step=")[-1].split('-')[0])
        pbar.update(global_step)
        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        root_dir = Path(save_dir)
        root_dir.mkdir(exist_ok=True, parents=True)
        img_dir  = Path(save_dir+ "/images")
        img_dir.mkdir(exist_ok=True, parents=True)
        ckpt_dir  = Path(save_dir+ "/ckpts")
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        tmp_dir = Path(config.tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
    accelerator.wait_for_everyone()

    # Compilation
    if config.torch_compile:
        if enc is not None: enc = torch.compile(enc)
        if enc_ema is not None: enc_ema = torch.compile(enc_ema)
        if dec is not None: dec = torch.compile(dec)
        if dec_ema is not None: dec_ema = torch.compile(dec_ema)
        if ar_model is not None: ar_model = torch.compile(ar_model)
        if ar_model_ema is not None: ar_model_ema = torch.compile(ar_model_ema)

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
        init_kwargs={"wandb": {"name": config.wandb_name, "dir": wandb_run_dir}}
    )
    
    best_rfid = extra_training_states["best_rfid"]
    best_gfid = extra_training_states["best_gfid"]
    rFID = 0.0
    gFID = 0.0
    prev_phase_idx = extra_training_states["prev_phase_idx"]
    prev_inner_idx = extra_training_states["prev_inner_idx"]
    buf = extra_training_states["data_buffer"]
    data_buffer = [tuple(t.to(accelerator.device) for t in item) for item in buf] if buf else []
    dev = accelerator.device
    cq, ci, cvq, csq, cvi, csi, coh = [
        t.to(dev) if isinstance(t, torch.Tensor) else None
        for t in (extra_training_states.get(k) for k in
        ("cached_quant", "cached_idx", "cached_visual_quant",
         "cached_semantic_quant", "cached_visual_indices", "cached_semantic_indices", "cached_one_hot"))
    ]
    if cq is not None:
        cached_enc_out = EncoderOutput(
            quant=cq, indices=ci, one_hot=coh if coh is not None else torch.zeros_like(ci),
            semantic_vq_loss=VQLossDetail.zero(dev),
            visual_vq_loss=VQLossDetail.zero(dev),
            semantic_quant=csq if csq is not None else cq,
            visual_quant=cvq if cvq is not None else cq,
            semantic_indices=csi if csi is not None else ci,
            visual_indices=cvi if cvi is not None else ci,
        )
    else:
        cached_enc_out = None

    cached_imgs_hat = extra_training_states.get("cached_imgs_hat")
    cached_imgs_hat = cached_imgs_hat.to(dev) if isinstance(cached_imgs_hat, torch.Tensor) else None
    
    uncond_labels = F.one_hot(torch.full((1,), config.num_classes - 1, device=accelerator.device, dtype=torch.long), num_classes=config.num_classes).float()

    exp_total_phase_steps = int(getattr(config, 'exp_total_phase_steps', 0)) or total_phase_steps
    while global_step < exp_total_phase_steps:
        phase_idx, inner_idx, target, internel_step = get_phase(global_step, phases, phase_step_accum, config.gan_start)
        enc_grad = target.DO_AE or target.DO_PRIOR_ENC
        dec_grad = target.DO_AE
        prior_grad = target.DO_PRIOR_AR
        disc_grad = target.DO_GAN_D
        semantic_grad = target.DO_PRIOR_ENC if _has_separate_semantic else False

        if phase_idx != prev_phase_idx or inner_idx != prev_inner_idx:
            if _has_separate_semantic:
                toggle_require_grad(enc, target.DO_AE, accelerator=accelerator, sub_modules=_visual_modules)
                toggle_require_grad(enc, semantic_grad, accelerator=accelerator, sub_modules=_semantic_modules)
            else:
                toggle_require_grad(enc, enc_grad, accelerator=accelerator)
            toggle_require_grad(dec, dec_grad, accelerator=accelerator)
            toggle_require_grad(ar_model, prior_grad, accelerator=accelerator)
            toggle_require_grad(gan_loss, disc_grad, accelerator=accelerator)
            toggle_train_eval(ar_model, train=prior_grad, accelerator=accelerator)

        opts = []
        if enc_grad and opt_enc:
            opts.append((enc, opt_enc, scheduler_enc, grad_clip_enc))
        if dec_grad:
            opts.append((dec, opt_dec, scheduler_dec, grad_clip_dec))
        if prior_grad:
            opts.append((ar_model, opt_ar, scheduler_ar, grad_clip_ar))
        if disc_grad:
            opts.append((gan_loss, opt_gan_loss, scheduler_gan_loss, grad_clip_gan))
        

        if internel_step == 0:
            data_buffer = []
        if inner_idx == 0 or len(data_buffer) == 0:
            batch = next(train_loader)
            imgs, labels = batch if isinstance(batch, (list, tuple)) and len(batch) == 2 else (batch, None)
            imgs = img_uint8_to_norm(imgs)
            x = patchify(imgs, config.patch_size)
            uncond_batch = uncond_labels.expand(imgs.shape[0], -1)
            if labels is None or len(labels)==0 or not config.use_label:
                labels = raw_labels = uncond_batch
            else:
                labels = raw_labels = torch.cat([labels, torch.full((labels.shape[0],1), 0, device=accelerator.device, dtype=torch.long)], dim=-1)
                if config.label_drop_prob > 0:
                    drop_mask = torch.rand((imgs.shape[0],1), device=accelerator.device) < config.label_drop_prob
                    labels = torch.where(drop_mask, uncond_batch, labels)
            data_buffer.append((x,imgs,raw_labels,labels))
        else:
            x, imgs, raw_labels, labels = data_buffer.pop(-1)
            uncond_batch = uncond_labels.expand(imgs.shape[0], -1)

        ae_labels = uncond_batch if _ae_no_label else raw_labels

        if enc_grad:
            enc_out = enc(x, ae_labels, training=True)
            cached_enc_out = EncoderOutput(
                quant=enc_out.quant.detach(), indices=enc_out.indices.detach(),
                one_hot=enc_out.one_hot.detach(),
                semantic_vq_loss=enc_out.semantic_vq_loss, visual_vq_loss=enc_out.visual_vq_loss,
                semantic_quant=enc_out.semantic_quant.detach() if enc_out.semantic_quant is not None else None,
                visual_quant=enc_out.visual_quant.detach(),
                semantic_indices=enc_out.semantic_indices.detach() if enc_out.semantic_indices is not None else None,
                visual_indices=enc_out.visual_indices.detach(),
            )
        else:
            enc_out = cached_enc_out

        if target.DO_AE:
            x_hat = dec(enc_out.visual_quant, ae_labels)
            imgs_hat = unpatchify(x_hat, config.image_size, config.patch_size)
            cached_imgs_hat = imgs_hat.detach()

        if target.DO_PRIOR_AR or target.DO_PRIOR_ENC:
            # STE needs encoder one_hot; on cache miss / pretoken mode use idx embedding.
            if _ste_ar_embedding and enc_grad:
                semantic_one_hot = enc_out.semantic_one_hot if _prologue else enc_out.one_hot
            else:
                semantic_one_hot = None
            ar_indices = enc_out.indices
            if _semantic_offset > 0 and _prologue:
                ar_indices = ar_indices.clone()
                ar_indices[:, :config.z_len] += _semantic_offset
            if _use_eos:
                ar_indices = insert_eos_token(ar_indices, int(config.z_len), _unwrap(ar_model).eos_token_id)

            ar_targets = ar_indices
            if _prior_visual_dropout > 0 and _prologue:
                vis_start = config.z_len + _eos_offset
                drop_mask = torch.rand(ar_indices.shape[0], device=ar_indices.device) < _prior_visual_dropout
                if drop_mask.any():
                    ar_input = ar_indices.clone()
                    ar_input[drop_mask, vis_start:] = _unwrap(ar_model).eos_token_id
                else:
                    ar_input = ar_indices
            else:
                ar_input = ar_indices

            ar_out = ar_model(ar_input, labels=labels, semantic_one_hot=semantic_one_hot)
        
        # calculate the loss
        l2_loss = config.l2_weight * F.mse_loss(x, x_hat) if target.DO_L2 else 0.
        l1_loss = config.l1_weight * F.l1_loss(x, x_hat) if target.DO_L1 else 0.
        convnext_loss = 0.
        if target.DO_LPIPS:
            if isinstance(perceptual_loss, BothPerceptualLoss):
                _lpips_val, _convnext_val = perceptual_loss(imgs, imgs_hat)
                lpips_loss = config.lpips_weight * _lpips_val.mean()
                convnext_loss = config.get("convnext_weight", 0.1) * _convnext_val.mean()
            else:
                lpips_loss = config.lpips_weight * perceptual_loss(imgs, imgs_hat).mean()
        else:
            lpips_loss = 0.
        ae_loss = l2_loss + l1_loss + lpips_loss + convnext_loss
        
        semantic_vqloss_dict = enc_out.semantic_vq_loss
        visual_vqloss_dict = enc_out.visual_vq_loss
        if semantic_vqloss_dict is not None:
            semantic_vqloss = config.commitment_loss_weight * semantic_vqloss_dict.quant_loss + config.entropy_loss_weight * semantic_vqloss_dict.entropy_loss if enc_grad else 0.
        else:
            semantic_vqloss = 0.
        visual_vqloss = config.commitment_loss_weight * visual_vqloss_dict.quant_loss + config.entropy_loss_weight * visual_vqloss_dict.entropy_loss if enc_grad else 0.
        vqloss = semantic_vqloss + visual_vqloss

        gan_G_loss, gan_G_loss_dict = gan_loss(imgs, imgs_hat, global_step=global_step, loss='G') if target.DO_GAN_G else (0., {})
        gan_D_loss, gan_D_loss_dict = gan_loss(imgs, cached_imgs_hat, global_step=global_step, loss='D') if target.DO_GAN_D else (0., {})
        
        gan_G_loss_weight = 0.
        adapt_weight = 0.
        if target.DO_GAN_G:
            if config.disc_adaptive_weight:
                adapt_weight = calculate_adaptive_weight_acc(ae_loss, gan_G_loss, accelerator.unwrap_model(dec).dec.out.weight, accelerator, dec)
                gan_G_loss_weight = (config.gan_G_weight * adapt_weight)
            else:
                gan_G_loss_weight = config.gan_G_weight
             
        gan_G_loss = gan_G_loss * gan_G_loss_weight if target.DO_GAN_G else 0.
        gan_D_loss = gan_D_loss * config.gan_D_weight if target.DO_GAN_D else 0.

        prior_ar_loss = 0.
        semantic_prior_ar_loss = 0.
        visual_prior_ar_loss = 0.
        eos_prior_ar_loss = 0.
        correct_token_rate = 0.
        semantic_correct_token_rate = 0.
        visual_correct_token_rate = 0.
        if target.DO_PRIOR_AR:
            ar_logits = ar_out.logits
            
            if _prologue:
                total_len = config.z_len + _eos_offset + config.x_len
                V = ar_logits.shape[-1]
                B = ar_logits.shape[0]

                sem_logits = ar_logits[:, :config.z_len]
                vis_logits = ar_logits[:, config.z_len + _eos_offset:]
                sem_targets = ar_targets[:, :config.z_len]
                vis_targets = ar_targets[:, config.z_len + _eos_offset:]

                sem_loss_per_token = F.cross_entropy(sem_logits.reshape(-1, V), sem_targets.reshape(-1), reduction='none')
                semantic_prior_ar_loss = sem_loss_per_token.reshape(B, -1).sum(dim=1).mean() / total_len
                semantic_correct_token_rate = (sem_logits.argmax(dim=-1) == sem_targets).detach().float().mean().item()

                vis_loss_per_token = F.cross_entropy(vis_logits.reshape(-1, V), vis_targets.reshape(-1), reduction='none')
                visual_prior_ar_loss = vis_loss_per_token.reshape(B, -1).sum(dim=1).mean() / total_len
                visual_correct_token_rate = (vis_logits.argmax(dim=-1) == vis_targets).detach().float().mean().item()

                if _eos_offset > 0:
                    eos_logits = ar_logits[:, config.z_len:config.z_len + 1]
                    eos_targets = ar_targets[:, config.z_len:config.z_len + 1]
                    eos_loss_per_token = F.cross_entropy(eos_logits.reshape(-1, V), eos_targets.reshape(-1), reduction='none')
                    eos_prior_ar_loss = eos_loss_per_token.reshape(B, -1).sum(dim=1).mean() / total_len

                prior_ar_loss = semantic_prior_ar_loss + eos_prior_ar_loss + visual_prior_ar_loss
            else:
                prior_ar_loss = F.cross_entropy(ar_logits.reshape(-1, ar_logits.shape[-1]), ar_targets.reshape(-1))
                visual_prior_ar_loss = prior_ar_loss
                visual_correct_token_rate = (ar_logits.argmax(dim=-1) == ar_targets).detach().float().mean().item()
            correct_token_rate = (ar_logits.argmax(dim=-1) == ar_targets).detach().float().mean().item()
            
        prior_enc_loss = raw_prior_enc_loss = 0.
        semantic_prior_enc_loss = 0.
        visual_prior_enc_loss = 0.
        eos_prior_enc_loss = 0.
        if target.DO_PRIOR_ENC:
            enc_ar_logits = ar_out.logits

            if _prologue:
                total_len_enc = config.z_len + _eos_offset + config.x_len
                V_enc = enc_ar_logits.shape[-1]
                B_enc = enc_ar_logits.shape[0]

                enc_sem_logits = enc_ar_logits[:, :config.z_len]
                enc_vis_logits = enc_ar_logits[:, config.z_len + _eos_offset:]
                enc_sem_targets = ar_targets[:, :config.z_len]
                enc_vis_targets = ar_targets[:, config.z_len + _eos_offset:]

                enc_sem_loss = F.cross_entropy(enc_sem_logits.reshape(-1, V_enc), enc_sem_targets.reshape(-1), reduction='none')
                semantic_prior_enc_loss = enc_sem_loss.reshape(B_enc, -1).sum(dim=1).mean() / total_len_enc

                enc_vis_loss = F.cross_entropy(enc_vis_logits.reshape(-1, V_enc), enc_vis_targets.reshape(-1), reduction='none')
                visual_prior_enc_loss = enc_vis_loss.reshape(B_enc, -1).sum(dim=1).mean() / total_len_enc

                if _eos_offset > 0:
                    enc_eos_logits = enc_ar_logits[:, config.z_len:config.z_len + 1]
                    enc_eos_targets = ar_targets[:, config.z_len:config.z_len + 1]
                    enc_eos_loss = F.cross_entropy(enc_eos_logits.reshape(-1, V_enc), enc_eos_targets.reshape(-1), reduction='none')
                    eos_prior_enc_loss = enc_eos_loss.reshape(B_enc, -1).sum(dim=1).mean() / total_len_enc
            else:
                enc_vis_logits = enc_ar_logits
                enc_vis_targets = ar_targets
                visual_prior_enc_loss = F.cross_entropy(enc_vis_logits.reshape(-1, enc_vis_logits.shape[-1]), enc_vis_targets.reshape(-1))

            raw_prior_enc_loss = semantic_prior_enc_loss + eos_prior_enc_loss + visual_prior_enc_loss
            prior_enc_loss = config.prior_enc_semantic_weight * (semantic_prior_enc_loss + eos_prior_enc_loss) + config.prior_enc_visual_weight * visual_prior_enc_loss

        loss = ae_loss + prior_ar_loss + gan_G_loss + gan_D_loss + vqloss + prior_enc_loss

        # backward and optimization
        accelerator.backward(loss)
        grad_norms = calc_grad_norm(
            {"Enc": enc, "Dec": dec, "AR": ar_model if train_ar else None, "GAN": gan_loss if use_gan_loss else None},
            global_step, int(getattr(config, "grad_norm_freq", 0)),
            accelerator=accelerator,
        )
        for model, opt, scheduler, clip_val in opts:
            if model is not None and opt is not None:
                cur_lr = max(pg['lr'] for pg in opt.param_groups)
                if cur_lr > 0:
                    zero_nan_gradients(model, accelerator=accelerator)
                    if clip_val > 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                    opt.step()
                opt.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        
        # EMA updates (skip when lr == 0 since params were not updated)
        if enc_grad and opt_enc is not None and max(pg['lr'] for pg in opt_enc.param_groups) > 0:
            ema_update(enc, enc_ema, ae_ema_rate)
        if dec_grad and opt_dec is not None and max(pg['lr'] for pg in opt_dec.param_groups) > 0:
            ema_update(dec, dec_ema, ae_ema_rate)
          
        if train_ar and prior_grad and opt_ar is not None and max(pg['lr'] for pg in opt_ar.param_groups) > 0:
            ema_update(ar_model, ar_model_ema, ar_ema_rate)

        #visualization
        if (global_step + 1) % config.visualize_freq == 0:
            grid = x[:config.visualize_img_num]
            
            # enc/dec: toggle whenever either ema flag is off (any non-EMA path needs train-mode toggling)
            if ((not config.ema_sampling) or (not config.ema_reconstruction)) and train_ae:
                toggle_train_eval(enc, train=False, accelerator=accelerator) 
                toggle_train_eval(dec, train=False, accelerator=accelerator)
            
            # ar: only ema_sampling controls whether to toggle
            if not config.ema_sampling and train_ar:
                toggle_train_eval(ar_model, train=False, accelerator=accelerator)
            
            # Pick the models used for recon vs sampling
            if config.ema_reconstruction:
                recon_enc, recon_dec = enc_ema, dec_ema
            else:
                recon_enc, recon_dec = enc, dec
                
            if config.ema_sampling:
                sample_enc, sample_dec, sample_ar = enc_ema, dec_ema, ar_model_ema
            else:
                sample_enc, sample_dec, sample_ar = enc, dec, ar_model
                
            with torch.no_grad():
                if train_ae:
                    all_recon_images,_ = reconstruction(recon_enc, recon_dec, x, ae_labels)
                    grid = torch.cat([grid, all_recon_images[:config.visualize_img_num]], dim=0)

                if train_ar:
                    _viz_cls = uncond_batch if _label_drop_always else raw_labels
                    all_sample_images = sampling(sample_enc, sample_dec, sample_ar, bz=x.shape[0], 
                                            class_label=_viz_cls,temperature=config.temperature, 
                                            topK=config.topK, topP=config.topP, 
                                            cfg=config.cfg, cfg_schedule=config.cfg_schedule, cfg_power=config.cfg_power, 
                                            cache_kv=config.cache_kv,
                                            ae_label=ae_labels,
                                            semantic_cfg_schedule=config.get("semantic_cfg_schedule", None),
                                            semantic_cfg_scale=config.get("semantic_cfg_scale", None),
                                            semantic_cfg_power=config.get("semantic_cfg_power", None),
                                            semantic_cfg_start=float(config.get("semantic_cfg_start", 0.0)),
                                            visual_cfg_schedule=config.get("visual_cfg_schedule", None),
                                            visual_cfg_scale=config.get("visual_cfg_scale", None),
                                            visual_cfg_power=config.get("visual_cfg_power", None),
                                            visual_cfg_start=float(config.get("visual_cfg_start", 1.0)))
                    grid = torch.cat([grid, all_sample_images[:config.visualize_img_num]], dim=0)

                grid = img_denormalize(unpatchify(grid, config.image_size, config.patch_size))
                grid = torchvision.utils.make_grid(grid, nrow=config.visualize_img_num, normalize=False)
                accelerator.log({"visualization/imgs":wandb.Image(grid),'global_step':global_step+1}, step=global_step+1)
            
            # Toggle back to train mode
            if ((not config.ema_sampling) or (not config.ema_reconstruction)) and train_ae:
                toggle_train_eval(enc, train=True, accelerator=accelerator) 
                toggle_train_eval(dec, train=True, accelerator=accelerator)
            if not config.ema_sampling and train_ar:
                toggle_train_eval(ar_model, train=True, accelerator=accelerator) 
           
        # eval and save ckpt
        rFID = 0.0
        gFID = 0.0
        # Also keep per-GPU sums/count so final aggregation can be weighted-correct in metrics block
        eval_lpips = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        eval_ssim = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        eval_psnr = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        semantic_codebook_usage = torch.zeros(config.z_len, config.SemanticQuantizer.codebook_size, device=accelerator.device).long() if (train_ae and _prologue) else None
        visual_codebook_usage = torch.zeros(config.x_len if _prologue else config.z_len, config.codebook_size, device=accelerator.device).long() if train_ae else None
        codebook_usage = None
        avg_agg_ent = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        entropy_acc = {"ent_sum": None, "ent_cnt": None}
        entropy_log_base = float(getattr(config, "prefix_entropy_log_base", 2.0))
        data_cond_entropy_trie = None  # For train_ae: collect real token sequences

        if eval_loader is not None and (global_step + 1) % config.eval_freq == 0:
            sample_cached_path = os.path.join(config.tmp_dir, "sample_images.npz")
            recon_cached_path = os.path.join(config.tmp_dir, "recon_images.npz")
            gt_cache_path = config.eval_fid_ref_path

            gt_buf = []
            samples_buf = []
            recons_buf = []
            semantic_idx_buf = []
            visual_idx_buf = []
            lpips_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            ssim_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            psnr_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            count = torch.zeros((), device=accelerator.device, dtype=torch.long)
            
            # enc/dec: toggle whenever either ema flag is off (any non-EMA path needs train-mode toggling)
            if ((not config.ema_sampling) or (not config.ema_reconstruction)) and train_ae:
                toggle_train_eval(enc, train=False, accelerator=accelerator) 
                toggle_train_eval(dec, train=False, accelerator=accelerator)
            
            # ar: only ema_sampling controls whether to toggle
            if not config.ema_sampling and train_ar:
                toggle_train_eval(ar_model, train=False, accelerator=accelerator)
            
            # Pick the models used for recon vs sampling
            if config.ema_reconstruction:
                recon_enc, recon_dec = enc_ema, dec_ema
            else:
                recon_enc, recon_dec = enc, dec
                
            if config.ema_sampling:
                sample_enc, sample_dec, sample_ar = enc_ema, dec_ema, ar_model_ema
            else:
                sample_enc, sample_dec, sample_ar = enc, dec, ar_model
            
            cuda_rng_state = torch.cuda.get_rng_state()
            eval_seed = int(config.seed) + accelerator.process_index
            torch.cuda.manual_seed(eval_seed)
            print0(f"[Eval] Per-rank CUDA seed: base={config.seed}, rank={accelerator.process_index}, effective={eval_seed}")

            with torch.no_grad():
                accelerator.wait_for_everyone()
                for i, batch in enumerate(tqdm(eval_loader,disable=not accelerator.is_main_process,dynamic_ncols=True,file=sys.stdout,desc="Evaluating")):
                    imgs, labels = batch if isinstance(batch, (list, tuple)) and len(batch) == 2 else (batch, None)
                    imgs_norm = img_uint8_to_norm(imgs)
                    x = patchify(imgs_norm, config.patch_size)
                    uncond_batch = uncond_labels.expand(imgs.shape[0], -1)
                    if labels is None or len(labels)==0 or not config.use_label:
                        labels = uncond_batch
                    else:
                        labels = torch.cat([labels, torch.full((labels.shape[0],1), 0, device=accelerator.device, dtype=torch.long)], dim=-1)
                    eval_ae_labels = uncond_batch if _ae_no_label else labels
                    # Align AR eval (encode / teacher-forcing / sampling) with unconditional training when label_drop_prob>=1.0
                    ar_eval_cls = uncond_batch if _label_drop_always else labels
                    idx = None
                    if train_ae:
                        recon_patches, idx = reconstruction(recon_enc, recon_dec, x, eval_ae_labels)  # patch domain
                        recon_img = unpatchify(recon_patches, config.image_size, config.patch_size)  # [B,C,H,W] float in [-1,1]

                        lpips_vals, ssim_vals, psnr_vals = calc_per_sample_reconstruction_metrics(imgs_norm, recon_img, perceptual_loss)
                        lpips_sum += lpips_vals
                        ssim_sum += ssim_vals
                        psnr_sum += psnr_vals
                        count += 1

                        recon_img_u8 = img_norm_to_uint8(recon_img)
                        recons_buf.append(recon_img_u8.permute(0,2,3,1).cpu().numpy())
                    elif train_ar:
                        idx = sample_enc(x, ar_eval_cls, training=False).indices

                    if train_ar:
                        ar_eval_idx = idx
                        if _semantic_offset > 0 and _prologue:
                            ar_eval_idx = idx.clone()
                            ar_eval_idx[:, :config.z_len] += _semantic_offset
                        if _use_eos:
                            ar_eval_idx = insert_eos_token(ar_eval_idx, int(config.z_len), _unwrap(ar_model).eos_token_id)
                        ar_logits = sample_ar(ar_eval_idx, ar_eval_cls).logits
                        draw_conditional_entropy(
                            entropy_acc,
                            logits=ar_logits,
                            accelerator=accelerator,
                            save_dir=save_dir,
                            global_step=global_step,
                            log_base=entropy_log_base,
                            finalize=False,
                            codebook_size=config.codebook_size,
                        )
                    if train_ar:
                        sample_images = sampling(sample_enc, sample_dec, sample_ar, 
                                        bz=x.shape[0], class_label=ar_eval_cls,temperature=config.temperature, 
                                        topK=config.topK, topP=config.topP, cfg=config.cfg, cfg_schedule=config.cfg_schedule, 
                                        cfg_power=config.cfg_power, cache_kv=config.cache_kv,
                                        ae_label=eval_ae_labels,
                                        semantic_cfg_schedule=config.get("semantic_cfg_schedule", None),
                                        semantic_cfg_scale=config.get("semantic_cfg_scale", None),
                                        semantic_cfg_power=config.get("semantic_cfg_power", None),
                                        semantic_cfg_start=float(config.get("semantic_cfg_start", 0.0)),
                                        visual_cfg_schedule=config.get("visual_cfg_schedule", None),
                                        visual_cfg_scale=config.get("visual_cfg_scale", None),
                                        visual_cfg_power=config.get("visual_cfg_power", None),
                                        visual_cfg_start=float(config.get("visual_cfg_start", 1.0)))
                        sample_images = img_norm_to_uint8(unpatchify(sample_images, config.image_size, config.patch_size))
                        
                        samples_buf.append(sample_images.permute(0,2,3,1).cpu().numpy())
                    
                    if not os.path.exists(gt_cache_path):
                        gt_buf.append(imgs.permute(0,2,3,1).cpu().numpy())

                    if train_ae and visual_codebook_usage is not None and idx is not None:
                        if _prologue:
                            semantic_idx_buf.append(idx[:, :-config.x_len])
                            visual_idx_buf.append(idx[:, -config.x_len:])
                        else:
                            visual_idx_buf.append(idx.detach())
                    
                    # Collect token sequences for data conditional entropy
                    if train_ae and idx is not None:
                        data_cond_entropy_trie = draw_data_conditional_entropy(
                            data_cond_entropy_trie,
                            idx=idx,
                            accelerator=accelerator,
                            save_dir=save_dir,
                            global_step=global_step,
                            log_base=entropy_log_base,
                            finalize=False,
                            max_depth=config.z_len,
                            codebook_size=config.codebook_size,
                        )
                
                accelerator.wait_for_everyone()

                if train_ae:
                    if semantic_codebook_usage is not None and len(semantic_idx_buf) > 0:
                        s_idx_all = accelerator.gather(torch.cat(semantic_idx_buf, dim=0))
                        if accelerator.is_main_process:
                            s_idx_all = s_idx_all.to(device=semantic_codebook_usage.device, dtype=torch.long)
                            s_pos = torch.arange(s_idx_all.shape[1], device=semantic_codebook_usage.device, dtype=torch.long).unsqueeze(0).expand_as(s_idx_all).reshape(-1)
                            semantic_codebook_usage[s_pos, s_idx_all.reshape(-1)] += 1
                    if visual_codebook_usage is not None and len(visual_idx_buf) > 0:
                        v_idx_all = accelerator.gather(torch.cat(visual_idx_buf, dim=0))
                        if accelerator.is_main_process:
                            v_idx_all = v_idx_all.to(device=visual_codebook_usage.device, dtype=torch.long)
                            v_pos = torch.arange(v_idx_all.shape[1], device=visual_codebook_usage.device, dtype=torch.long).unsqueeze(0).expand_as(v_idx_all).reshape(-1)
                            visual_codebook_usage[v_pos, v_idx_all.reshape(-1)] += 1

                if train_ae:
                    eval_lpips = lpips_sum / count
                    eval_ssim = ssim_sum / count
                    eval_psnr = psnr_sum / count
                    print0("eval_lpips",eval_lpips,"eval_ssim",eval_ssim,"eval_psnr",eval_psnr)

                # Gather all buffers at once and save
                gathered_samples = accelerator.gather(torch.from_numpy(np.concatenate(samples_buf, axis=0)).to(accelerator.device)).cpu().numpy() if train_ar and len(samples_buf) > 0 else None
                gathered_recons = accelerator.gather(torch.from_numpy(np.concatenate(recons_buf, axis=0)).to(accelerator.device)).cpu().numpy() if train_ae and len(recons_buf) > 0 else None
                gathered_gt = accelerator.gather(torch.from_numpy(np.concatenate(gt_buf, axis=0)).to(accelerator.device)).cpu().numpy() if len(gt_buf) > 0 else None

                if accelerator.is_main_process:
                    if gathered_samples is not None:
                        print0(f"[Eval] Gathered samples: {gathered_samples.shape[0]}, eval_dataset: {eval_dataset_size}")
                        if gathered_samples.shape[0] != eval_dataset_size:
                            print0(f"WARNING: Gathered samples count ({gathered_samples.shape[0]}) != eval_dataset size ({eval_dataset_size})")
                        np.savez(sample_cached_path, gathered_samples)
                    if gathered_recons is not None:
                        print0(f"[Eval] Gathered recons: {gathered_recons.shape[0]}, eval_dataset: {eval_dataset_size}")
                        if gathered_recons.shape[0] != eval_dataset_size:
                            print0(f"WARNING: Gathered recons count ({gathered_recons.shape[0]}) != eval_dataset size ({eval_dataset_size})")
                        np.savez(recon_cached_path, gathered_recons)
                    if gathered_gt is not None:
                        print0(f"[Eval] Gathered gt: {gathered_gt.shape[0]}, eval_dataset: {eval_dataset_size}")
                        if gathered_gt.shape[0] != eval_dataset_size:
                            print0(f"WARNING: Gathered gt count ({gathered_gt.shape[0]}) != eval_dataset size ({eval_dataset_size})")
                        if not os.path.exists(gt_cache_path):
                            np.savez(gt_cache_path, gathered_gt)
                    
                    if train_ar:
                        gFID = adm_fid_evaluator(sample_cached_path, gt_cache_path, config, accelerator)
                    if train_ae:
                        rFID = adm_fid_evaluator(recon_cached_path, gt_cache_path, config, accelerator)
                accelerator.wait_for_everyone()

                # Sync FID across all processes to ensure consistent checkpoint naming
                rFID_t = torch.tensor(rFID, device=accelerator.device)
                gFID_t = torch.tensor(gFID, device=accelerator.device)
                rFID = accelerator.reduce(rFID_t, reduction="sum").item()
                gFID = accelerator.reduce(gFID_t, reduction="sum").item()

            if config.save_best and config.save_ckpt:
                if train_ae and rFID < best_rfid:
                    best_rfid = float(rFID)
                    if accelerator.is_main_process:
                        remove_old_best_checkpoints(f"{save_dir}/ckpts", metric_type="rFID")
                    accelerator.wait_for_everyone()
                    save_training_state(accelerator, f"{save_dir}/ckpts/best-Step={global_step+1}-rFID={best_rfid:.4f}", extra_training_states,
                        global_step=global_step + 1, dl_generator=train_loader._pre_epoch_gen_state,
                        total_yielded=train_loader.total_yielded, prev_phase_idx=phase_idx, prev_inner_idx=inner_idx,
                        data_buffer=[tuple(t.detach().cpu() for t in item) for item in data_buffer],
                        **make_cache_kwargs(cached_enc_out, cached_imgs_hat),
                        best_rfid=best_rfid)
                    if accelerator.is_main_process:
                        print0(f"[best] Saved new best rFID checkpoint: {best_rfid:.4f}")
                if train_ar and gFID < best_gfid:
                    best_gfid = float(gFID)
                    if accelerator.is_main_process:
                        remove_old_best_checkpoints(f"{save_dir}/ckpts", metric_type="gFID")
                    accelerator.wait_for_everyone()
                    save_training_state(accelerator, f"{save_dir}/ckpts/best-Step={global_step+1}-gFID={best_gfid:.4f}", extra_training_states,
                        global_step=global_step + 1, dl_generator=train_loader._pre_epoch_gen_state,
                        total_yielded=train_loader.total_yielded, prev_phase_idx=phase_idx, prev_inner_idx=inner_idx,
                        data_buffer=[tuple(t.detach().cpu() for t in item) for item in data_buffer],
                        **make_cache_kwargs(cached_enc_out, cached_imgs_hat),
                        best_gfid=best_gfid)
                    if accelerator.is_main_process:
                        print0(f"[best] Saved new best gFID checkpoint: {best_gfid:.4f}")

            if train_ar:
                draw_conditional_entropy(
                    entropy_acc,
                    accelerator=accelerator,
                    save_dir=save_dir,
                    global_step=global_step,
                    log_base=entropy_log_base,
                    finalize=True,
                    rFID=rFID,
                    gFID=gFID,
                    codebook_size=config.codebook_size,
                )

            if train_ae:
                # Plot data conditional entropy (from real token sequences)
                draw_data_conditional_entropy(
                    data_cond_entropy_trie,
                    accelerator=accelerator,
                    save_dir=save_dir,
                    global_step=global_step,
                    log_base=entropy_log_base,
                    finalize=True,
                    rFID=rFID,
                    gFID=gFID,
                    max_depth=config.z_len,
                    codebook_size=config.codebook_size,
                )
                
                if semantic_codebook_usage is not None and visual_codebook_usage is not None:
                    max_cb = max(semantic_codebook_usage.shape[1], visual_codebook_usage.shape[1])
                    s_pad = F.pad(semantic_codebook_usage, (0, max_cb - semantic_codebook_usage.shape[1])) if semantic_codebook_usage.shape[1] < max_cb else semantic_codebook_usage
                    v_pad = F.pad(visual_codebook_usage, (0, max_cb - visual_codebook_usage.shape[1])) if visual_codebook_usage.shape[1] < max_cb else visual_codebook_usage
                    codebook_usage = torch.cat([s_pad, v_pad], dim=0)
                elif visual_codebook_usage is not None:
                    codebook_usage = visual_codebook_usage
                else:
                    codebook_usage = None
                plot_codebook_usage(
                    codebook_usage,
                    accelerator=accelerator,
                    save_dir=save_dir,
                    global_step=global_step,
                    rFID=rFID,
                    gFID=gFID,
                )
                
                if codebook_usage is not None and accelerator.is_main_process:
                    aggregated_ent = compute_aggregated_entropy_from_counts(codebook_usage, log_base=entropy_log_base)  # [L]
                    aggregated_ent_list = aggregated_ent.detach().cpu().tolist()
                    
                    avg_agg_ent_val = float(np.nanmean(np.array(aggregated_ent_list, dtype=np.float64)))
                    avg_agg_ent = torch.tensor(avg_agg_ent_val, device=accelerator.device, dtype=torch.float32)
                    
                    plot_posterior_entropy(
                        sample_entropy=None,
                        aggregated_entropy=aggregated_ent_list,
                        accelerator=accelerator,
                        save_dir=save_dir,
                        global_step=global_step,
                        rFID=rFID,
                        gFID=gFID,
                        log_base=entropy_log_base,
                        codebook_size=config.codebook_size,
                    )
                
                # Reduce avg_agg_ent across all processes
                if train_ae:
                    avg_agg_ent = accelerator.reduce(avg_agg_ent, reduction='sum')
                    
            torch.cuda.set_rng_state(cuda_rng_state)

            # Toggle back to train mode
            if ((not config.ema_sampling) or (not config.ema_reconstruction)) and train_ae:
                toggle_train_eval(enc, train=True, accelerator=accelerator) 
                toggle_train_eval(dec, train=True, accelerator=accelerator)
            if not config.ema_sampling and train_ar:
                toggle_train_eval(ar_model, train=True, accelerator=accelerator) 

            # Cleanup temporary npz files produced during this eval
            if accelerator.is_main_process:
                if train_ar:
                    safe_remove_file(sample_cached_path)
                if train_ae:
                    safe_remove_file(recon_cached_path)
            
            # Explicitly clear large buffers to free memory
            del samples_buf, recons_buf, gt_buf, semantic_idx_buf, visual_idx_buf
            if 'gathered_samples' in locals(): del gathered_samples
            if 'gathered_recons' in locals(): del gathered_recons
            if 'gathered_gt' in locals(): del gathered_gt
            gc.collect()
            torch.cuda.empty_cache()
            
            accelerator.wait_for_everyone()

        # metrics
        if (global_step +1) % config.log_freq == 0:
            lr_enc = opt_enc.param_groups[0]['lr'] if opt_enc is not None else 0.
            lr_dec = opt_dec.param_groups[0]['lr'] if opt_dec is not None else 0.
            lr_ar = opt_ar.param_groups[0]['lr'] if opt_ar is not None else 0.
            lr_gan_loss = opt_gan_loss.param_groups[0]['lr'] if opt_gan_loss is not None else 0.
            metrics = {
                        'Phase':phase_idx,
                        'ae_loss/l1_loss':l1_loss,
                        'ae_loss/l2_loss':l2_loss,
                        'ae_loss/lpips_loss':lpips_loss,
                        'ae_loss/convnext_loss':convnext_loss,
                        'ae_loss/loss':ae_loss, 
                        'ae_loss/vqloss': vqloss,
                        'ae_loss/semantic_vqloss': semantic_vqloss,
                        'ae_loss/visual_vqloss': visual_vqloss,
                        **(({
                        'semantic_vqloss/quant_loss': semantic_vqloss_dict.quant_loss.detach(),
                        'semantic_vqloss/entropy_loss': semantic_vqloss_dict.entropy_loss.detach(),
                        'semantic_vqloss/sample_entropy': semantic_vqloss_dict.sample_entropy,
                        'semantic_vqloss/batch_entropy': semantic_vqloss_dict.batch_entropy,
                        'semantic_vqloss/z_norm': semantic_vqloss_dict.l2norm_z,
                        'semantic_vqloss/code_norm': semantic_vqloss_dict.l2norm_code,
                        }) if semantic_vqloss_dict is not None else {}),
                        'visual_vqloss/quant_loss': visual_vqloss_dict.quant_loss.detach(),
                        'visual_vqloss/entropy_loss': visual_vqloss_dict.entropy_loss.detach(),
                        'visual_vqloss/sample_entropy': visual_vqloss_dict.sample_entropy,
                        'visual_vqloss/batch_entropy': visual_vqloss_dict.batch_entropy,
                        'visual_vqloss/z_norm': visual_vqloss_dict.l2norm_z,
                        'visual_vqloss/code_norm': visual_vqloss_dict.l2norm_code,
                        'GAN/gan_G_loss':gan_G_loss,
                        'GAN/gan_D_loss':gan_D_loss,
                        'prior_loss/ar_loss':prior_ar_loss,
                        'prior_loss/semantic_ar_loss':semantic_prior_ar_loss,
                        'prior_loss/visual_ar_loss':visual_prior_ar_loss,
                        'prior_loss/eos_ar_loss':eos_prior_ar_loss,
                        'prior_loss/prior_enc_loss':raw_prior_enc_loss,
                        'prior_loss/semantic_prior_enc_loss':semantic_prior_enc_loss,
                        'prior_loss/visual_prior_enc_loss':visual_prior_enc_loss,
                        'prior_loss/eos_prior_enc_loss':eos_prior_enc_loss,
                        'prior_loss/correct_token_rate': correct_token_rate,
                        'prior_loss/semantic_correct_token_rate': semantic_correct_token_rate,
                        'prior_loss/visual_correct_token_rate': visual_correct_token_rate,
                        'eval/semantic_codebook_usage': ((semantic_codebook_usage > 0).any(dim=0).float().mean().item() if (train_ae and semantic_codebook_usage is not None) else 0.0),
                        'eval/visual_codebook_usage': ((visual_codebook_usage > 0).any(dim=0).float().mean().item() if (train_ae and visual_codebook_usage is not None) else 0.0),
                        'eval/rFID':rFID,
                        'eval/gFID':gFID,
                        'eval/lpips': eval_lpips,
                        'eval/psnr':eval_psnr,
                        'eval/ssim':eval_ssim,
                        'eval/aggregated_post_entropy': avg_agg_ent,
                        'lr/lr_enc': lr_enc,
                        'lr/lr_dec': lr_dec,
                        'lr/lr_ar': lr_ar,
                        'lr/lr_gan_loss': lr_gan_loss,
                      }

            if target.DO_GAN_G or target.DO_GAN_D:
                gan_loss_dict = {**gan_G_loss_dict, **gan_D_loss_dict, 'GAN/gan_G_loss_weight':gan_G_loss_weight, 'GAN/adapt_weight':adapt_weight}
                metrics.update(gan_loss_dict)
            metrics = {k: accelerator.reduce(v, reduction='mean').item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()  }
            
            metrics_logger = {k: v for k, v in metrics.items() if v!=0. }
            metrics_logger.update(grad_norms)
            metrics_4f = {k: f"{v:.4f}" if k!='Phase' else int(v) for k, v in metrics.items() }
            accelerator.log(metrics_logger, step=global_step+1)
            pbar.set_postfix(metrics_4f, refresh=False)
            pbar.update(config.log_freq)

        if config.save_ckpt and ((global_step+1) % config.ckpt_freq == 0):
            ckpt_name = f'Phase={phase_idx}-Step={global_step+1}-rFID={rFID:.4f}-gFID={gFID:.4f}' if rFID is not None and gFID is not None else "-".join([f'{k}={v:.4f}' for k, v in metrics.items()]) 
            save_path = f'{save_dir}/ckpts/{ckpt_name}'
            save_training_state(accelerator, save_path, extra_training_states,
                global_step=global_step + 1, dl_generator=train_loader._pre_epoch_gen_state,
                total_yielded=train_loader.total_yielded, prev_phase_idx=phase_idx, prev_inner_idx=inner_idx,
                data_buffer=[tuple(t.detach().cpu() for t in item) for item in data_buffer],
                **make_cache_kwargs(cached_enc_out, cached_imgs_hat))
            
        global_step += 1
        prev_phase_idx = phase_idx
        prev_inner_idx = inner_idx

    # Save last state
    if config.save_ckpt:
        save_path = f'{save_dir}/ckpts/last-Step={global_step}-rFID={rFID:.4f}-gFID={gFID:.4f}'
        save_training_state(accelerator, save_path, extra_training_states,
            global_step=global_step, dl_generator=train_loader._pre_epoch_gen_state,
            total_yielded=train_loader.total_yielded, prev_phase_idx=phase_idx, prev_inner_idx=inner_idx,
            data_buffer=[tuple(t.detach().cpu() for t in item) for item in data_buffer],
            **make_cache_kwargs(cached_enc_out, cached_imgs_hat))
    
    accelerator.wait_for_everyone()
    pbar.close()

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            tracker.finish()
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    from utils import load_config
    train(load_config())
