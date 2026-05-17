"""Class-conditional sampling and prologue-fix resampling for Prologue models.

Two modes:

    sample        Class-conditional generation grid (one row per class).
    prologue_fix  Sample a reference; freeze its first z_len prologue tokens
                  and resample the remaining visual tokens. Requires a Prologue
                  tokenizer (configs/tokenizer/prologue.yaml).

The functions below are also used as a library by `app.py` (Gradio demo).

CLI usage:

    python sample_vis.py --configs=configs/default.yaml,configs/ar/_defaults.yaml,\\
        configs/ar/xlarge.yaml,configs/tokenizer/default.yaml,\\
        configs/tokenizer/prologue.yaml,configs/train/ar.yaml,configs/train/eval_ar.yaml \\
        tokenizer_ckpt_path=<tok> resume_ckpt_path=<ar> \\
        mode=prologue_fix class_ids="207,388" num_resample=8 \\
        output_dir=out/

Module-internal attributes (``semantic_emb``, ``z_len``, ``semantic_drop``, ...) match
the released safetensors keys; "prologue" is the user-facing name everywhere else.
"""

import math
import os

import torch
import torch.nn.functional as F
import torchvision.utils
from safetensors.torch import load_file

from train_ar import (
    _codes_from_indices,
    _labels_from_label_idx,
    _load_decoder,
    _load_quantizer,
    _load_semantic_quantizer,
)
from models import ARModel
from utils import (
    build_ar_logit_mask,
    img_denormalize,
    load_config,
    print0,
    save_tensor_image_png_pdf,
    seed_everything,
    unpatchify,
)

torch.backends.cudnn.benchmark = True


IMAGENET_NAMES = {
    33: "loggerhead_turtle", 88: "macaw", 90: "lorikeet", 94: "hummingbird",
    100: "black_swan", 107: "jellyfish", 117: "chambered_nautilus", 130: "flamingo",
    144: "pelican", 146: "albatross", 207: "golden_retriever", 250: "Siberian_husky",
    259: "Pomeranian", 279: "arctic_fox", 281: "tabby_cat", 291: "lion",
    292: "tiger", 293: "cheetah", 295: "brown_bear", 323: "monarch_butterfly",
    340: "zebra", 360: "otter", 386: "African_elephant", 387: "red_panda",
    388: "giant_panda", 417: "balloon", 628: "liner", 817: "sports_car",
    927: "trifle", 928: "ice_cream", 930: "French_loaf", 933: "cheeseburger",
    934: "hotdog", 963: "pizza", 971: "bubble", 972: "cliff", 973: "coral_reef",
    978: "seashore", 979: "valley", 980: "volcano", 985: "daisy", 988: "acorn",
    996: "alp",
}

DEFAULT_CLASS_IDS = [
    207, 388, 387, 88, 130, 279, 417, 928,
    980, 973, 985, 33, 360, 250, 293, 323,
]


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_models(config, device):
    """Return ``(quantizer, dec, prologue_quantizer, ar_model)``; prologue_quantizer is ``None`` for 1D/2D tokenizers."""
    prologue = (
        bool(config.get("Prologue", False))
        and not bool(config.get("share_semantic_codebook", False))
    )
    tok_ckpt = config.tokenizer_ckpt_path

    quantizer = _load_quantizer(config, ckpt_dir=tok_ckpt).to(device)
    dec = _load_decoder(config, ckpt_dir=tok_ckpt).to(device)

    prologue_quantizer = None
    if prologue:
        prologue_quantizer = _load_semantic_quantizer(
            config, ckpt_dir=tok_ckpt
        ).to(device)
        print0("Prologue: loaded prologue (semantic) quantizer")

    ar_model = ARModel(config)
    _logit_mask = build_ar_logit_mask(
        getattr(quantizer, "pos_select_mask", None),
        getattr(prologue_quantizer, "pos_select_mask", None) if prologue else None,
        vis_cb_size=int(config["Quantizer"]["codebook_size"]),
        sem_cb_size=int(config["SemanticQuantizer"]["codebook_size"]) if prologue else 0,
    )
    ar_model.set_logit_mask(_logit_mask)

    ema = bool(config.get("ema_sampling", False))

    if bool(config.get("continuous_training", False)) and tok_ckpt:
        # OneStage: AR weights live in the tokenizer ckpt as model_5/6.safetensors.
        path = os.path.join(tok_ckpt, "model_6.safetensors" if ema else "model_5.safetensors")
        if ema and not os.path.exists(path):
            path = os.path.join(tok_ckpt, "model_5.safetensors")
            print0("AR EMA not found, falling back to regular weights")
        print0(f"Loading AR weights (joint training): {path}")
        ar_model.load_state_dict(load_file(path), strict=True)
    elif getattr(config, "resume_ckpt_path", ""):
        ckpt = config.resume_ckpt_path
        fname = "model_1.safetensors" if ema else "model.safetensors"
        path = os.path.join(ckpt, fname)
        if ema and not os.path.exists(path):
            path = os.path.join(ckpt, "model.safetensors")
            print0("AR EMA not found, falling back to regular weights")
        print0(f"Loading AR weights: {path}")
        ar_model.load_state_dict(load_file(path), strict=False)
    else:
        raise ValueError(
            "Must provide resume_ckpt_path or "
            "(continuous_training=True + tokenizer_ckpt_path)"
        )

    ar_model.to(device).eval()
    print0(
        f"AR model ready  (z_len={ar_model.z_len}, "
        f"max_length={ar_model.max_length})"
    )
    return quantizer, dec, prologue_quantizer, ar_model


# ═══════════════════════════════════════════════════════════════════════════
# Sampling helpers
# ═══════════════════════════════════════════════════════════════════════════

def _cfg_params(config):
    """Pack every CFG-related field into a kwargs dict for ARModel.sampling."""
    get = config.get
    return dict(
        temperature=config.temperature,
        topK=config.topK,
        topP=config.topP,
        cfg=config.cfg,
        cfg_schedule=config.cfg_schedule,
        cfg_power=config.cfg_power,
        cache_kv=config.cache_kv,
        semantic_cfg=get("semantic_cfg", None),
        semantic_cfg_schedule=get("semantic_cfg_schedule", None),
        semantic_cfg_scale=get("semantic_cfg_scale", None),
        semantic_cfg_power=get("semantic_cfg_power", None),
        semantic_cfg_start=float(get("semantic_cfg_start", 0.0)),
        visual_cfg_schedule=get("visual_cfg_schedule", None),
        visual_cfg_scale=get("visual_cfg_scale", None),
        visual_cfg_power=get("visual_cfg_power", None),
        visual_cfg_start=float(get("visual_cfg_start", 1.0)),
        cfg_continuous=bool(get("cfg_continuous", False)),
        semantic_temperature=(
            float(get("semantic_temperature"))
            if get("semantic_temperature") is not None
            else None
        ),
    )


@torch.no_grad()
def sample_tokens(ar_model, *, bz, class_label, config):
    """Thin wrapper over ``ARModel.sampling``; returns token ids ``[bz, max_length]``."""
    get = config.get
    sem_temp = get("semantic_temperature")
    return ar_model.sampling(
        bz, class_label,
        config.temperature, config.topK, config.topP,
        config.cfg, config.cfg_schedule, config.cfg_power,
        config.cache_kv,
        semantic_cfg_schedule=get("semantic_cfg_schedule", None),
        semantic_cfg_scale=get("semantic_cfg_scale", None),
        semantic_cfg_power=get("semantic_cfg_power", None),
        semantic_cfg_start=float(get("semantic_cfg_start", 0.0)),
        visual_cfg_schedule=get("visual_cfg_schedule", None),
        visual_cfg_scale=get("visual_cfg_scale", None),
        visual_cfg_power=get("visual_cfg_power", None),
        visual_cfg_start=float(get("visual_cfg_start", 1.0)),
        semantic_temperature=float(sem_temp) if sem_temp is not None else None,
    )


def decode_tokens(quantizer, dec, token_ids, *, config, ae_label):
    """Decode token ids to images ``[B, 3, H, W]`` in [0, 1]."""
    prologue = (
        bool(config.get("Prologue", False))
        and not bool(config.get("share_semantic_codebook", False))
    )
    if prologue:
        z_len = int(config.z_len)
        eos_len = 1 if bool(config.get("use_eos", False)) and z_len > 0 else 0
        visual_ids = token_ids[:, z_len + eos_len:]
        quant = _codes_from_indices(quantizer, visual_ids, ae_label)
    else:
        quant = _codes_from_indices(quantizer, token_ids, ae_label)
    patches = dec(quant, ae_label)
    return img_denormalize(unpatchify(patches, config.image_size, config.patch_size))


def _make_labels(class_id, bz, num_classes, device):
    """Build ``(class_label, unconditional_label)`` each ``[bz, num_classes]``."""
    uncond_idx = num_classes - 1
    idx = torch.full((bz,), class_id, device=device, dtype=torch.long)
    cls = _labels_from_label_idx(idx, num_classes=num_classes, uncond_idx=uncond_idx)
    uncond_idx_t = torch.full((bz,), uncond_idx, device=device, dtype=torch.long)
    ae = _labels_from_label_idx(uncond_idx_t, num_classes=num_classes, uncond_idx=uncond_idx)
    return cls, ae


# ═══════════════════════════════════════════════════════════════════════════
# Prologue-fix sampling (teacher-force masked prologue positions)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
@torch._dynamo.disable
def sampling_with_fixed_prologue(ar_model, *, bz, class_label, config,
                                 fixed_prologue_ids, n_fix=None, fix_mask=None):
    """AR sample with selected prologue positions teacher-forced (``fix_mask`` overrides ``n_fix``)."""
    m = ar_model
    z_len = m.z_len
    if z_len <= 0:
        raise ValueError("sampling_with_fixed_prologue requires a Prologue tokenizer (z_len > 0)")
    fixed = fixed_prologue_ids.expand(bz, -1)

    if fix_mask is None:
        n = n_fix if n_fix is not None else z_len
        fix_mask = torch.zeros(z_len, dtype=torch.bool, device=fixed.device)
        fix_mask[:n] = True

    p = _cfg_params(config)
    cfg_val       = 0.0 if class_label is None else p["cfg"]
    cfg_schedule  = p["cfg_schedule"]
    cfg_power     = p["cfg_power"]
    temperature   = p["temperature"]
    topK          = p["topK"]
    topP          = p["topP"]
    cache_kv      = p["cache_kv"]
    sem_cfg       = p["semantic_cfg"]
    sem_cfg_sched = p["semantic_cfg_schedule"]
    sem_cfg_scale = p["semantic_cfg_scale"]
    sem_cfg_pow   = p["semantic_cfg_power"]
    sem_cfg_start = p["semantic_cfg_start"]
    vis_cfg_sched = p["visual_cfg_schedule"]
    vis_cfg_scale = p["visual_cfg_scale"]
    vis_cfg_pow   = p["visual_cfg_power"]
    vis_cfg_start = p["visual_cfg_start"]
    cfg_cont      = p["cfg_continuous"]
    sem_temp      = p["semantic_temperature"]

    use_seg = sem_cfg_sched is not None or vis_cfg_sched is not None
    use_cfg = cfg_val > 0.0 or (sem_cfg is not None and sem_cfg > 0.0)
    if use_seg:
        _ss = sem_cfg_scale if sem_cfg_scale is not None else cfg_val
        _vs = vis_cfg_scale if vis_cfg_scale is not None else cfg_val
        use_cfg = _ss > 0.0 or _vs > 0.0

    uncond_idx = int(m.ar_model.cond_input_dim) - 1
    device = m.bos_emb.weight.device

    if m.conditional_injection == "llamagen":
        cond_bos = m.bos_emb(torch.argmax(class_label, dim=1)).unsqueeze(1)
        uncond_bos = m.bos_emb(
            torch.full((bz,), uncond_idx, device=device, dtype=torch.long)
        ).unsqueeze(1)
        ar_labels = m.uncond_ar_labels.expand(bz, -1).to(device=device)
        uncond_labels = m.uncond_ar_labels.expand(bz, -1).to(device=device)
    else:
        cond_bos = m.bos_emb(
            torch.zeros(bz, device=device, dtype=torch.long)
        ).unsqueeze(1)
        uncond_bos = cond_bos
        ar_labels = class_label
        uncond_labels = m.uncond_ar_labels.expand(bz, -1).to(device=device)

    quant_input = (
        torch.cat([cond_bos, uncond_bos], dim=0) if use_cfg else cond_bos
    )
    ar_labels_2x = (
        torch.cat([ar_labels, uncond_labels], dim=0) if use_cfg else ar_labels
    )
    quant_output = []
    past_kvs = None

    for step in range(m.max_length):
        is_sem = z_len > 0 and step < z_len

        if use_cfg:
            ar_out = m.ar_model(quant_input, ar_labels_2x, cache_kv=cache_kv, past_kvs=past_kvs)
            hidden_all, past_kvs = ar_out if cache_kv else (ar_out, None)
            if m.tied_embedding:
                hidden_all = F.linear(hidden_all[:, -1:], m.semantic_emb.weight)
            logits_all = hidden_all[:, -1]
            logits, uncond_logits = logits_all.chunk(2, dim=0)

            # Scheduled CFG strength c(step), identical to ARModel.sampling.
            if use_seg:
                if is_sem:
                    sc = sem_cfg_sched or "constant"
                    ss = sem_cfg_scale if sem_cfg_scale is not None else cfg_val
                    sp = sem_cfg_pow if sem_cfg_pow is not None else cfg_power
                    s0 = sem_cfg_start
                    st = step / z_len if z_len > 0 else 0.0
                else:
                    sc = vis_cfg_sched or "constant"
                    ss = vis_cfg_scale if vis_cfg_scale is not None else cfg_val
                    sp = vis_cfg_pow if vis_cfg_pow is not None else cfg_power
                    s0 = (
                        (sem_cfg_scale if sem_cfg_scale is not None else cfg_val)
                        if cfg_cont else vis_cfg_start
                    )
                    vl = m.max_length - z_len
                    st = (step - z_len) / vl if vl > 0 else 0.0
                if sc == "constant":
                    c = ss
                elif sc == "linear":
                    c = s0 + (ss - s0) * st
                elif sc == "cosine":
                    c = s0 + (ss - s0) * (1 - math.cos((st ** sp) * math.pi)) * 0.5
                else:
                    raise ValueError(sc)
            elif cfg_schedule == "constant" and is_sem and sem_cfg is not None:
                c = sem_cfg
            elif cfg_schedule == "constant":
                c = cfg_val
            elif cfg_schedule == "linear":
                c = 1.0 * (1 - step / m.max_length) + cfg_val * (step / m.max_length)
            elif cfg_schedule == "cosine":
                c = (1 - math.cos(((step / m.max_length) ** cfg_power) * math.pi)) * 0.5
                c = (cfg_val - 1) * c + 1
            else:
                raise ValueError(cfg_schedule)

            logits = c * logits + (1 - c) * uncond_logits
        else:
            ar_out = m.ar_model(quant_input, ar_labels_2x, cache_kv=cache_kv, past_kvs=past_kvs)
            hidden, past_kvs = ar_out if cache_kv else (ar_out, None)
            if m.tied_embedding:
                hidden = F.linear(hidden[:, -1:], m.semantic_emb.weight)
            logits = hidden[:, -1]

        t = sem_temp if (sem_temp is not None and is_sem) else temperature
        logits = logits / t

        if m.logit_mask is not None:
            logits = logits + m.logit_mask[step]

        if topK is not None and topK > 0.0:
            tl, ti = logits.topk(int(topK), dim=-1)
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(dim=-1, index=ti, src=tl)

        if topP is not None and 0.0 < topP < 1.0:
            sl, si = torch.sort(logits, dim=-1, descending=True)
            ps = sl.softmax(dim=-1).cumsum(dim=-1)
            mask = ps > topP
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sl[mask] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(dim=-1, index=si, src=sl)

        if step < z_len and fix_mask[step]:
            next_idx = fixed[:, step : step + 1]
        else:
            with torch.amp.autocast("cuda", enabled=False):
                next_idx = torch.multinomial(F.softmax(logits.float(), dim=-1), 1)
            next_idx = next_idx.to(dtype=torch.long)

        quant_output.append(next_idx)

        next_emb = m.semantic_emb(next_idx)
        if use_cfg:
            # semantic_drop branch kept for legacy ckpts; default = symmetric uncond.
            if getattr(m, "semantic_drop", False) and is_sem:
                uncond_sem = torch.full_like(next_idx, m.uncond_sem_token_id)
                uncond_emb = m.semantic_emb(uncond_sem)
                next_emb = torch.cat([next_emb, uncond_emb], dim=0)
            else:
                next_emb = torch.cat([next_emb, next_emb], dim=0)

        if not cache_kv:
            quant_input = torch.cat((quant_input, next_emb), dim=1)
        else:
            quant_input = next_emb

    return torch.cat(quant_output, dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# CLI mode implementations
# ═══════════════════════════════════════════════════════════════════════════

def mode_sample(quantizer, dec, ar_model, *, config, class_ids, num_per_class,
                output_dir, device):
    """One row per class, ``num_per_class`` independent samples."""
    num_classes = int(config.num_classes)
    ae_no_label = bool(config.get("ae_no_label", False))

    all_imgs = []
    for cid in class_ids:
        name = IMAGENET_NAMES.get(cid, f"class{cid}")
        print0(f"  Sampling class {cid} ({name}) -> {num_per_class} ...")
        cls_lbl, uncond_lbl = _make_labels(cid, num_per_class, num_classes, device)
        ae_lbl = uncond_lbl if ae_no_label else cls_lbl

        token_ids = sample_tokens(ar_model, bz=num_per_class,
                                  class_label=cls_lbl, config=config)
        imgs = decode_tokens(quantizer, dec, token_ids,
                             config=config, ae_label=ae_lbl)
        all_imgs.append(imgs)

        grid = torchvision.utils.make_grid(imgs, nrow=num_per_class, padding=2)
        out = os.path.join(output_dir, f"class_{cid}_{name}.png")
        save_tensor_image_png_pdf(grid, out)
        print0(f"    -> {out} (+ .pdf)")

    combined = torch.cat(all_imgs, dim=0)
    grid = torchvision.utils.make_grid(combined, nrow=num_per_class, padding=2)
    out = os.path.join(output_dir, "sample_grid.png")
    save_tensor_image_png_pdf(grid, out)
    print0(f"  Combined grid -> {out} (+ .pdf)")


def mode_prologue_fix(quantizer, dec, ar_model, *, config, class_ids,
                      num_resample, num_prologue_sets, output_dir, device):
    """Per class: ``num_prologue_sets`` refs × ``num_resample`` visuals with fixed prologue prefix."""
    prologue = (
        bool(config.get("Prologue", False))
        and not bool(config.get("share_semantic_codebook", False))
    )
    if not prologue:
        raise ValueError(
            "prologue_fix mode requires a Prologue tokenizer "
            "(Prologue=True, share_semantic_codebook=False)"
        )

    num_classes = int(config.num_classes)
    ae_no_label = bool(config.get("ae_no_label", False))
    z_len = ar_model.z_len

    all_class_imgs = []
    for cid in class_ids:
        name = IMAGENET_NAMES.get(cid, f"class{cid}")
        print0(f"  Prologue-fix: class {cid} ({name}), "
               f"{num_prologue_sets} set(s) -> {num_resample} visual resamples ...")

        cls_lbl_1, uncond_lbl_1 = _make_labels(cid, 1, num_classes, device)
        cls_lbl_n, uncond_lbl_n = _make_labels(cid, num_resample, num_classes, device)
        ae_lbl_n = uncond_lbl_n if ae_no_label else cls_lbl_n

        rows = []
        for _ in range(num_prologue_sets):
            token_ids = sample_tokens(ar_model, bz=1,
                                      class_label=cls_lbl_1, config=config)
            prologue_ids = token_ids[:, :z_len]

            resampled = sampling_with_fixed_prologue(
                ar_model, bz=num_resample, class_label=cls_lbl_n,
                config=config, fixed_prologue_ids=prologue_ids,
            )
            imgs = decode_tokens(quantizer, dec, resampled,
                                 config=config, ae_label=ae_lbl_n)
            rows.append(imgs)

        grid_imgs = torch.cat(rows, dim=0)
        all_class_imgs.append(grid_imgs)
        grid = torchvision.utils.make_grid(grid_imgs, nrow=num_resample, padding=2)
        out = os.path.join(output_dir, f"prologue_fix_{cid}_{name}.png")
        save_tensor_image_png_pdf(grid, out)
        print0(f"    -> {out} (+ .pdf)")

    combined = torch.cat(all_class_imgs, dim=0)
    grid = torchvision.utils.make_grid(combined, nrow=num_resample, padding=2)
    out = os.path.join(output_dir, "all_prologue_fix.png")
    save_tensor_image_png_pdf(grid, out)
    print0(f"  Combined grid -> {out} (+ .pdf)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.get("use_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    torch.cuda.manual_seed(seed)

    mode = str(config.get("mode", "sample"))
    output_dir = str(config.get("output_dir", "sample_vis_output"))
    os.makedirs(output_dir, exist_ok=True)

    raw_ids = str(config.get("class_ids", ""))
    class_ids = (
        [int(x.strip()) for x in raw_ids.split(",") if x.strip()]
        if raw_ids else DEFAULT_CLASS_IDS
    )

    print0(f"Mode        : {mode}")
    print0(f"Class IDs   : {class_ids}")
    print0(f"Output dir  : {output_dir}")
    print0(f"Seed        : {seed}")

    quantizer, dec, _, ar_model = load_models(config, device)

    if mode == "sample":
        num_per_class = int(config.get("num_per_class", 8))
        mode_sample(quantizer, dec, ar_model, config=config,
                    class_ids=class_ids, num_per_class=num_per_class,
                    output_dir=output_dir, device=device)
    elif mode == "prologue_fix":
        num_resample = int(config.get("num_resample", 8))
        num_prologue_sets = int(config.get("num_prologue_sets", 1))
        mode_prologue_fix(quantizer, dec, ar_model, config=config,
                          class_ids=class_ids, num_resample=num_resample,
                          num_prologue_sets=num_prologue_sets,
                          output_dir=output_dir, device=device)
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}  (expected 'sample' or 'prologue_fix')"
        )

    print0("Done.")


if __name__ == "__main__":
    main()
