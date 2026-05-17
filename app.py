"""Prologue: interactive demo for class-conditional image generation.

Two operations are exposed:

  1. *Resample all*: draw a fresh (prologue + visual) sample for the
     selected class. The latest prologue tokens are cached for (2).

  2. *Resample visual only*: keep the cached prologue tokens, redraw only
     the visual tokens. Shows that prologue tokens carry class identity and
     global layout while visual tokens carry texture / fine details.

Loads the released Prologue-L-XL checkpoint (paper gFID = 1.46 with CFG).
Run locally:

    conda activate prologue
    python app.py                  # binds to 0.0.0.0:7860, local only
    python app.py --share          # also expose a temporary gradio.live URL
    python app.py --port 7870      # change the port

Checkpoint resolution order:
  1. Local paths from env vars ``PROLOGUE_TOK_CKPT`` / ``PROLOGUE_AR_CKPT``
     (or their defaults ``ckpts/prologue-l-tokenizer`` /
     ``ckpts/ar-prologue-l-xl``).
  2. If those paths are missing, the demo will ``snapshot_download`` the
     Prologue-L-XL bundle (~16.6 GB) from the HuggingFace Hub repo
     ``$PROLOGUE_HF_REPO`` (default ``Zyriix/prologue``). Pass
     ``--no-bootstrap`` to disable this.

So the same file works locally with pre-downloaded ckpts and on HF Spaces
with a fresh container.
"""

from __future__ import annotations

import argparse
import os
import time

import gradio as gr
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from sample_vis import (
    IMAGENET_NAMES,
    _make_labels,
    decode_tokens,
    load_models,
    sample_tokens,
    sampling_with_fixed_prologue,
)
from utils import seed_everything

# `spaces.GPU` reserves a GPU slice on HF Spaces ZeroGPU; no-op otherwise.
try:
    import spaces                                         # type: ignore
    GPU = spaces.GPU(duration=120)
except ImportError:
    def GPU(fn):                                          # type: ignore
        return fn


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_TOK_CKPT = "ckpts/prologue-l-tokenizer"
DEFAULT_AR_CKPT  = "ckpts/ar-prologue-l-xl"

TOK_CKPT   = os.environ.get("PROLOGUE_TOK_CKPT", DEFAULT_TOK_CKPT)
AR_CKPT    = os.environ.get("PROLOGUE_AR_CKPT",  DEFAULT_AR_CKPT)
HF_REPO_ID = os.environ.get("PROLOGUE_HF_REPO", "Zyriix/prologue")


def _has_ckpt(path: str) -> bool:
    """A ckpt dir is "complete enough" for inference if it has a safetensors."""
    if not os.path.isdir(path):
        return False
    for f in os.listdir(path):
        if f.endswith(".safetensors"):
            return True
    return False


def bootstrap_ckpts(tok_path: str = TOK_CKPT, ar_path: str = AR_CKPT,
                    repo_id: str = HF_REPO_ID) -> None:
    """``snapshot_download`` Prologue-L-XL from HF if missing locally (inference files only)."""
    if _has_ckpt(tok_path) and _has_ckpt(ar_path):
        return

    from huggingface_hub import snapshot_download

    tok_subdir = os.path.basename(tok_path.rstrip("/"))
    ar_subdir  = os.path.basename(ar_path.rstrip("/"))
    local_root = os.path.dirname(os.path.abspath(tok_path)) or "."
    os.makedirs(local_root, exist_ok=True)

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    print(f"[app] checkpoints missing; fetching from hf://{repo_id} -> {local_root}/")
    print(f"[app]   HF token: {'present' if token else 'missing'}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_root,
        token=token,
        allow_patterns=[
            f"{tok_subdir}/model*.safetensors",
            f"{tok_subdir}/custom_checkpoint_*.pkl",
            f"{tok_subdir}/extra_state.pt",
            f"{ar_subdir}/model*.safetensors",
            f"{ar_subdir}/custom_checkpoint_*.pkl",
            f"{ar_subdir}/extra_state.pt",
        ],
        max_workers=8,
    )
    print(f"[app] bootstrap finished: {tok_path} + {ar_path}")

CONFIG_LAYERS = [
    "configs/default.yaml",
    "configs/ar/_defaults.yaml",
    "configs/ar/xlarge.yaml",
    "configs/tokenizer/default.yaml",
    "configs/tokenizer/prologue.yaml",
    "configs/train/ar.yaml",
    "configs/train/eval_ar.yaml",
]

# L-tokenizer overrides (ConvNeXt LPIPS, 4k codebook, asymmetric decoder) + Prologue prior weights.
ARCH_OVERRIDES = {
    "prior_enc_semantic_weight": 3.0,
    "prior_enc_visual_weight":   3.0,
    "ae_no_label":               True,
    "ARModel.ste_ar_embedding":  True,
    "SemanticQuantizer.temperature": 0.1,
    "use_eos":                   True,
    "prior_visual_dropout":      0.5,
    "perceptual_network":        "convnext",
    "codebook_size":             4096,
    "Decoder.dim":               1024,
    "Decoder.layer_num":         24,
    "Decoder.heads":             16,
    "tokenizer_ckpt_path":       TOK_CKPT,
    "resume_ckpt_path":          AR_CKPT,
}

# Two sampling presets (matching paper Tab tab:sota for Prologue-L-XL):
#   paper_cfg  -> gFID = 1.46,  IS = 257.7  (sc:0.7:2.25:0.225)
#   no_cfg     -> gFID = 2.26                (nt2:1.05:0.9)
SAMPLING_PRESETS = {
    "paper_cfg": {
        "semantic_cfg_schedule": "constant",
        "semantic_cfg_scale":    0.7,
        "visual_cfg_schedule":   "cosine",
        "visual_cfg_scale":      2.25,
        "visual_cfg_power":      0.225,
    },
    "no_cfg": {
        "cfg":                   0.0,
        "cfg_schedule":          "constant",
        "semantic_temperature":  1.05,
        "temperature":           0.9,
    },
}


def build_config(preset: str):
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    here = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(here, p) for p in CONFIG_LAYERS]
    conf = OmegaConf.merge(*[OmegaConf.load(p) for p in paths])
    for k, v in ARCH_OVERRIDES.items():
        OmegaConf.update(conf, k, v, merge=True)
    for k, v in SAMPLING_PRESETS[preset].items():
        OmegaConf.update(conf, k, v, merge=True)
    return conf


# ============================================================================
# Model state (loaded once)
# ============================================================================

class ModelBundle:
    """Holds AR model + tokenizer pieces for the active sampling preset."""

    def __init__(self, preset: str = "paper_cfg"):
        self.preset = preset
        self.config = build_config(preset)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.environ.get("PROLOGUE_BOOTSTRAP", "1") == "1":
            bootstrap_ckpts(TOK_CKPT, AR_CKPT)

        print(f"[app] loading Prologue-L-XL ({preset}) on {self.device} ...")
        t0 = time.time()
        self.quantizer, self.dec, self._pq, self.ar_model = load_models(
            self.config, self.device,
        )
        print(f"[app] ready in {time.time() - t0:.1f}s "
              f"(z_len={self.ar_model.z_len}, max_length={self.ar_model.max_length})")

    def switch_preset(self, preset: str):
        """Patch CFG/temperature fields in place; cheap, no model reload."""
        if preset == self.preset:
            return
        for k in SAMPLING_PRESETS["paper_cfg"]:
            OmegaConf.update(self.config, k, None, merge=False)
        for k in SAMPLING_PRESETS["no_cfg"]:
            OmegaConf.update(self.config, k, None, merge=False)
        for k, v in SAMPLING_PRESETS[preset].items():
            OmegaConf.update(self.config, k, v, merge=True)
        self.preset = preset


BUNDLE: ModelBundle | None = None


def _ensure_bundle():
    global BUNDLE
    if BUNDLE is None:
        BUNDLE = ModelBundle()
    return BUNDLE


# ============================================================================
# Inference handlers
# ============================================================================

def _tensor_to_pils(imgs: torch.Tensor):
    """``[B, 3, H, W]`` float in [0, 1] -> list of PIL.Image."""
    arr = (imgs.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def _resolve_seed(seed) -> int:
    """``-1`` / empty -> fresh random seed each call; otherwise use as-is."""
    if seed in (None, ""):
        return int(time.time() * 1000) & 0xFFFFFFFF
    s = int(seed)
    return s if s >= 0 else int(time.time() * 1000) & 0xFFFFFFFF


@GPU
@torch.inference_mode()
def resample_all_handler(class_id: int, num_samples: int, seed: int,
                         preset: str):
    """Draw fresh prologue + visual tokens for the chosen class."""
    bundle = _ensure_bundle()
    bundle.switch_preset(preset)

    class_id = int(class_id) % 1000
    num_samples = int(np.clip(num_samples, 1, 8))
    seed = _resolve_seed(seed)
    seed_everything(seed)
    torch.cuda.manual_seed(seed)

    num_classes = int(bundle.config.num_classes)
    ae_no_label = bool(bundle.config.get("ae_no_label", False))
    cls_lbl, uncond_lbl = _make_labels(class_id, num_samples, num_classes, bundle.device)
    ae_lbl = uncond_lbl if ae_no_label else cls_lbl

    t0 = time.time()
    token_ids = sample_tokens(
        bundle.ar_model, bz=num_samples, class_label=cls_lbl, config=bundle.config,
    )
    imgs = decode_tokens(
        bundle.quantizer, bundle.dec, token_ids, config=bundle.config, ae_label=ae_lbl,
    )
    dt = time.time() - t0

    # Cache the prologue prefix of the first sample for "resample visual".
    z_len = bundle.ar_model.z_len
    prologue_ids = token_ids[:1, :z_len].cpu().tolist()

    name = IMAGENET_NAMES.get(class_id, f"class{class_id}")
    status = (
        f"**Resample all**  •  class **{class_id} – {name}**  •  "
        f"{num_samples} samples  •  preset **{preset}**  •  "
        f"seed {seed}  •  {dt:.1f}s  •  prologue tokens cached"
    )
    return _tensor_to_pils(imgs), status, prologue_ids, class_id


@GPU
@torch.inference_mode()
def resample_visual_handler(class_id: int, num_samples: int, seed: int,
                            preset: str,
                            prologue_state, cached_class_id):
    """Keep cached prologue prefix, redraw the visual suffix."""
    if prologue_state is None or cached_class_id is None:
        # No cached prologue yet; fall back to a fresh full sample.
        pils, _status, p_new, c_new = resample_all_handler(
            class_id, num_samples, seed, preset,
        )
        note = (
            "*(no prologue cached yet, automatically ran "
            "**Resample all** instead; click **Resample visual only** "
            "again to share these prologue tokens.)*"
        )
        return pils, note, p_new, c_new
    bundle = _ensure_bundle()
    bundle.switch_preset(preset)

    class_id = int(class_id) % 1000
    num_samples = int(np.clip(num_samples, 1, 8))
    seed = _resolve_seed(seed)
    seed_everything(seed)
    torch.cuda.manual_seed(seed)

    num_classes = int(bundle.config.num_classes)
    ae_no_label = bool(bundle.config.get("ae_no_label", False))
    cls_lbl, uncond_lbl = _make_labels(class_id, num_samples, num_classes, bundle.device)
    ae_lbl = uncond_lbl if ae_no_label else cls_lbl

    prologue_ids = torch.tensor(prologue_state, device=bundle.device, dtype=torch.long)
    t0 = time.time()
    token_ids = sampling_with_fixed_prologue(
        bundle.ar_model, bz=num_samples, class_label=cls_lbl,
        config=bundle.config, fixed_prologue_ids=prologue_ids,
    )
    imgs = decode_tokens(
        bundle.quantizer, bundle.dec, token_ids, config=bundle.config, ae_label=ae_lbl,
    )
    dt = time.time() - t0

    cached_name = IMAGENET_NAMES.get(int(cached_class_id), f"class{cached_class_id}")
    cur_name = IMAGENET_NAMES.get(class_id, f"class{class_id}")
    cls_note = (
        f"class **{class_id} – {cur_name}**"
        if class_id == int(cached_class_id)
        else (
            f"class **{class_id} – {cur_name}** "
            f"(prologue tokens were sampled for **{cached_class_id} – {cached_name}**)"
        )
    )
    status = (
        f"**Resample visual only**  •  {cls_note}  •  "
        f"{num_samples} samples  •  preset **{preset}**  •  "
        f"seed {seed}  •  {dt:.1f}s"
    )
    return _tensor_to_pils(imgs), status, prologue_state, cached_class_id


# ============================================================================
# UI
# ============================================================================

INTRO = """
# Prologue · interactive demo

**Prologue**: a small set of token positions at the front of an autoregressive
sequence that are trained *only* with cross-entropy loss; the rest of the
sequence is the usual reconstruction-trained visual tokens.

This demo shows the qualitative property that motivates the method: the
**prologue prefix carries class identity and global layout**, while the
**visual suffix carries texture and fine detail**.

1. Pick an ImageNet class and click **Resample all** to draw a fresh sample.
2. Click **Resample visual only** to keep the prologue tokens from step 1 and
   redraw only the visual tokens. Class and layout stay, texture varies.

Backbone: **Prologue-L-XL** (685M AR + L-tokenizer); paper gFID = 1.46 with CFG.
"""

CLASS_CHOICES = [
    (f"{cid}: {name}", cid)
    for cid, name in sorted(IMAGENET_NAMES.items())
]


def build_ui():
    here = os.path.dirname(os.path.abspath(__file__))
    showcase_jpg = os.path.join(here, "assets", "semantic_fix_grid.jpg")
    showcase_initial = [showcase_jpg] if os.path.exists(showcase_jpg) else None

    with gr.Blocks(title="Prologue · Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO)

        prologue_state = gr.State(None)
        class_state    = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                class_picker = gr.Dropdown(
                    choices=CLASS_CHOICES,
                    value=207,
                    label="ImageNet class",
                )
                class_id_box = gr.Number(
                    value=207, precision=0, minimum=0, maximum=999,
                    label="Class ID (0–999)",
                )
                num_samples = gr.Slider(1, 8, value=4, step=1,
                                         label="Number of samples")
                seed = gr.Number(
                    value=-1, precision=0, label="Seed",
                    info="-1 = fresh random seed every click; "
                         "any non-negative integer = deterministic",
                )
                preset = gr.Radio(
                    choices=["paper_cfg", "no_cfg"],
                    value="paper_cfg",
                    label="Sampling preset",
                    info=(
                        "paper_cfg = best CFG schedule (gFID 1.46). "
                        "no_cfg = unconditional temperature (gFID 2.26)."
                    ),
                )
                btn_all = gr.Button(
                    "1. Resample all (new prologue + visual)",
                    variant="primary",
                )
                btn_visual = gr.Button(
                    "2. Resample visual only (fix prologue)",
                )
                status = gr.Markdown(
                    "*Showing the pre-rendered teaser from the paper. "
                    "Click **Resample all** to generate samples for your own class.*"
                )

            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Samples",
                    columns=4, rows=2, height=620,
                    object_fit="contain", show_label=True,
                    show_download_button=True,
                    value=showcase_initial,
                )

        class_picker.change(lambda v: v, [class_picker], [class_id_box])
        btn_all.click(
            resample_all_handler,
            inputs=[class_id_box, num_samples, seed, preset],
            outputs=[gallery, status, prologue_state, class_state],
        )
        btn_visual.click(
            resample_visual_handler,
            inputs=[class_id_box, num_samples, seed, preset,
                    prologue_state, class_state],
            outputs=[gallery, status, prologue_state, class_state],
        )

    return demo


# ============================================================================
# Entrypoint
# ============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--share", action="store_true",
        help="Expose a temporary public gradio.live URL (lasts 72h).",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", 7860)),
        help="Local port to bind (default: 7860, or $PORT if set).",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Local host to bind (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Don't auto-download ckpts from HF when local paths are missing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.no_bootstrap:
        os.environ["PROLOGUE_BOOTSTRAP"] = "0"
    _ensure_bundle()
    ui = build_ui()

    # HF Spaces: let the Space wrapper own networking; locally honor --host/--port.
    if os.environ.get("SPACE_ID"):
        launch_kwargs = {"share": True}
    else:
        launch_kwargs = {
            "share":       args.share,
            "server_name": args.host,
            "server_port": args.port,
        }

    ui.queue(default_concurrency_limit=1).launch(show_api=False, **launch_kwargs)
