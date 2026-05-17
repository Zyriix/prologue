import json
import os
import sys
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed,DataLoaderConfiguration
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Encoder, Decoder
from dataset import ImageFolderDataset
from utils import img_uint8_to_norm, make_worker_init_fn, patchify, unpatchify, print0, _unwrap
from torchmetrics.functional.image import peak_signal_noise_ratio as calc_psnr




def _idx_dtype(codebook_size: int):
    if codebook_size <= 256:
        return np.uint8
    elif codebook_size <= 65536:
        return np.uint16
    else:
        return np.uint32


def _compute_divisible_batch_size(dataset_size: int, num_processes: int, target_batch_size: int) -> int:
    """Largest ``bs <= target_batch_size`` such that ``(dataset_size // num_processes) % bs == 0``."""
    samples_per_rank = dataset_size // num_processes
    for bs in range(target_batch_size, 0, -1):
        if samples_per_rank % bs == 0:
            return bs
    return 1


def _default_out_dir(tokenizer_ckpt_path: str) -> str:
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


def _load_encoder_from_accelerate_ckpt(accelerator, config, ckpt_dir: str, use_ema: bool = False):
    enc = Encoder(config).to(accelerator.device)
    
    fname = "model_2.safetensors" if use_ema else "model.safetensors"
    ckpt_path = os.path.join(ckpt_dir, fname)
    
    kind = "EMA" if use_ema else "non-EMA"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{kind} encoder weights not found: {ckpt_path}")
    
    sd = load_file(ckpt_path)
    
    enc.load_state_dict(sd, strict=True)
    
    return enc.eval().requires_grad_(False)


def _load_decoder_from_accelerate_ckpt(accelerator, config, ckpt_dir: str, use_ema: bool = False):
    """Load the decoder from an accelerate-format checkpoint directory."""
    dec = Decoder(config).to(accelerator.device)
    
    fname = "model_3.safetensors" if use_ema else "model_1.safetensors"
    ckpt_path = os.path.join(ckpt_dir, fname)
    
    kind = "EMA" if use_ema else "non-EMA"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{kind} decoder weights not found: {ckpt_path}")
    
    sd = load_file(ckpt_path)
    
    dec.load_state_dict(sd, strict=True)
    
    return dec.eval().requires_grad_(False)


def _codes_from_indices(quantizer, indices: torch.Tensor, labels: torch.Tensor):
    """Look up codes from indices; tolerates quantizer interfaces with or without a labels arg."""
    try:
        return quantizer.get_codes_w_indices(indices, labels)
    except TypeError:
        return quantizer.get_codes_w_indices(indices)


@torch.no_grad()
@torch._dynamo.disable
def _sanity_check_reconstruction(config, enc, dec, accelerator, is_ten_crop, sanity_check_max_size=64):
    """One-batch reconstruction PSNR check on a tiny standalone split."""
    if accelerator.is_main_process:
        print0("\n" + "="*60)
        print0("Running Sanity Check: Reconstruction Test")
        print0(f"Using max_size={sanity_check_max_size}")
        print0("="*60)

    # Build a small standalone dataset just for the sanity check
    sanity_dataset = ImageFolderDataset(
        path=config['data_dir'],
        resolution=config.image_size,
        use_label=config.use_label,
        max_size=sanity_check_max_size,
        xflip=False,
        crop_type=config.crop_type,
        deterministic_crop=True,
        crop_seed=int(config.seed) if config.seed is not None else 0,
    )

    sanity_loader = DataLoader(
        sanity_dataset,
        batch_size=min(sanity_check_max_size, len(sanity_dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(sanity_loader))
    imgs, labels = batch
    
    if is_ten_crop:
        # (B, 10, C, H, W) -> (B*10, C, H, W), sample-major within crops.
        assert imgs.ndim == 5 and imgs.shape[1] == 10
        batch_size = imgs.shape[0]
        imgs = imgs.contiguous().view(-1, *imgs.shape[2:])
        print0(f"Ten crop detected: reshaped from ({batch_size}, 10, ...) to ({imgs.shape[0]}, ...)")
    else:
        batch_size = imgs.shape[0]
    
    imgs = imgs.to(accelerator.device)
    
    # Normalize and patchify
    imgs_norm = img_uint8_to_norm(imgs)
    x = patchify(imgs_norm, config.patch_size)
    
    # Prepare labels (same as _encode_once)
    num_classes = config.num_classes
    use_label = config.use_label
    _ae_no_label = bool(getattr(config, 'ae_no_label', False))
    if labels is None or len(labels) == 0 or not use_label or _ae_no_label:
        enc_labels = torch.nn.functional.one_hot(
            torch.full((batch_size,), num_classes - 1, device=accelerator.device, dtype=torch.long),
            num_classes=num_classes
        ).float()
    else:
        labels = labels.to(accelerator.device)
        enc_labels = labels
    
    # For ten_crop, repeat labels 10 times: (B, num_classes) -> (B*10, num_classes)
    if is_ten_crop:
        enc_labels = enc_labels.unsqueeze(1).repeat(1, 10, 1).reshape(-1, enc_labels.shape[-1])
    
    # Extra dim only when ae_no_label=False (matches train_tokenizer).
    if not _ae_no_label:
        enc_labels = torch.cat([enc_labels, torch.zeros((enc_labels.shape[0], 1), device=accelerator.device, dtype=enc_labels.dtype)], dim=-1)
    
    enc_raw = _unwrap(enc)
    idx = enc_raw.encode_idx(x, enc_labels)
    
    quant = enc_raw.get_visual_codes(idx, enc_labels)
    
    # 3. decoder -> reconstruction
    recon_patches = dec(quant, enc_labels)
    recon_img = unpatchify(recon_patches, config.image_size, config.patch_size)

    # Calculate PSNR
    imgs_norm = imgs_norm.to(dtype=torch.float32)
    recon_img = recon_img.to(dtype=torch.float32)
    img01 = ((imgs_norm + 1.0) * 0.5).clamp(0.0, 1.0)
    recon_img01 = ((recon_img + 1.0) * 0.5).clamp(0.0, 1.0)
    psnr_val = calc_psnr(recon_img01, img01, data_range=1.0).item()

    if accelerator.is_main_process:
        print0(f"  Batch size: {imgs.shape[0]}")
        print0(f"  Image shape: {imgs.shape[1:]}")
        print0(f"  Reconstruction PSNR: {psnr_val:.2f} dB")
        print0("="*60 + "\n")
    
    return psnr_val


@torch.no_grad()
def _encode_once(loader, config, enc, accelerator, dataset_size: int, out_dir: str, is_ten_crop: bool):
    """Stream token rows (+ label) to a per-rank memmap on disk."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(out_dir, f".tokens.rank{accelerator.process_index}.npy")

    enc_raw = _unwrap(enc)
    token_len = enc_raw.total_token_len
    num_classes = config.num_classes
    use_label = config.use_label
    ae_no_label = bool(getattr(config, 'ae_no_label', False))
    idx_dtype = _idx_dtype(config.codebook_size)

    row_len = (10 * token_len + 1) if is_ten_crop else (token_len + 1)

    # Allocate a memory-mapped array large enough for one rank's share plus a small slack.
    n_samples = dataset_size // accelerator.num_processes + config.batch_size
    memmap_array = np.lib.format.open_memmap(
        output_path,
        mode='w+',
        dtype=idx_dtype,
        shape=(n_samples, row_len)
    )

    num_written = 0
    i=0
    for batch in tqdm(
        loader,
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
        desc=f"Encoding [rank {accelerator.process_index}]",
        file=sys.stdout
    ):
        i+=1
        imgs, labels = batch
        
        if is_ten_crop:
            assert imgs.ndim == 5 and imgs.shape[1] == 10
            batch_size = imgs.shape[0]
            imgs = imgs.contiguous().view(-1, *imgs.shape[2:])  # (B*10, C, H, W)
        else:
            batch_size = imgs.shape[0]
        
        imgs = imgs.to(accelerator.device)
        
        # Normalize images
        imgs_norm = img_uint8_to_norm(imgs)
        
        # Patchify first (before encoding)
        x = patchify(imgs_norm, config.patch_size)
        
        has_real_label = False
        uncond_enc_labels = torch.nn.functional.one_hot(
            torch.full((batch_size,), num_classes - 1, device=accelerator.device, dtype=torch.long),
            num_classes=num_classes
        ).float()
        if labels is None or len(labels) == 0 or not use_label:
            enc_labels = uncond_enc_labels
        else:
            labels = labels.to(accelerator.device)
            has_real_label = True
            enc_labels = uncond_enc_labels if ae_no_label else labels
        
        # For ten_crop, repeat labels 10 times: (B, num_classes) -> (B*10, num_classes)
        if is_ten_crop:
            enc_labels = enc_labels.unsqueeze(1).repeat(1, 10, 1).reshape(-1, enc_labels.shape[-1])
        
        # Extra dim only when ae_no_label=False (matches train_tokenizer).
        if not ae_no_label:
            enc_labels = torch.cat([enc_labels, torch.zeros((enc_labels.shape[0], 1), device=accelerator.device, dtype=enc_labels.dtype)], dim=-1)
        
        idx = enc_raw.encode_idx(x, enc_labels)
        
        # idx shape: (B*10, token_len) for ten_crop, (B, token_len) otherwise
        idx_np = idx.detach().contiguous().cpu().numpy().astype(idx_dtype)
        
        # Saved label is per-image (B), not B*10 under ten_crop.
        if has_real_label:
            label_idx = torch.argmax(labels, dim=-1)
            label_np = label_idx.detach().cpu().numpy().astype(idx_dtype)
        else:
            label_np = np.full((batch_size,), num_classes - 1, dtype=idx_dtype)
        
        # Write to the memmap (streamed to disk)
        if is_ten_crop:
            idx_np = idx_np.reshape(batch_size, 10, token_len)
            idx_np_flat = idx_np.reshape(batch_size, 10 * token_len)
            memmap_array[num_written:num_written+batch_size, :10*token_len] = idx_np_flat
            memmap_array[num_written:num_written+batch_size, 10*token_len] = label_np
            num_written += batch_size
        else:
            memmap_array[num_written:num_written+batch_size, :token_len] = idx_np
            memmap_array[num_written:num_written+batch_size, token_len] = label_np
            num_written += batch_size

        memmap_array.flush()

    # Final flush + release
    memmap_array.flush()
    accelerator.wait_for_everyone()
    del memmap_array
    print0("finished")
    
    return num_written


def pretokenize(config):
    """Entry point: pretokenize the configured dataset(s) and save .npz caches."""
    if config.seed is not None:
        set_seed(config.seed)

    accelerator = Accelerator( mixed_precision="no", dataloader_config=DataLoaderConfiguration(even_batches=False))

    dl_generator = torch.Generator()
    if config.seed is not None:
        dl_generator.manual_seed(config.seed)

    worker_init = make_worker_init_fn(config.seed) if config.seed is not None else None

    # Train set
    train_dataset = ImageFolderDataset(
        path=config['data_dir'],
        resolution=config.image_size,
        use_label=config.use_label,
        max_size=None,
        xflip=config.xflip,
        crop_type=config.crop_type,
        deterministic_crop=bool(getattr(config, "deterministic_crop", False)),
        crop_seed=int(config.seed) if config.seed is not None else 0,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size // accelerator.num_processes,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        worker_init_fn=worker_init,
        generator=dl_generator,
    )
    
    # Eval set (if any)
    eval_loader = None
    eval_dataset = None
    if config.eval_data_dir is not None:
        eval_dataset = ImageFolderDataset(
            path=config['eval_data_dir'],
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
            batch_size=config.eval_batch_size//accelerator.num_processes,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
            worker_init_fn=worker_init,
            generator=dl_generator,
        )
    
    # Load encoder
    ckpt_dir = config.tokenizer_ckpt_path
    use_ema = config.get("ema_reconstruction", True)
    enc = _load_encoder_from_accelerate_ckpt(accelerator, config, ckpt_dir, use_ema=use_ema)

    # Load decoder (only if we run a sanity check)
    dec = None
    do_sanity_check = config.get("sanity_check", True)
    if do_sanity_check:
        dec = _load_decoder_from_accelerate_ckpt(accelerator, config, ckpt_dir, use_ema=use_ema)
    
    # Prepare with accelerator
    all_components = [train_loader, enc, eval_loader, dec]
    prepared_components = iter(accelerator.prepare(*filter(lambda x: x is not None, all_components)))
    train_loader = next(prepared_components) if train_loader is not None else None
    enc = next(prepared_components) if enc is not None else None
    eval_loader = next(prepared_components) if eval_loader is not None else None
    dec = next(prepared_components) if dec is not None else None
    
    # Compilation
    if config.get("torch_compile", True):
        enc = torch.compile(enc)
        if dec is not None:
            dec = torch.compile(dec)
    
    # Determine crop types for train / eval splits
    is_ten_crop_train = (config.crop_type == 'ten_crop')
    is_ten_crop_eval = (config.eval_crop_type == 'ten_crop') if config.eval_data_dir is not None else False

    if do_sanity_check and dec is not None:
        sanity_check_max_size = config.get("sanity_check_max_size", 64)
        _sanity_check_reconstruction(config, enc, dec, accelerator, is_ten_crop_train, sanity_check_max_size=sanity_check_max_size)

    out_dir = config.get("pretoken_dir", None)
    if out_dir is None:
        out_dir = _default_out_dir(config.tokenizer_ckpt_path)

    # Encode the training set
    if accelerator.is_main_process:
        print0(f"Pretokenizing train set to: {out_dir}")
        print0(f"Train crop_type: {config.crop_type}, is_ten_crop: {is_ten_crop_train}")

    train_num_written = _encode_once(
        train_loader, config, enc, accelerator,
        len(train_dataset), os.path.join(out_dir, "train"), is_ten_crop_train
    )

    # Gather actual row counts written by each rank
    train_counts = accelerator.gather(torch.tensor([train_num_written], device=accelerator.device))
    if accelerator.is_main_process:
        train_counts = train_counts.cpu().numpy().tolist()
    print0("gather train_counts:",train_counts)
    accelerator.wait_for_everyone()

    # Encode the eval set
    eval_counts = None
    if eval_loader is not None:
        if accelerator.is_main_process:
            print0(f"Pretokenizing eval set to: {out_dir}")
            print0(f"Eval crop_type: {config.eval_crop_type}, is_ten_crop: {is_ten_crop_eval}")

        eval_num_written = _encode_once(
            eval_loader, config, enc, accelerator,
            len(eval_dataset), os.path.join(out_dir, "eval"), is_ten_crop_eval
        )

        eval_counts = accelerator.gather(torch.tensor([eval_num_written], device=accelerator.device))
        if accelerator.is_main_process:
            eval_counts = eval_counts.cpu().numpy().tolist()
        print0("gather eval:",eval_counts)
        accelerator.wait_for_everyone()

    # Real dataset sizes (used to trim distributed padding)
    train_dataset_size = len(train_dataset)
    eval_dataset_size = len(eval_dataset) if eval_dataset is not None else None

    def get_filename_with_crop_suffix(base_name: str, crop_type: str) -> str:
        """Append a crop_type suffix before the .npz extension."""
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
    
    train_crop_type = config.get('crop_type', None)
    eval_crop_type = config.get('eval_crop_type', None)
    
    train_filename = get_filename_with_crop_suffix("train_pretoken.npz", train_crop_type)
    eval_filename = get_filename_with_crop_suffix("eval_pretoken.npz", eval_crop_type)
    
    # Main process merges per-rank .npy shards into a single .npz per split
    if accelerator.is_main_process:
        print0("Merging and saving tokenized data...")

        for split, split_name, counts, dataset_size in [
            ("train", train_filename, train_counts, train_dataset_size),
            ("eval", eval_filename, eval_counts, eval_dataset_size)
        ]:
            if counts is None:
                continue

            split_dir = os.path.join(out_dir, split)
            if not os.path.exists(split_dir):
                continue

            all_data = []
            for rank in range(accelerator.num_processes):
                npy_path = os.path.join(split_dir, f".tokens.rank{rank}.npy")

                if os.path.exists(npy_path):
                    data = np.load(npy_path)

                    actual_count = int(counts[rank])
                    if actual_count < data.shape[0]:
                        data = data[:actual_count]

                    all_data.append(data)
                    print0(f"Loaded {npy_path}: {data.shape[0]} samples")
                    print0(f"Use only {actual_count} samples")

            if len(all_data) > 0:
                merged_data = np.concatenate(all_data, axis=0)

                if dataset_size is not None and merged_data.shape[0] != dataset_size:
                    print0(f"  WARNING: Sample count mismatch! Got {merged_data.shape[0]}, expected {dataset_size}.")
                    print0(f"     This may indicate a distributed padding issue. Consider adjusting batch_size.")

                output_path = os.path.join(out_dir, split_name)
                np.savez_compressed(output_path, data=merged_data)
                print0(f"Saved {split} data: {output_path} ({merged_data.shape[0]} samples)")

                for rank in range(accelerator.num_processes):
                    npy_path = os.path.join(split_dir, f".tokens.rank{rank}.npy")
                    if os.path.exists(npy_path):
                        os.remove(npy_path)
                        print0(f"  Removed temporary file: {npy_path}")

                try:
                    os.rmdir(split_dir)
                    print0(f"  Removed temporary directory: {split_dir}")
                except OSError:
                    pass
    
    accelerator.wait_for_everyone()
    print0("Pretokenization complete!")
    accelerator.end_training()



if __name__ == "__main__":
    from utils import load_config
    pretokenize(load_config())
