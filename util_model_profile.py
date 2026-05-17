import copy
import torch
from models import Linear, Attention
from utils import print0

try:
    from thop import profile
    from thop.vision.basic_hooks import count_linear
except ImportError:
    profile = None
    count_linear = None


def count_attention_ops(m, x, y):
    inp = x[0]
    B, L, C = inp.shape
    linear_macs = 4 * B * L * C * C
    attn_macs = 2 * B * L * L * C
    rope_macs = 6 * B * L * C if m.rope else 0
    m.total_ops += torch.DoubleTensor([linear_macs + attn_macs + rope_macs])


def print_model_info(model, inputs, device, name="Model"):
    if model is None:
        print0(f"[{name}] Model is None, skipping stats calculation.")
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    gflops_str = "N/A (thop not installed)"
    if profile is not None:
        custom_ops = {}
        if count_linear is not None:
            custom_ops[Linear] = count_linear
        custom_ops[Attention] = count_attention_ops

        model_copy = copy.deepcopy(model).to(device).eval()
        inputs_dev = tuple(i.to(device) if isinstance(i, torch.Tensor) else i for i in inputs)
        try:
            macs, _ = profile(model_copy, inputs=inputs_dev, custom_ops=custom_ops, verbose=False)
            gflops_str = f"{macs * 2 / 1e9:.2f}"
        except Exception as e:
            gflops_str = f"N/A (thop failed: {e})"
        del model_copy, inputs_dev
        torch.cuda.empty_cache()

    print0(f"[{name}] Params: {total_params/1e6:.2f}M (trainable {trainable_params/1e6:.2f}M), GFLOPs: {gflops_str}")


def print_model_stats(config, device, enc, dec, ar_model):
    x_len = int(config.x_len)
    patch_dim = int(config.patch_dim)
    num_classes = int(config.num_classes)
    z_len = int(config.z_len)
    z_dim = int(config.z_dim)
    codebook_size = int(config.codebook_size)
    prologue = getattr(config, 'Prologue', False)

    x_dummy = torch.randn(1, x_len, patch_dim)
    label_dummy = torch.zeros(1, num_classes)
    print_model_info(enc, (x_dummy, label_dummy), device, "Encoder")

    dec_seq_len = x_len if prologue else z_len
    z_dummy = torch.randn(1, dec_seq_len, z_dim)
    print_model_info(dec, (z_dummy, label_dummy), device, "Decoder")

    ar_seq_len = z_len + x_len if prologue else z_len
    idx_dummy = torch.randint(0, codebook_size, (1, ar_seq_len))
    print_model_info(ar_model, (idx_dummy, label_dummy), device, "ARModel")
