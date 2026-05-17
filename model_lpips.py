"""Perceptual loss: supports VGG16 LPIPS and ConvNeXt-S logit-level loss.

VGG LPIPS: https://github.com/richzhang/PerceptualSimilarity/tree/master/models
ConvNeXt-S: adapted from https://github.com/bytedance/1d-tokenizer (TiTok/AliTok)
"""

import os, hashlib
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import namedtuple

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

def build_perceptual_loss(network="vgg"):
    """Frozen perceptual loss; ``network`` ∈ ``{"vgg", "convnext", "both"}``."""
    if network == "vgg":
        return LPIPS()
    elif network == "convnext":
        return ConvNeXtPerceptualLoss()
    elif network == "both":
        return BothPerceptualLoss()
    else:
        raise ValueError(f"Unknown perceptual network: {network}")


# ---------------------------------------------------------------------------
# ConvNeXt-S perceptual loss  (logit-level MSE, same as TiTok / AliTok)
# ---------------------------------------------------------------------------

class BothPerceptualLoss(nn.Module):
    """LPIPS-VGG16 + ConvNeXt-S used together. Returns a named tuple so
    the training loop can log and weight them separately."""

    def __init__(self):
        super().__init__()
        self.lpips = LPIPS()
        self.convnext = ConvNeXtPerceptualLoss()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input, target):
        lpips_val = self.lpips(input, target)
        convnext_val = self.convnext(input, target)
        return lpips_val, convnext_val


class ConvNeXtPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnext = models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        ).eval()
        self.register_buffer(
            "imagenet_mean",
            torch.Tensor(_IMAGENET_MEAN)[None, :, None, None],
        )
        self.register_buffer(
            "imagenet_std",
            torch.Tensor(_IMAGENET_STD)[None, :, None, None],
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input, target):
        """input/target: [-1, 1] range (same convention as LPIPS)."""
        self.eval()
        input = input * 0.5 + 0.5  # [-1, 1] → [0, 1]
        target = target * 0.5 + 0.5
        input = F.interpolate(input, size=224, mode="bilinear", align_corners=False, antialias=True)
        target = F.interpolate(target, size=224, mode="bilinear", align_corners=False, antialias=True)
        pred_input = self.convnext((input - self.imagenet_mean) / self.imagenet_std)
        pred_target = self.convnext((target - self.imagenet_mean) / self.imagenet_std)
        loss = F.mse_loss(pred_input, pred_target, reduction="none").mean(-1)
        return loss


# ---------------------------------------------------------------------------
# VGG16 LPIPS  (original implementation)
# ---------------------------------------------------------------------------

class LPIPS(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)