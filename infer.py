"""Standalone MASOR 2.2 checkpoint loader + single-image inference.

Builds the MASOR network, loads a released fold checkpoint, and writes a
grade overlay for one H&E image. The model definition is extracted verbatim
from the training notebook so the released state_dict loads without it.

Classes (per-pixel): 0 background, 1 normal, 2 low-grade, 3 high-grade,
4 adenocarcinoma. Foreground comes from the binary head (thresholded);
grade comes from the nearest ordinal prototype inside the foreground.

NOTE: inference from the released checkpoint has not yet been independently
re-validated outside the training notebook. Treat output as a starting point
for pathologist review, never as ground truth.

Usage:
    python infer.py image.png --weights masor22_full_case_s42_fold_4_final.pt --out overlay.png
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image

# --- hyperparameters, must match the checkpoint's run_config ---
BACKBONE = "convnext_tiny.in12k_ft_in1k"
IMAGE_SIZE = 224
FPN_DIM = 128
PROJ_DIM = 64
N_GRADES = 4
PROTO_TEMP = 0.2
PROTO_MIN_GAP = 0.1
PROTO_RESIDUAL_SCALE = 0.05
SEM_CLASS_NAMES = ["background", "normal_gland", "low_grade", "high_grade", "adk_invasion"]
# overlay colours (RGB) indexed by final class id
PALETTE = np.array([[0, 0, 0], [60, 180, 75], [255, 225, 25], [245, 130, 48], [230, 25, 75]], dtype=np.uint8)


class FPNDecoder(nn.Module):
    def __init__(self, channels, out_dim=FPN_DIM):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_dim, 1) for c in channels])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
                          nn.GroupNorm(8, out_dim), nn.GELU())
            for _ in channels])

    def forward(self, feats):
        xs = [lat(f) for lat, f in zip(self.lateral, feats)]
        for i in range(len(xs) - 1, 0, -1):
            xs[i - 1] = xs[i - 1] + F.interpolate(xs[i], size=xs[i - 1].shape[-2:], mode="bilinear", align_corners=False)
        xs = [sm(x) for sm, x in zip(self.smooth, xs)]
        return xs[0]


class MASORNet(nn.Module):
    def __init__(self, backbone=BACKBONE, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        channels = self.encoder.feature_info.channels()
        self.decoder = FPNDecoder(channels, FPN_DIM)
        self.binary_head = nn.Conv2d(FPN_DIM, 1, 1)
        self.projector = nn.Sequential(nn.Conv2d(FPN_DIM, PROJ_DIM, 1, bias=False), nn.GroupNorm(8, PROJ_DIM), nn.GELU())
        self.ordinal_axis = nn.Parameter(torch.randn(PROJ_DIM) * 0.02)
        self.gap_raw = nn.Parameter(torch.zeros(N_GRADES - 1))
        self.proto_residual = nn.Parameter(torch.zeros(N_GRADES, PROJ_DIM))
        self.grade_bias = nn.Parameter(torch.zeros(N_GRADES))

    def prototypes(self):
        axis = F.normalize(self.ordinal_axis, dim=0)
        gaps = F.softplus(self.gap_raw) + PROTO_MIN_GAP
        centers = torch.cat([torch.zeros(1, device=gaps.device), torch.cumsum(gaps, dim=0)])
        centers = centers - centers.mean()
        protos = centers[:, None] * axis[None, :] + PROTO_RESIDUAL_SCALE * self.proto_residual
        return protos

    def forward(self, x):
        f = self.decoder(self.encoder(x))
        bin_logits = self.binary_head(f)
        z = F.normalize(self.projector(f), dim=1)
        protos = self.prototypes()
        d = ((z[:, None] - protos[None, :, :, None, None]) ** 2).sum(dim=2)
        grade_logits = -d / PROTO_TEMP + self.grade_bias.view(1, -1, 1, 1)
        return bin_logits, grade_logits


def load_model(weights, device="cpu"):
    ckpt = torch.load(weights, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net = MASORNet(pretrained=False).to(device).eval()
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={list(missing)} unexpected={list(unexpected)}")
    thr = ckpt.get("bin_threshold", 0.5) if isinstance(ckpt, dict) else 0.5
    return net, float(thr)


@torch.no_grad()
def predict(net, thr, image_path, device="cpu"):
    cfg = timm.data.resolve_model_data_config(net.encoder)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    rgb = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(rgb).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)[None]
    x = ((x - mean) / std).to(device)
    bin_logits, grade_logits = net(x)
    bin_logits = F.interpolate(bin_logits, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    grade_logits = F.interpolate(grade_logits, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    fg = (torch.sigmoid(bin_logits)[0, 0] > thr).cpu().numpy()
    grade = grade_logits[0].argmax(0).cpu().numpy()  # 0..3
    cls = np.where(fg, grade + 1, 0).astype(np.uint8)  # 0 bg, 1..4
    return np.asarray(rgb), cls


def save_overlay(rgb, cls, out):
    colour = PALETTE[cls]
    blend = (0.55 * rgb + 0.45 * colour).astype(np.uint8)
    blend[cls == 0] = rgb[cls == 0]
    Image.fromarray(blend).save(out)
    present = sorted(set(int(c) for c in np.unique(cls)))
    print("classes present:", [SEM_CLASS_NAMES[c] for c in present])
    print("saved:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    net, thr = load_model(a.weights, a.device)
    rgb, cls = predict(net, thr, a.image, a.device)
    save_overlay(rgb, cls, a.out)


if __name__ == "__main__":
    main()
