"""Qualitative Grad-CAM figure for ResNet50 and Swin-T on random PlantDoc images (illustration only).

Requires: pip install grad-cam matplotlib
The script follows the procedure used for Figure 1 of the paper (8 random images, seed 0), but the
image order differs from the original notebook, so the exact same 8 images are not guaranteed.

Example:
    python src/gradcam.py --out-dir runs --plantdoc-dir PlantDoc-Dataset \
        --resnet-tag resnet50_s1 --swin-tag swin_t_s1
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from common import REPO_ROOT, arch_from_tag, build_model, collect_plantdoc, eval_transform, load_classes, load_mapping


def load(out_dir, tag, classes, device):
    model, _ = build_model(arch_from_tag(tag), len(classes), pretrained=False)
    model.load_state_dict(torch.load(Path(out_dir) / f"{tag}_best.pt", map_location=device))
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--plantdoc-dir", required=True)
    ap.add_argument("--resnet-tag", default="resnet50_s1")
    ap.add_argument("--swin-tag", default="swin_t_s1")
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--output", default="gradcam.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes()
    mapping = load_mapping(REPO_ROOT / "mappings" / "plantdoc_mapping.csv")
    items = collect_plantdoc(args.plantdoc_dir, mapping, classes)
    random.seed(0)
    sample = random.sample(items, args.n_images)

    resnet, swin = load(args.out_dir, args.resnet_tag, classes, device), load(args.out_dir, args.swin_tag, classes, device)
    nets = {
        "ResNet50": (resnet, [resnet.layer4[-1]], None),
        "Swin-T": (swin, [swin.features[-1][-1].norm1], lambda t: t.permute(0, 3, 1, 2)),
    }

    tf = eval_transform()
    fig, axes = plt.subplots(len(sample), 3, figsize=(9, 3 * len(sample)))
    for r, (path, label) in enumerate(sample):
        image = Image.open(path).convert("RGB").resize((224, 224))
        rgb = np.asarray(image).astype(np.float32) / 255
        x = tf(image).unsqueeze(0).to(device)
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(classes[label][:24], fontsize=7)
        for c, (name, (model, layers, reshape)) in enumerate(nets.items(), start=1):
            pred = model(x).argmax(1).item()
            with GradCAM(model=model, target_layers=layers, reshape_transform=reshape) as cam:
                heat = cam(input_tensor=x, targets=None)[0]
            axes[r, c].imshow(show_cam_on_image(rgb, heat, use_rgb=True))
            axes[r, c].set_title(f"{name}: {classes[pred][:18]}", fontsize=7,
                                 color="green" if pred == label else "red")
        for ax in axes[r]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print("saved", args.output)


if __name__ == "__main__":
    main()
