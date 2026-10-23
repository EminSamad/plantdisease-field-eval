"""Fine-tune an ImageNet-pretrained model on PlantVillage (two-stage protocol).

Example:
    python src/train.py --arch swin_t --seed 1 \
        --data-dir "data/plantvillage dataset/color" --out-dir runs
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from common import ARCHITECTURES, build_model, load_classes, make_plantvillage_splits


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, use_amp=False):
    train = optimizer is not None
    model.train(train)
    ys, ps = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            ys += y.cpu().tolist()
            ps += out.argmax(1).cpu().tolist()
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=ARCHITECTURES)
    ap.add_argument("--seed", type=int, required=True, help="training seed (data split is fixed separately)")
    ap.add_argument("--data-dir", required=True, help="PlantVillage 'color' folder (38 class sub-folders)")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--head-epochs", type=int, default=3, help="epochs with only the new head trained")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    classes = load_classes()
    tag = f"{args.arch}_s{args.seed}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_path, ckpt_path, result_path = out / f"{tag}_best.pt", out / f"ckpt_{tag}.pt", out / f"results_{tag}.json"

    train_ds, val_ds, test_ds = make_plantvillage_splits(args.data_dir, classes)
    loader = lambda ds, shuffle=False: DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds), loader(test_ds)
    print(f"[{tag}] train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)} | device {device}")

    model, head = build_model(args.arch, len(classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)

    start, best_f1 = 0, 0.0
    if ckpt_path.exists():  # resume an interrupted run
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        start, best_f1 = ck["epoch"], ck["best"]
        print(f"[{tag}] resuming from epoch {start}/{args.epochs}")

    phase, optimizer = None, None
    for ep in range(start, args.epochs):
        new_phase = "head" if ep < args.head_epochs else "full"
        if new_phase != phase:
            phase = new_phase
            if phase == "head":
                for p in model.parameters():
                    p.requires_grad = False
                for p in head.parameters():
                    p.requires_grad = True
                optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
            else:
                for p in model.parameters():
                    p.requires_grad = True
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        tr = run_epoch(model, train_dl, device, criterion, optimizer, scaler, use_amp)
        va = run_epoch(model, val_dl, device, criterion, use_amp=use_amp)
        print(f"[{tag}|{phase}] ep{ep + 1}/{args.epochs} train acc {tr[0]:.4f} | val acc {va[0]:.4f} F1 {va[1]:.4f}")
        if va[1] > best_f1:
            best_f1 = va[1]
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "epoch": ep + 1, "best": best_f1}, ckpt_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    te = run_epoch(model, test_dl, device, criterion, use_amp=use_amp)
    result = {"tag": tag, "arch": args.arch, "seed": args.seed,
              "pv_test_acc": te[0], "pv_test_f1": te[1], "best_val_f1": best_f1}
    result_path.write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
