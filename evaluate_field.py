"""Evaluate trained PlantVillage models on PlantDoc and/or PlantWild without any adaptation.

Example:
    python src/evaluate_field.py --out-dir runs \
        --plantdoc-dir PlantDoc-Dataset --plantwild-dir plantwild/plantwild
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from common import (ARCHITECTURES, FieldImageDataset, arch_from_tag, build_model, collect_plantdoc,
                    collect_plantwild, eval_transform, load_classes, load_mapping, predict, REPO_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs", help="folder with <tag>_best.pt files from train.py")
    ap.add_argument("--plantdoc-dir", help="PlantDoc classification repository folder (has train/ and test/)")
    ap.add_argument("--plantwild-dir", help="PlantWild v1 folder that contains images/")
    ap.add_argument("--runs", nargs="*", help="run tags, e.g. resnet50_s1 swin_t_s2 (default: all found)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    if not (args.plantdoc_dir or args.plantwild_dir):
        ap.error("give at least one of --plantdoc-dir / --plantwild-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes()
    out = Path(args.out_dir)
    tags = args.runs or sorted(p.name[:-len("_best.pt")] for p in out.glob("*_best.pt"))
    if not tags:
        raise SystemExit(f"no *_best.pt files in {out}")

    datasets_to_run = []
    if args.plantdoc_dir:
        mapping = load_mapping(REPO_ROOT / "mappings" / "plantdoc_mapping.csv")
        datasets_to_run.append(("plantdoc", collect_plantdoc(args.plantdoc_dir, mapping, classes), mapping))
    if args.plantwild_dir:
        mapping = load_mapping(REPO_ROOT / "mappings" / "plantwild_mapping.csv")
        datasets_to_run.append(("plantwild", collect_plantwild(args.plantwild_dir, mapping, classes), mapping))

    for name, items, mapping in datasets_to_run:
        restricted_idx = sorted({classes.index(v) for v in mapping.values()})
        loader = DataLoader(FieldImageDataset(items, eval_transform()),
                            batch_size=args.batch_size, num_workers=args.num_workers)
        print(f"{name}: {len(items)} images, {len(restricted_idx)} shared classes")
        for tag in tags:
            arch = arch_from_tag(tag)
            if arch not in ARCHITECTURES:
                print(f"skipping {tag}: unknown architecture")
                continue
            model, _ = build_model(arch, len(classes), pretrained=False)
            model.load_state_dict(torch.load(out / f"{tag}_best.pt", map_location=device))
            model = model.to(device)
            y, p_all, p_res = predict(model, loader, device, restricted_idx)
            labels = sorted(set(y.tolist()))
            np.savez(out / f"preds_{name}_{tag}.npz", y=y, p_all=p_all, p_restricted=p_res)
            print(f"  {tag:22s} restricted acc {accuracy_score(y, p_res):.4f} "
                  f"F1 {f1_score(y, p_res, labels=labels, average='macro'):.4f} | "
                  f"unrestricted acc {accuracy_score(y, p_all):.4f} "
                  f"F1 {f1_score(y, p_all, labels=labels, average='macro'):.4f}")


if __name__ == "__main__":
    main()
