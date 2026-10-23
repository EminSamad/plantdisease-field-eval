"""Check the class-mapping files. Pure Python: python tests/test_mappings.py"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "mappings"


def read(name):
    with open(M / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    classes = [line.rstrip("\n") for line in open(M / "plantvillage_classes.txt", encoding="utf-8") if line.strip()]
    assert len(classes) == 38 and len(set(classes)) == 38
    assert classes == sorted(classes), "ImageFolder sorts class names; the list must be sorted"

    for name, expected in (("plantdoc_mapping.csv", 27), ("plantwild_mapping.csv", 33)):
        rows = read(name)
        assert len(rows) == expected, f"{name}: {len(rows)} rows, expected {expected}"
        sources = [r["field_class"] for r in rows]
        assert len(set(sources)) == len(sources), f"{name}: duplicate field class"
        for r in rows:
            assert r["plantvillage_class"] in classes, f"{name}: unknown class {r['plantvillage_class']}"
        print(f"{name}: OK ({len(rows)} classes, {len({r['plantvillage_class'] for r in rows})} distinct targets)")


if __name__ == "__main__":
    main()
