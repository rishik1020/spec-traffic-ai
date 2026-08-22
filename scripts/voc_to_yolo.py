"""
voc_to_yolo.py
==============
Convert Pascal VOC annotations (.xml) into YOLO format (.txt).

The DataCluster plate sample ships as VOC XML. Ultralytics wants YOLO txt:

    VOC   <xmin>412</xmin> <ymin>288</ymin> <xmax>533</xmax> <ymax>331</ymax>
    YOLO  0 0.492188 0.430556 0.094531 0.059722
          ^ class    ^ x_centre ^ y_centre ^ width ^ height     (all 0-1 normalised)

YOLO coordinates are normalised to image size, so a box keeps its meaning if the
image is resized during augmentation -- which is exactly what happens in training.

USAGE
    python scripts/voc_to_yolo.py --dataset datasets/plates/datacluster-sample
    python scripts/voc_to_yolo.py --dataset <dir> --split 0.8
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_voc(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """Return (width, height, [(label, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path.name}: no <size> element")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))

    boxes = []
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if not label or bb is None:
            continue
        boxes.append((
            label,
            float(bb.findtext("xmin", "0")),
            float(bb.findtext("ymin", "0")),
            float(bb.findtext("xmax", "0")),
            float(bb.findtext("ymax", "0")),
        ))
    return width, height, boxes


def to_yolo_line(cls_id: int, xmin, ymin, xmax, ymax, w: int, h: int) -> str | None:
    """Normalise a VOC box to a YOLO line, clamped to the image."""
    xmin, xmax = max(0.0, min(xmin, xmax)), min(float(w), max(xmin, xmax))
    ymin, ymax = max(0.0, min(ymin, ymax)), min(float(h), max(ymin, ymax))

    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 1 or bh <= 1:      # degenerate box -- drop it
        return None

    xc = (xmin + xmax) / 2.0 / w
    yc = (ymin + ymax) / 2.0 / h
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="folder containing images/ and Annotations/")
    ap.add_argument("--split", type=float, default=0.8, help="train fraction")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    img_dir = root / "images"
    ann_dir = root / "Annotations"

    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise SystemExit(f"expected {img_dir} and {ann_dir} to exist")

    # ---- pair images with their annotation -------------------------------
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        xml = ann_dir / f"{img.stem}.xml"
        if xml.exists():
            pairs.append((img, xml))

    if not pairs:
        raise SystemExit("no image/annotation pairs found")

    # ---- discover the label set ------------------------------------------
    labels: set[str] = set()
    for _, xml in pairs:
        try:
            _, _, boxes = parse_voc(xml)
        except Exception:
            continue
        labels.update(lbl for lbl, *_ in boxes)

    classes = sorted(labels)
    cls_index = {name: i for i, name in enumerate(classes)}
    print(f"[voc2yolo] {len(pairs)} pairs, {len(classes)} classes: {classes}")

    # ---- split ------------------------------------------------------------
    random.Random(args.seed).shuffle(pairs)
    cut = max(1, int(len(pairs) * args.split))
    splits = {"train": pairs[:cut], "val": pairs[cut:]}

    out = root / "yolo"
    if out.exists():
        shutil.rmtree(out)

    written = skipped = 0
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img, xml in items:
            try:
                w, h, boxes = parse_voc(xml)
            except Exception as exc:
                print(f"  skip {xml.name}: {exc}")
                skipped += 1
                continue

            if w <= 0 or h <= 0:
                print(f"  skip {xml.name}: bad image size")
                skipped += 1
                continue

            lines = []
            for label, *coords in boxes:
                line = to_yolo_line(cls_index[label], *coords, w, h)
                if line:
                    lines.append(line)

            shutil.copy2(img, out / "images" / split / img.name)
            (out / "labels" / split / f"{img.stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            written += 1

    # ---- dataset yaml that ultralytics reads directly ---------------------
    yaml_text = (
        f"# generated by voc_to_yolo.py\n"
        f"path: {out.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(classes)}\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes))
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"[voc2yolo] wrote {written} (skipped {skipped})")
    print(f"[voc2yolo] train={len(splits['train'])} val={len(splits['val'])}")
    print(f"[voc2yolo] -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
