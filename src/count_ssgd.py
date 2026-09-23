"""按 COCO 标注统计 SSGD，并映射到三档。

官方 7 类在 annotations_lb101/lb101.json 与 annotations_lb201/lb201.json。
lb101、lb201 里的 XML 用的是另一套拼音标签，不参与计数。

划痕长度 = 框的长边。全部图像是 1500×1000，短边 S=1000，0.2S=200 像素。
面积 = 框面积 / 整图面积。阈值 5%。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "ssgd" / "SSGD"
SEVERE = {"crack", "broken", "light-leakage", "broken-membrane"}


def grade_of(name: str, bbox: list[float], width: int, height: int) -> str:
    box_w, box_h = bbox[2], bbox[3]
    short = min(width, height)
    if name in SEVERE:
        return "中重"
    if name == "scratch":
        return "中重" if max(box_w, box_h) >= 0.2 * short else "轻"
    if name in {"spot", "blot"}:
        area = (box_w * box_h) / (width * height)
        return "中重" if area >= 0.05 else "轻"
    raise KeyError(name)


def main() -> None:
    box_rows: Counter[tuple[str, str]] = Counter()
    image_rows: Counter[tuple[str, str]] = Counter()
    label_boxes: Counter[str] = Counter()
    final = Counter()
    sizes: Counter[tuple[int, int]] = Counter()
    for part in ("lb101", "lb201"):
        data = json.loads((ROOT / f"annotations_{part}" / f"{part}.json").read_text(encoding="utf-8"))
        images = {item["id"]: item for item in data["images"]}
        names = {item["id"]: item["name"] for item in data["categories"]}
        hit: dict[int, set[tuple[str, str]]] = {item["id"]: set() for item in data["images"]}
        grades: dict[int, list[str]] = {item["id"]: [] for item in data["images"]}
        for image in images.values():
            sizes[(image["width"], image["height"])] += 1
        for ann in data["annotations"]:
            image = images[ann["image_id"]]
            name = names[ann["category_id"]]
            grade = grade_of(name, ann["bbox"], image["width"], image["height"])
            label_boxes[name] += 1
            box_rows[(name, grade)] += 1
            hit[ann["image_id"]].add((name, grade))
            grades[ann["image_id"]].append(grade)
        for pairs in hit.values():
            for pair in pairs:
                image_rows[pair] += 1
        for values in grades.values():
            if not values:
                final["没有7类框"] += 1
            elif "中重" in values:
                final["中重"] += 1
            else:
                final["轻"] += 1
    print(f"images {sum(sizes.values())} sizes {dict(sizes)}")
    print("BOXES")
    for name, count in sorted(label_boxes.items()):
        print(f"{name} {count}")
    print("ROWS box image")
    for key in sorted(box_rows):
        print(f"{key[0]} {key[1]} {box_rows[key]} {image_rows[key]}")
    print("FINAL")
    for key, count in sorted(final.items()):
        print(f"{key} {count}")


if __name__ == "__main__":
    main()
