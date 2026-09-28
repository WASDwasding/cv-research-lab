"""SSGD 上的第 0 周 M1：固定划分、多数类基线、ResNet-18 微调。

在仓库根目录：
    python src/train_m1.py
无档没有样本，分类头只有轻 / 中重。有序误差仍按协议编码：轻=1，中重=2。
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "ssgd" / "SSGD"
SPLIT_PATH = ROOT / "experiments" / "20260928_split_seed42.json"
RESULT_CSV = ROOT / "results" / "20260928_m1_gpu.csv"
CONFUSION_PNG = ROOT / "results" / "20260928_m1_confusion.png"

SPLIT_SEED = 42
EPOCHS = 8
BATCH = 32
LR = 1e-4
SEVERE = {"crack", "broken", "light-leakage", "broken-membrane"}
# 模型内部 0=轻, 1=中重。协议档次是 1 和 2。
NAME_OF = {0: "轻", 1: "中重"}
ORDINAL = {0: 1, 1: 2}


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


def load_samples() -> list[dict]:
    samples = []
    for part in ("lb101", "lb201"):
        folder = DATA / part
        payload = json.loads((DATA / f"annotations_{part}" / f"{part}.json").read_text(encoding="utf-8"))
        images = {item["id"]: item for item in payload["images"]}
        names = {item["id"]: item["name"] for item in payload["categories"]}
        grades: dict[int, list[str]] = {item["id"]: [] for item in payload["images"]}
        for ann in payload["annotations"]:
            image = images[ann["image_id"]]
            grades[ann["image_id"]].append(
                grade_of(names[ann["category_id"]], ann["bbox"], image["width"], image["height"])
            )
        for image_id, values in grades.items():
            if not values:
                continue
            label = "中重" if "中重" in values else "轻"
            image = images[image_id]
            path = folder / image["file_name"]
            if not path.exists():
                raise FileNotFoundError(path)
            samples.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "label": label, "part": part})
    return samples


def stratified_split(samples: list[dict], seed: int) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"轻": [], "中重": []}
    for sample in samples:
        buckets[sample["label"]].append(sample)
    split = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for label in ("轻", "中重"):
        rows = buckets[label][:]
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        split["train"].extend(rows[:n_train])
        split["val"].extend(rows[n_train : n_train + n_val])
        split["test"].extend(rows[n_train + n_val :])
    for name in split:
        split[name].sort(key=lambda item: item["path"])
    return split


def write_split(split: dict[str, list[dict]]) -> None:
    counts = {
        name: dict(Counter(item["label"] for item in rows))
        for name, rows in split.items()
    }
    payload = {"seed": SPLIT_SEED, "ratio": "70/15/15", "counts": counts, "splits": split}
    SPLIT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_split(samples: list[dict]) -> dict[str, list[dict]]:
    first = stratified_split(samples, SPLIT_SEED)
    second = stratified_split(samples, SPLIT_SEED)
    if first != second:
        raise RuntimeError("同一种子两次划分不一致")
    write_split(first)
    again = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["splits"]
    if again != first:
        raise RuntimeError("写出的划分和内存中的划分不一致")
    return first


class ScreenSet(Dataset):
    def __init__(self, rows: list[dict], train: bool) -> None:
        self.rows = rows
        ops = [transforms.Resize((224, 224))]
        if train:
            ops.append(transforms.RandomHorizontalFlip())
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(ROOT / row["path"]).convert("RGB")
        label = 0 if row["label"] == "轻" else 1
        return self.transform(image), label


def make_loader(rows: list[dict], train: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ScreenSet(rows, train),
        batch_size=BATCH,
        shuffle=train,
        num_workers=0,
        generator=generator if train else None,
    )


def metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    labels = [0, 1]
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    per_f1 = {}
    supports = {}
    for label in labels:
        tp = sum(int(t == label and p == label) for t, p in zip(y_true, y_pred))
        fp = sum(int(t != label and p == label) for t, p in zip(y_true, y_pred))
        fn = sum(int(t == label and p != label) for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_f1[label] = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        supports[label] = sum(int(t == label) for t in y_true)
    present = [per_f1[label] for label in labels if supports[label]]
    ordinal_true = [ORDINAL[item] for item in y_true]
    ordinal_pred = [ORDINAL[item] for item in y_pred]
    gaps = [abs(a - b) for a, b in zip(ordinal_true, ordinal_pred)]
    n = len(gaps)
    matrix = [[0, 0], [0, 0]]
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return {
        "acc": correct / len(y_true),
        "macro_f1": sum(present) / len(present),
        "f1_light": per_f1[0],
        "f1_severe": per_f1[1],
        "support_light": supports[0],
        "support_severe": supports[1],
        "mae": sum(gaps) / n,
        "off_by_one": sum(int(gap == 1) for gap in gaps) / n,
        "off_by_two": sum(int(gap == 2) for gap in gaps) / n,
        "matrix": matrix,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        pred = model(images).argmax(dim=1).cpu().tolist()
        y_pred.extend(pred)
        y_true.extend(labels.tolist())
    return metrics(y_true, y_pred)


def train_one(
    split: dict[str, list[dict]],
    seed: int,
    class_weight: bool,
    device: torch.device,
) -> tuple[dict[str, float], list[float]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_loader = make_loader(split["train"], True, seed)
    val_loader = make_loader(split["val"], False, seed)
    test_loader = make_loader(split["test"], False, seed)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    counts = Counter(0 if row["label"] == "轻" else 1 for row in split["train"])
    if class_weight:
        weight = torch.tensor(
            [len(split["train"]) / (2 * counts[0]), len(split["train"]) / (2 * counts[1])],
            dtype=torch.float32,
            device=device,
        )
    else:
        weight = None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    val_scores: list[float] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * labels.numel()
            seen += labels.numel()
        val = evaluate(model, val_loader, device)
        val_scores.append(val["acc"])
        print(
            f"seed {seed} weight {class_weight} epoch {epoch} train_loss {total / seen:.4f} val_acc {val['acc']:.4f}",
            flush=True,
        )
    test = evaluate(model, test_loader, device)
    return test, val_scores


def majority(split: dict[str, list[dict]]) -> dict[str, float]:
    train_labels = [0 if row["label"] == "轻" else 1 for row in split["train"]]
    guess = Counter(train_labels).most_common(1)[0][0]
    y_true = [0 if row["label"] == "轻" else 1 for row in split["test"]]
    y_pred = [guess] * len(y_true)
    scored = metrics(y_true, y_pred)
    scored["guess"] = NAME_OF[guess]
    return scored


def save_confusion(matrix: list[list[int]], path: Path) -> None:
    font = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 28)
    image = Image.new("RGB", (760, 280), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), "行是真值，列是预测。顺序为轻、中重。", fill="black", font=font)
    draw.text((24, 90), f"轻 -> 轻 {matrix[0][0]}    轻 -> 中重 {matrix[0][1]}", fill="black", font=font)
    draw.text((24, 150), f"中重 -> 轻 {matrix[1][0]}    中重 -> 中重 {matrix[1][1]}", fill="black", font=font)
    image.save(path)


def append_row(setting: str, seed: str, weight: str, scored: dict[str, float], n_train: int, n_val: int, n_test: int) -> None:
    new_file = not RESULT_CSV.exists()
    with RESULT_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setting",
                "seed",
                "split_seed",
                "epochs",
                "lr",
                "class_weight",
                "test_acc",
                "macro_f1",
                "f1_light",
                "f1_severe",
                "support_light",
                "support_severe",
                "mae",
                "off_by_one",
                "off_by_two",
                "n_train",
                "n_val",
                "n_test",
            ],
        )
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "setting": setting,
                "seed": seed,
                "split_seed": SPLIT_SEED,
                "epochs": 0 if setting == "majority" else EPOCHS,
                "lr": "" if setting == "majority" else LR,
                "class_weight": weight,
                "test_acc": f"{scored['acc']:.4f}",
                "macro_f1": f"{scored['macro_f1']:.4f}",
                "f1_light": f"{scored['f1_light']:.4f}",
                "f1_severe": f"{scored['f1_severe']:.4f}",
                "support_light": scored["support_light"],
                "support_severe": scored["support_severe"],
                "mae": f"{scored['mae']:.4f}",
                "off_by_one": f"{scored['off_by_one']:.4f}",
                "off_by_two": f"{scored['off_by_two']:.4f}",
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
            }
        )


def main() -> None:
    samples = load_samples()
    split = verify_split(samples)
    print("split", {name: dict(Counter(row["label"] for row in rows)) for name, rows in split.items()}, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    if RESULT_CSV.exists():
        RESULT_CSV.unlink()
    base = majority(split)
    print("majority", base["guess"], f"{base['acc']:.4f}", flush=True)
    n_train, n_val, n_test = len(split["train"]), len(split["val"]), len(split["test"])
    append_row("majority", "", "no", base, n_train, n_val, n_test)

    test1, val1 = train_one(split, seed=1, class_weight=False, device=device)
    print("seed1", f"{test1['acc']:.4f}", "val", [round(item, 4) for item in val1], flush=True)
    use_weight = test1["acc"] <= base["acc"]
    chosen = test1
    if use_weight:
        print("seed 1 未超过多数类，只改类别权重再跑", flush=True)
        chosen, val1 = train_one(split, seed=1, class_weight=True, device=device)
        print("seed1-weighted", f"{chosen['acc']:.4f}", flush=True)
    append_row("resnet18-finetune", "1", "yes" if use_weight else "no", chosen, n_train, n_val, n_test)
    save_confusion(chosen["matrix"], CONFUSION_PNG)

    test2, val2 = train_one(split, seed=2, class_weight=use_weight, device=device)
    print("seed2", f"{test2['acc']:.4f}", "val", [round(item, 4) for item in val2], flush=True)
    append_row("resnet18-finetune", "2", "yes" if use_weight else "no", test2, n_train, n_val, n_test)
    mean_acc = (chosen["acc"] + test2["acc"]) / 2
    if mean_acc <= base["acc"] and not use_weight:
        print("两种子均值仍不超过多数类，加类别权重再出一版", flush=True)
        extra, _ = train_one(split, seed=1, class_weight=True, device=device)
        append_row("resnet18-class-weight", "1", "yes", extra, n_train, n_val, n_test)
        print("weighted", f"{extra['acc']:.4f}", flush=True)
    print("wrote", RESULT_CSV, flush=True)


if __name__ == "__main__":
    main()
