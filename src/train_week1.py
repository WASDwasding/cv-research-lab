"""第 1 周对照：线性探测、CORAL、强增广、无增广、定位再分级、错例清单。

划分用 experiments/20260928_split_seed42.json，不重新抽样。
论文正文不在这个脚本里改。

    python src/train_week1.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from train_m1 import (
    BATCH,
    DATA,
    EPOCHS,
    LR,
    ROOT,
    SPLIT_PATH,
    metrics,
)

OUT_CSV = ROOT / "results" / "20261008_week1_gpu.csv"
SIZE_CSV = ROOT / "results" / "20261008_size_groups.csv"
FAIL_CSV = ROOT / "results" / "20261008_failures.csv"
VIS_DIR = ROOT / "results" / "20261008_locate_vis"
CKPT = ROOT / "results" / "checkpoints" / "resnet18_seed1.pt"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SCORE_KEEP = 0.3


def normalize():
    return transforms.Normalize(mean=MEAN, std=STD)


def build_transform(mode: str, train: bool):
    ops = [transforms.Resize((224, 224))]
    if train and mode == "flip":
        ops.append(transforms.RandomHorizontalFlip())
    if train and mode == "strong":
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
                transforms.RandomRotation(15),
            ]
        )
    ops.extend([transforms.ToTensor(), normalize()])
    return transforms.Compose(ops)


class GradeSet(Dataset):
    def __init__(self, rows: list[dict], mode: str, train: bool) -> None:
        self.rows = rows
        self.transform = build_transform(mode, train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(ROOT / row["path"]).convert("RGB")
        label = 0 if row["label"] == "轻" else 1
        return self.transform(image), label


def loader(rows, mode, train, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        GradeSet(rows, mode, train),
        batch_size=BATCH,
        shuffle=train,
        num_workers=0,
        generator=generator if train else None,
    )


@torch.no_grad()
def predict(model, rows, device, mode="flip") -> list[int]:
    model.eval()
    preds = []
    data = DataLoader(GradeSet(rows, mode, False), batch_size=BATCH, shuffle=False, num_workers=0)
    for images, _labels in data:
        preds.extend(model(images.to(device)).argmax(dim=1).cpu().tolist())
    return preds


def train_classifier(rows_train, rows_val, device, seed, mode, linear_probe, coral):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if linear_probe:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    if coral:
        model.fc = nn.Identity()
        head = CoralHead(in_features).to(device)
    else:
        model.fc = nn.Linear(in_features, 2)
        head = None
    model = model.to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    if head is not None:
        params += list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCEWithLogitsLoss()
    train_loader = loader(rows_train, mode, True, seed)
    val_loader = loader(rows_val, mode, False, seed)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if head is not None:
            head.train()
        total = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if coral:
                logits = head(model(images))
                target = torch.stack([torch.ones_like(labels), labels], dim=1).float()
                loss = loss_bce(logits, target)
            else:
                loss = loss_ce(model(images), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * labels.numel()
            seen += labels.numel()
        val_pred = run_eval(model, head, val_loader, device, coral)
        history.append(val_pred["acc"])
        print(
            f"{'coral' if coral else 'ce'} {mode} probe {linear_probe} epoch {epoch} "
            f"loss {total / seen:.4f} val_acc {val_pred['acc']:.4f}",
            flush=True,
        )
    return model, head, history


@torch.no_grad()
def run_eval(model, head, data_loader, device, coral) -> dict:
    model.eval()
    if head is not None:
        head.eval()
    y_true, y_pred = [], []
    none_pred = 0
    for images, labels in data_loader:
        images = images.to(device)
        if coral:
            bits = (torch.sigmoid(head(model(images))) > 0.5).int()
            none_pred += int((bits[:, 0] == 0).sum().item())
            pred = (bits.sum(dim=1) >= 2).int()
        else:
            pred = model(images).argmax(dim=1)
        y_pred.extend(pred.cpu().tolist())
        y_true.extend(labels.tolist())
    scored = metrics(y_true, y_pred)
    scored["pred_none"] = none_pred
    return scored


class CoralHead(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.weight = nn.Linear(features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, features):
        return self.weight(features) + self.bias


def load_split():
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    return payload["splits"]


def index_boxes() -> dict[str, list[dict]]:
    boxes = defaultdict(list)
    for part in ("lb101", "lb201"):
        payload = json.loads((DATA / f"annotations_{part}" / f"{part}.json").read_text(encoding="utf-8"))
        images = {item["id"]: item for item in payload["images"]}
        names = {item["id"]: item["name"] for item in payload["categories"]}
        for ann in payload["annotations"]:
            image = images[ann["image_id"]]
            path = f"data/raw/ssgd/SSGD/{part}/{image['file_name']}"
            x, y, w, h = ann["bbox"]
            boxes[path].append(
                {
                    "name": names[ann["category_id"]],
                    "cid": ann["category_id"],
                    "box": [x, y, x + w, y + h],
                    "area": w * h / (image["width"] * image["height"]),
                }
            )
    return boxes


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_of(setting: str, scored: dict, extra: dict | None = None) -> dict:
    payload = {
        "setting": setting,
        "test_acc": f"{scored['acc']:.4f}",
        "macro_f1": f"{scored['macro_f1']:.4f}",
        "f1_light": f"{scored['f1_light']:.4f}",
        "f1_severe": f"{scored['f1_severe']:.4f}",
        "support_light": scored["support_light"],
        "support_severe": scored["support_severe"],
        "mae": f"{scored['mae']:.4f}",
        "off_by_one": f"{scored['off_by_one']:.4f}",
        "off_by_two": f"{scored['off_by_two']:.4f}",
    }
    if extra:
        payload.update(extra)
    return payload


def train_detector(split, box_index, device):
    class DetSet(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            image = Image.open(ROOT / row["path"]).convert("RGB")
            tensor = transforms.ToTensor()(image)
            items = box_index.get(row["path"], [])
            if items:
                boxes = torch.tensor([item["box"] for item in items], dtype=torch.float32)
                labels = torch.tensor([item["cid"] for item in items], dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            return tensor, {"boxes": boxes, "labels": labels}

    def collate(batch):
        return tuple(zip(*batch))

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 8)
    model.transform.min_size = (480,)
    model.transform.max_size = 640
    model = model.to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
    data = DataLoader(DetSet(split["train"]), batch_size=2, shuffle=True, collate_fn=collate, num_workers=0)
    model.train()
    for epoch in range(1, 4):
        total = 0.0
        steps = 0
        for images, targets in data:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            steps += 1
        print(f"detector epoch {epoch} loss {total / max(steps, 1):.4f}", flush=True)
    return model


@torch.no_grad()
def detect(model, rows, device):
    model.eval()
    outputs = []
    for row in rows:
        image = Image.open(ROOT / row["path"]).convert("RGB")
        tensor = transforms.ToTensor()(image).to(device)
        pred = model([tensor])[0]
        keep = pred["scores"] >= SCORE_KEEP
        outputs.append(
            {
                "boxes": pred["boxes"][keep].cpu(),
                "scores": pred["scores"][keep].cpu(),
                "labels": pred["labels"][keep].cpu(),
                "all_boxes": pred["boxes"].cpu(),
                "all_scores": pred["scores"].cpu(),
                "all_labels": pred["labels"].cpu(),
                "size": image.size,
            }
        )
    return outputs


def iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def average_precision(rows, detections, box_index) -> float:
    by_class = defaultdict(list)
    gt_count = Counter()
    for row, pred in zip(rows, detections):
        gts = box_index.get(row["path"], [])
        for class_id in range(1, 8):
            gt_count[class_id] += sum(item["cid"] == class_id for item in gts)
            for box, score, label in zip(pred["all_boxes"], pred["all_scores"], pred["all_labels"]):
                if int(label) == class_id:
                    by_class[class_id].append((float(score), row["path"], [float(v) for v in box]))
    aps = []
    for class_id in range(1, 8):
        preds = sorted(by_class[class_id], key=lambda item: item[0], reverse=True)
        matched = defaultdict(set)
        tp = []
        fp = []
        for _score, path, box in preds:
            gts = [item["box"] for item in box_index.get(path, []) if item["cid"] == class_id]
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gts):
                if j in matched[path]:
                    continue
                value = iou(box, gt)
                if value > best_iou:
                    best_iou = value
                    best_j = j
            if best_iou >= 0.5 and best_j >= 0:
                matched[path].add(best_j)
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        if gt_count[class_id] == 0:
            continue
        tp_cum = 0
        fp_cum = 0
        recall_points = []
        precision_points = []
        for t, f in zip(tp, fp):
            tp_cum += t
            fp_cum += f
            recall_points.append(tp_cum / gt_count[class_id])
            precision_points.append(tp_cum / (tp_cum + fp_cum))
        ap = 0.0
        for recall_level in [i / 10 for i in range(11)]:
            precisions = [p for r, p in zip(recall_points, precision_points) if r >= recall_level]
            ap += max(precisions) if precisions else 0.0
        aps.append(ap / 11)
    return sum(aps) / len(aps) if aps else 0.0


def apply_region(image: Image.Image, boxes: torch.Tensor, masked: bool) -> Image.Image:
    if boxes.numel() == 0:
        if masked:
            return Image.new("RGB", image.size, "black")
        return image
    x1 = int(max(0, boxes[:, 0].min().item()))
    y1 = int(max(0, boxes[:, 1].min().item()))
    x2 = int(min(image.width, boxes[:, 2].max().item()))
    y2 = int(min(image.height, boxes[:, 3].max().item()))
    if x2 <= x1 or y2 <= y1:
        if masked:
            return Image.new("RGB", image.size, "black")
        return image
    if not masked:
        return image.crop((x1, y1, x2, y2))
    canvas = Image.new("RGB", image.size, "black")
    canvas.paste(image.crop((x1, y1, x2, y2)), (x1, y1))
    return canvas


@torch.no_grad()
def grade_regions(model, rows, detections, device, masked: bool) -> list[int]:
    model.eval()
    preds = []
    transform = build_transform("flip", False)
    for row, pred in zip(rows, detections):
        image = Image.open(ROOT / row["path"]).convert("RGB")
        region = apply_region(image, pred["boxes"], masked)
        tensor = transform(region).unsqueeze(0).to(device)
        preds.append(int(model(tensor).argmax(dim=1).item()))
    return preds


def save_overlays(rows, detections, box_index):
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 18)
    saved = 0
    for row, pred in zip(rows, detections):
        if saved >= 10:
            break
        gts = box_index.get(row["path"], [])
        if not gts:
            continue
        image = Image.open(ROOT / row["path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for item in gts:
            draw.rectangle(item["box"], outline="lime", width=3)
        for box, label in zip(pred["boxes"], pred["labels"]):
            draw.rectangle([float(v) for v in box], outline="red", width=2)
            draw.text((float(box[0]), float(box[1])), str(int(label)), fill="red", font=font)
        image.save(VIS_DIR / f"{saved:02d}.jpg")
        saved += 1
    print(f"overlays {saved}", flush=True)


def subset_metrics(rows, preds, keep) -> dict:
    y_true = [0 if row["label"] == "轻" else 1 for row, flag in zip(rows, keep) if flag]
    y_pred = [pred for pred, flag in zip(preds, keep) if flag]
    if not y_true:
        return {"acc": 0, "macro_f1": 0, "f1_light": 0, "f1_severe": 0, "support_light": 0, "support_severe": 0, "mae": 0, "off_by_one": 0, "off_by_two": 0, "n": 0}
    scored = metrics(y_true, y_pred)
    scored["n"] = len(y_true)
    return scored


def main() -> None:
    split = load_split()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    y_true = [0 if row["label"] == "轻" else 1 for row in split["test"]]
    rows = []

    print("retrain seed-1 grader", flush=True)
    grader, _head, _hist = train_classifier(split["train"], split["val"], device, 1, "flip", False, False)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grader.state_dict(), CKPT)
    test_loader = loader(split["test"], "flip", False, 1)
    seed1 = run_eval(grader, None, test_loader, device, False)
    print("seed1 recheck", f"{seed1['acc']:.4f}", flush=True)
    whole_pred = predict(grader, split["test"], device)

    print("linear probe", flush=True)
    probe, _head, _hist = train_classifier(split["train"], split["val"], device, 1, "flip", True, False)
    probe_score = run_eval(probe, None, test_loader, device, False)
    rows.append(row_of("linear-probe", probe_score))
    write_csv(OUT_CSV, rows)
    print("linear", probe_score["acc"], flush=True)

    print("coral", flush=True)
    coral_model, coral_head, _hist = train_classifier(split["train"], split["val"], device, 1, "flip", False, True)
    coral_score = run_eval(coral_model, coral_head, test_loader, device, True)
    rows.append(row_of("coral", coral_score, {"pred_none": coral_score["pred_none"]}))
    write_csv(OUT_CSV, rows)
    print("coral", coral_score["acc"], "pred_none", coral_score["pred_none"], flush=True)

    print("strong aug", flush=True)
    strong, _head, _hist = train_classifier(split["train"], split["val"], device, 1, "strong", False, False)
    strong_loader = loader(split["test"], "strong", False, 1)
    strong_score = run_eval(strong, None, strong_loader, device, False)
    rows.append(row_of("strong-aug", strong_score))
    write_csv(OUT_CSV, rows)
    print("strong", strong_score["acc"], flush=True)

    print("no flip", flush=True)
    plain, _head, _hist = train_classifier(split["train"], split["val"], device, 1, "none", False, False)
    plain_loader = loader(split["test"], "none", False, 1)
    plain_score = run_eval(plain, None, plain_loader, device, False)
    rows.append(row_of("aug-off", plain_score))
    write_csv(OUT_CSV, rows)
    print("aug-off", plain_score["acc"], flush=True)

    print("detector", flush=True)
    box_index = index_boxes()
    detector = train_detector(split, box_index, device)
    val_det = detect(detector, split["val"], device)
    test_det = detect(detector, split["test"], device)
    map50 = average_precision(split["val"], val_det, box_index)
    print(f"val mAP@0.5 {map50:.4f}", flush=True)
    save_overlays(split["val"], val_det, box_index)
    empty = sum(int(item["boxes"].numel() == 0) for item in test_det)
    print(f"test images with no kept box {empty}", flush=True)

    crop_pred = grade_regions(grader, split["test"], test_det, device, masked=False)
    mask_pred = grade_regions(grader, split["test"], test_det, device, masked=True)
    crop_score = metrics(y_true, crop_pred)
    mask_score = metrics(y_true, mask_pred)
    rows.append(row_of("locate-crop", crop_score, {"map50_val": f"{map50:.4f}", "empty_boxes": empty}))
    rows.append(row_of("locate-mask", mask_score, {"map50_val": f"{map50:.4f}", "empty_boxes": empty}))
    write_csv(OUT_CSV, rows)
    print("crop", crop_score["acc"], "mask", mask_score["acc"], flush=True)

    large = []
    for row in split["test"]:
        area = sum(item["area"] for item in box_index.get(row["path"], []))
        large.append(area >= 0.05)
    size_rows = []
    for name, preds in (("whole-seed1", whole_pred), ("locate-crop", crop_pred), ("locate-mask", mask_pred)):
        for group, flags in (("large", large), ("small", [not flag for flag in large])):
            scored = subset_metrics(split["test"], preds, flags)
            size_rows.append(
                {
                    "setting": name,
                    "group": group,
                    "n": scored["n"],
                    "acc": f"{scored['acc']:.4f}",
                    "macro_f1": f"{scored['macro_f1']:.4f}",
                    "f1_light": f"{scored['f1_light']:.4f}",
                    "f1_severe": f"{scored['f1_severe']:.4f}",
                }
            )
    write_csv(SIZE_CSV, size_rows)

    failures = []
    for row, true, pred in zip(split["test"], y_true, whole_pred):
        if true == pred:
            continue
        failures.append(
            {
                "path": row["path"],
                "true": "轻" if true == 0 else "中重",
                "pred": "轻" if pred == 0 else "中重",
                "gap": abs((1 if true == 0 else 2) - (1 if pred == 0 else 2)),
                "type": "相邻档",
            }
        )
    write_csv(FAIL_CSV, failures)
    print("failures", len(failures), flush=True)
    print("wrote", OUT_CSV, flush=True)


if __name__ == "__main__":
    main()
