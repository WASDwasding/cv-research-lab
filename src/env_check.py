"""最小训练：证明这台机器能把分类跑完，并打印验证准确率。

优先用 CIFAR-10。下载失败时改用本地合成的 3 档图像，不中断检查。
数据放在 data/raw/，该目录不进 git。
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SyntheticGrades(Dataset):
    """合成 3 档：无 / 轻 / 中重。只用于环境检查，不进论文主表。"""

    def __init__(self, n: int, seed: int, train: bool) -> None:
        generator = torch.Generator().manual_seed(seed + (0 if train else 1))
        self.images = torch.rand(n, 3, 32, 32, generator=generator)
        self.labels = torch.randint(0, 3, (n,), generator=generator)
        yy = torch.linspace(0, 1, 32).view(32, 1)
        xx = torch.linspace(0, 1, 32).view(1, 32)
        grid = xx + yy
        for i, label in enumerate(self.labels.tolist()):
            if label == 1:
                self.images[i, :, 16, :] += 0.8
            elif label == 2:
                self.images[i] += 0.35 * grid
                self.images[i, :, 8:24, 8:24] += 0.9
        self.images.clamp_(0, 1)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        return self.images[index], int(self.labels[index])


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(root: Path, seed: int, train_n: int, val_n: int):
    try:
        from torchvision import datasets, transforms

        transform = transforms.ToTensor()
        train_set = datasets.CIFAR10(root, train=True, download=True, transform=transform)
        val_set = datasets.CIFAR10(root, train=False, download=True, transform=transform)
        generator = torch.Generator().manual_seed(seed)
        train_idx = torch.randperm(len(train_set), generator=generator)[:train_n].tolist()
        val_idx = torch.randperm(len(val_set), generator=generator)[:val_n].tolist()
        return Subset(train_set, train_idx), Subset(val_set, val_idx), "CIFAR-10", 10
    except Exception as exc:
        print(f"CIFAR-10 下载或读取失败，改用合成 3 档：{exc}")
        return (
            SyntheticGrades(train_n, seed, train=True),
            SyntheticGrades(val_n, seed, train=False),
            "synthetic-3grade",
            3,
        )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images).argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-n", type=int, default=8000)
    parser.add_argument("--val-n", type=int, default=2000)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/cifar10"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, source, num_classes = load_data(
        args.data_root, args.seed, args.train_n, args.val_n
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = TinyCNN(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"torch {torch.__version__}")
    print(f"cuda {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"gpu {torch.cuda.get_device_name(0)}")
    print(f"dataset {source} classes {num_classes}")
    print(f"train {len(train_set)} val {len(val_set)} device {device}")

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.numel()
            seen += labels.numel()
        val_acc = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch} train_loss {running / seen:.4f} val_acc {val_acc:.4f}",
            flush=True,
        )
    print(f"elapsed_sec {time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
