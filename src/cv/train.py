# src/cv/train.py
import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from datasets import load_from_disk
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ── Dataset ──────────────────────────────────────────────────────────────────

class RVLCDIPDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, item["label"]


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(image_size, augmentation=True):
    if augmentation:
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(architecture, num_classes, pretrained):
    model = timm.create_model(
        architecture,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


# ── Evaluate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {cfg['experiment']['name']}")
    print(f"Architecture: {cfg['model']['architecture']}")
    print(f"Augmentation: {cfg['training']['augmentation']}")

    # Data
    data_dir = Path("data/samples/rvlcdip") if cfg["data"]["sample_mode"] \
               else Path(cfg["data"]["data_dir"])

    train_tf, val_tf = get_transforms(
        cfg["data"]["image_size"],
        cfg["training"]["augmentation"]
    )
    train_ds = RVLCDIPDataset(load_from_disk(str(data_dir / "train")), train_tf)
    val_ds   = RVLCDIPDataset(load_from_disk(str(data_dir / "test")),  val_tf)

    train_loader = DataLoader(train_ds,
                              batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,
                              batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=2)

    # Model
    model = build_model(
        cfg["model"]["architecture"],
        cfg["data"]["num_classes"],
        cfg["model"]["pretrained"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    # MLflow setup
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=cfg["experiment"]["name"]):

        # Log all config params
        mlflow.log_params({
            "architecture":  cfg["model"]["architecture"],
            "pretrained":    cfg["model"]["pretrained"],
            "epochs":        cfg["training"]["epochs"],
            "batch_size":    cfg["training"]["batch_size"],
            "lr":            cfg["training"]["learning_rate"],
            "augmentation":  cfg["training"]["augmentation"],
            "dataset":       cfg["data"]["dataset"],
            "num_classes":   cfg["data"]["num_classes"],
            "sample_mode":   cfg["data"]["sample_mode"],
        })
        mlflow.set_tags(cfg["experiment"]["tags"])

        best_val_acc = 0.0
        patience_counter = 0
        patience = cfg["training"]["early_stopping_patience"]

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, preds, labels = evaluate(
                model, val_loader, criterion, device)
            scheduler.step()

            # Log metrics every epoch -> full learning curve in MLflow
            mlflow.log_metrics({
                "train_loss": round(train_loss, 4),
                "train_acc":  round(train_acc, 4),
                "val_loss":   round(val_loss, 4),
                "val_acc":    round(val_acc, 4),
            }, step=epoch)

            print(f"Epoch {epoch:02d} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            # Track best model weights
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                out_dir = Path(cfg["output"]["model_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                print(f"  -> New best model saved (val_acc={val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Log final summary metric
        mlflow.log_metric("best_val_acc", best_val_acc)
        print(f"\nBest val accuracy: {best_val_acc:.4f}")
        print(classification_report(labels, preds))

        # Log model ONCE at end of training
        if cfg["mlflow"]["log_model"]:
            print("Logging model to MLflow...")
            mlflow.pytorch.log_model(
                model,
                name="model"
            )
            print("Model logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cv_classifier.yaml")
    args = parser.parse_args()
    main(args.config)
