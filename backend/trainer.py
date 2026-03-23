import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ── config ────────────────────────────────────────────────
DATASET_DIR = "dataset"
MODEL_PATH  = "models/resnet18_classifier.pth"
LABELS_PATH = "models/class_labels.json"
EPOCHS      = 10
BATCH_SIZE  = 16
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False
    # replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def train():
    train_tf, val_tf = get_transforms()

    # ImageFolder expects dataset/class_name/image.jpg structure
    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_tf)

    if len(full_dataset.classes) < 2:
        print("Need at least 2 class folders inside dataset/")
        return

    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")

    # 80/20 train-val split
    val_size   = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_ds.dataset = datasets.ImageFolder(DATASET_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = build_model(len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    print(f"\nTraining on {DEVICE} for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        # ── train ──
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        # ── validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs     = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS}  "
            f"Loss: {train_loss/len(train_loader):.4f}  "
            f"Train Acc: {100*correct/total:.1f}%  "
            f"Val Acc: {100*val_correct/val_total:.1f}%"
        )

    # ── save ──
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(class_names, f)

    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Labels saved → {LABELS_PATH}")


if __name__ == "__main__":
    train()
