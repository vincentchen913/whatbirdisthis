# train.py
import os, json, random
from pathlib import Path
from typing import Tuple, Dict
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import timm

# ======== CONFIG (adjust as needed) ========
DATA_DIR   = "cropped_data/"   # <- your cropped images
OUT_DIR    = "artifacts"
MODEL_NAME = "convnext_tiny.fb_in1k"     # good for fine-grained; try convnext_base too
IMG_SIZE   = 288                    # 4060 Ti can handle 384–448 easily
BATCH_SIZE = 16                     # use 32 if you have 8GB VRAM
LR         = 3e-4
EPOCHS     = 30
VAL_SPLIT  = 0.2
PATIENCE   = 6
NUM_WORKERS= 4
SEED       = 42
# ===========================================

random.seed(SEED); torch.manual_seed(SEED)

def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.15, 0.15, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])
    return train_tf, val_tf

def stratified_split(dataset: datasets.ImageFolder, val_size: float, seed: int=42) -> Tuple[Subset, Subset]:
    y = [label for _, label in dataset.samples]
    idx = list(range(len(y)))
    train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=seed, stratify=y)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

class TransformedSubset(Dataset):
    def __init__(self, subset: Subset, transform=None, target_transform=None):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        path, target = self.dataset.samples[idx]
        img = self.dataset.loader(path)
        if self.transform: img = self.transform(img)
        if self.target_transform: target = self.target_transform(target)
        return img, target

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / target.size(0)))
    return res

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Base dataset (for class mapping)
    ds_base = datasets.ImageFolder(DATA_DIR)
    class_to_idx: Dict[str, int] = ds_base.class_to_idx
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"Found {num_classes} species.")

    # Split with no transform, then wrap with transforms
    train_subset, val_subset = stratified_split(ds_base, VAL_SPLIT, SEED)
    train_tf, val_tf = build_transforms(IMG_SIZE)
    train_ds = TransformedSubset(train_subset, transform=train_tf)
    val_ds   = TransformedSubset(val_subset,   transform=val_tf)

    loader_args = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader   = DataLoader(val_ds, shuffle=False, **loader_args)

    # Model
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes, drop_rate=0.2, drop_path_rate=0.1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Save mappings
    with open(Path(OUT_DIR, "idx_to_class.json"), "w") as f:
        json.dump(idx_to_class, f, indent=2)

    best_acc = 0.0
    no_improve = 0
    ckpt_path = str(Path(OUT_DIR, f"{MODEL_NAME}_best.pt"))

    for epoch in range(1, EPOCHS+1):
        # ---- Train
        model.train()
        tr_loss = tr_acc1 = tr_acc5 = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()
            a1, a5 = accuracy(logits, y, topk=(1, min(5, num_classes)))
            bs = x.size(0)
            tr_loss += loss.item() * bs
            tr_acc1 += a1.item() * bs / 100.0
            tr_acc5 += a5.item() * bs / 100.0

        ntr = len(train_ds)
        tr_loss /= ntr; tr_acc1 = 100*tr_acc1/ntr; tr_acc5 = 100*tr_acc5/ntr

        # ---- Val
        model.eval()
        va_loss = va_acc1 = va_acc5 = 0.0
        all_t, all_p = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                a1, a5 = accuracy(logits, y, topk=(1, min(5, num_classes)))
                bs = x.size(0)
                va_loss += loss.item() * bs
                va_acc1 += a1.item() * bs / 100.0
                va_acc5 += a5.item() * bs / 100.0
                all_t.extend(y.cpu().tolist())
                all_p.extend(logits.argmax(1).cpu().tolist())

        nva = len(val_ds)
        va_loss /= nva; va_acc1 = 100*va_acc1/nva; va_acc5 = 100*va_acc5/nva
        scheduler.step()

        print(f"\nEpoch {epoch}: "
              f"train_loss={tr_loss:.4f} top1={tr_acc1:.2f}% top5={tr_acc5:.2f}% | "
              f"val_loss={va_loss:.4f} top1={va_acc1:.2f}% top5={va_acc5:.2f}%")

        # Save best
        if va_acc1 > best_acc:
            best_acc = va_acc1; no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "class_to_idx": class_to_idx,
                "model_name": MODEL_NAME,
                "img_size": IMG_SIZE
            }, ckpt_path)
            print(f"✅ Saved best to {ckpt_path} (val_top1 {best_acc:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("⏹ Early stopping."); break

    print(f"\nBest validation top-1: {best_acc:.2f}%")
    # Brief report on the last epoch (val set)
    try:
        print("\nValidation classification report (last epoch):")
        print(classification_report(all_t, all_p, digits=3))
    except Exception:
        pass

if __name__ == "__main__":
    main()
