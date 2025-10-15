import os
from pathlib import Path
from typing import Tuple
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

# COCO class index for 'bird' is 16 (0-based indexing with background ignored in predictions)
COCO_BIRD_CLASS = 16

def load_detector(device="cpu"):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device)
    model.eval()
    return model

@torch.no_grad()
def detect_bird_bbox(model, img_pil: Image.Image, device="cpu", score_thresh=0.6):
    # Convert PIL -> tensor
    tf = transforms.ToTensor()
    x = tf(img_pil).to(device)
    pred = model([x])[0]
    boxes = pred["boxes"]
    labels = pred["labels"]
    scores = pred["scores"]

    best = None
    best_score = -1.0
    for b, l, s in zip(boxes, labels, scores):
        if int(l.item()) == COCO_BIRD_CLASS and float(s.item()) >= score_thresh and s.item() > best_score:
            best = b
            best_score = float(s.item())
    if best is None:
        return None, 0.0
    # box = [xmin, ymin, xmax, ymax]
    return best.cpu().numpy().tolist(), best_score

def expand_and_clip(box, w, h, margin=0.10) -> Tuple[int,int,int,int]:
    xmin, ymin, xmax, ymax = box
    bw = xmax - xmin
    bh = ymax - ymin
    xmin -= bw * margin
    ymin -= bh * margin
    xmax += bw * margin
    ymax += bh * margin
    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(w, int(xmax))
    ymax = min(h, int(ymax))
    # Ensure at least 1px
    xmax = max(xmax, xmin + 1)
    ymax = max(ymax, ymin + 1)
    return xmin, ymin, xmax, ymax

def crop_save_png(src_path: Path, dst_path: Path, model, device="cpu",
                  score_thresh=0.6, margin=0.10, min_side=64):
    with Image.open(src_path) as im0:
        im = ImageOps.exif_transpose(im0.convert("RGB"))
    w, h = im.size

    box, score = detect_bird_bbox(model, im, device=device, score_thresh=score_thresh)
    if box is None:
        # Fallback: center crop square to avoid losing image
        side = max(min(w, h), min_side)
        left = (w - side) // 2
        top  = (h - side) // 2
        right = left + side
        bottom = top + side
    else:
        left, top, right, bottom = expand_and_clip(box, w, h, margin=margin)

    crop = im.crop((left, top, right, bottom))

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path = dst_path.with_suffix(".png")
    crop.save(dst_path, format="PNG", compress_level=6)

def main():
    import argparse
    print("in main")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source root (e.g., data/birds)")
    ap.add_argument("--dst", required=True, help="Destination root for cropped PNGs")
    ap.add_argument("--score", type=float, default=0.6, help="Detection score threshold")
    ap.add_argument("--margin", type=float, default=0.10, help="Relative bbox margin")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    model = load_detector(args.device)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in src_root.rglob("*") if p.suffix.lower() in exts]

    for p in tqdm(files, desc="Cropping"):
        rel = p.relative_to(src_root)
        out = (dst_root / rel).with_suffix(".png")
        try:
            crop_save_png(p, out, model, device=args.device, score_thresh=args.score, margin=args.margin)
        except Exception as e:
            print(f"[WARN] Failed on {p}: {e}")

if __name__ == "__main__":
    main()
