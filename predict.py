# predict.py
import argparse
import json
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms


def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt.get("model_name", "convnext_tiny.fb_in1k")
    img_size = ckpt.get("img_size", 224)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return model, tf, idx_to_class, img_size, model_name


@torch.no_grad()
def predict_one(img_path: str, model, tf, idx_to_class, device="cpu", topk=5, tta=False):
    """Return list of (label, prob) sorted by prob desc."""
    im = Image.open(img_path).convert("RGB")

    def _run(im_):
        x = tf(im_).unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
        return probs

    # Test-time augmentation (optional): average with horizontal flip
    if tta:
        probs = 0.5 * (_run(im) + _run(im.transpose(Image.FLIP_LEFT_RIGHT)))
    else:
        probs = _run(im)

    vals, idxs = probs.topk(min(topk, probs.numel()))
    vals = vals.cpu().tolist()
    idxs = idxs.cpu().tolist()
    return [(idx_to_class[i], float(v)) for i, v in zip(idxs, vals)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to saved model .pt")
    ap.add_argument("--img", required=True, help="Path to image file")
    ap.add_argument("--topk", type=int, default=5, help="How many labels to show")
    ap.add_argument("--device", choices=["cuda", "cpu", "mps"], default=None,
                    help="Force device. Default: auto")
    ap.add_argument("--tta", action="store_true", help="Enable simple TTA (adds a horizontal flip)")
    ap.add_argument("--json_out", default=None, help="Optional path to save JSON result")
    args = ap.parse_args()

    # Pick device
    if args.device is not None:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load model + preprocessing
    model, tf, idx_to_class, img_size, model_name = load_checkpoint(args.ckpt)
    model.to(device)

    # Optional memory/throughput niceties
    if device == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    # Predict
    results = predict_one(args.img, model, tf, idx_to_class, device=device, topk=args.topk, tta=args.tta)

    # Print nicely
    print(f"\nModel: {model_name} | img_size={img_size} | device={device} | TTA={'on' if args.tta else 'off'}")
    print(f"Image: {args.img}\n")
    width = max(len(name) for name, _ in results)
    for name, p in results:
        print(f"{name:<{width}}  {p:.4f}")

    # Optional JSON dump
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "image": str(args.img),
                    "model_name": model_name,
                    "img_size": img_size,
                    "device": device,
                    "tta": bool(args.tta),
                    "topk": results,
                },
                f,
                indent=2,
            )
        print(f"\nSaved JSON to: {args.json_out}")


if __name__ == "__main__":
    main()
