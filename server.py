import io
import json
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch, timm
from torchvision import transforms
from fastapi.staticfiles import StaticFiles

CKPT_PATH = "artifacts/resnet18_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="What Is This Bird")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("species_mapping.json") as f:
    common_to_scientific = json.load(f)

def normalize_label(raw: str) -> str:
    parts = raw.split(".", 1)
    if len(parts) == 2:
        raw = parts[1]

    return raw.replace("_", " ")


# ----- Load model once at startup -----
def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt.get("model_name", "convnext_tiny.fb_in1k")
    img_size = ckpt.get("img_size", 224)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(DEVICE)

    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return model, tf, idx_to_class, img_size, model_name

model, tf, idx_to_class, IMG_SIZE, MODEL_NAME = load_checkpoint(CKPT_PATH)

# optional: memory/throughput niceties on CUDA
if DEVICE == "cuda":
    try: model = model.to(memory_format=torch.channels_last)
    except Exception: pass

# ----- Schemas -----
class PredItem(BaseModel):
    label: str
    prob: float

class PredResponse(BaseModel):
    model: str
    img_size: int
    topk: List[PredItem]

# ----- Routes -----
@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_NAME, "img_size": IMG_SIZE, "device": DEVICE}

@app.post("/predict", response_model=PredResponse)
def predict(file: UploadFile = File(...), topk: int = 5, tta: bool = False):
    if file.content_type not in {"image/jpeg","image/png","image/webp","image/bmp"}:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    if topk <= 0:
        raise HTTPException(status_code=400, detail="topk must be > 0")

    try:
        image_bytes = file.file.read()
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")

    @torch.no_grad()
    def _run(img: Image.Image):
        x = tf(img).unsqueeze(0).to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
        return probs

    probs = _run(im)
    if tta:
        probs = 0.5 * (probs + _run(im.transpose(Image.FLIP_LEFT_RIGHT)))

    k = min(topk, probs.numel())
    vals, idxs = probs.topk(k)
    items = []
    for v, i in zip(vals, idxs):
        raw_label = idx_to_class[i.item()]
        clean_common = normalize_label(raw_label)
        sci = common_to_scientific.get(clean_common, clean_common)
        items.append(PredItem(label=f"{clean_common} ({sci})", prob=float(v.item())))

    return PredResponse(model=MODEL_NAME, img_size=IMG_SIZE, topk=items)

app.mount("/", StaticFiles(directory="static", html=True), name="static")
