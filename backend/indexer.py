import os
import json
import numpy as np
import faiss
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms, models

# ── config ──────────────────────────────────────────────
IMAGES_DIR   = "dataset"
INDEX_PATH   = "embeddings/index.faiss"
META_PATH    = "embeddings/metadata.json"
MODEL_PATH   = "models/resnet18_classifier.pth"
LABELS_PATH  = "models/class_labels.json"
CLIP_MODEL   = "openai/clip-vit-base-patch32"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────

def load_clip():
    print(f"Loading CLIP on {DEVICE}...")
    model     = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return model, processor


def load_classifier():
    """Load trained ResNet18 classifier if available."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("No classifier found — labels will be null.")
        return None, []

    with open(LABELS_PATH) as f:
        class_names = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(f"Classifier loaded — classes: {class_names}")
    return model, class_names


def get_image_embedding(image_path, clip_model, processor):
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output    = clip_model.vision_model(**inputs)
        embedding = clip_model.visual_projection(output.pooler_output)
    embedding = embedding / torch.norm(embedding, dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype("float32")


_clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict_label(image_path, clf_model, class_names):
    """Return predicted class name for one image."""
    if clf_model is None:
        return None
    image  = Image.open(image_path).convert("RGB")
    tensor = _clf_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = clf_model(tensor)
    idx = logits.argmax(1).item()
    return class_names[idx]


def index_images(images_dir=IMAGES_DIR):
    clip_model, processor          = load_clip()
    clf_model,  class_names        = load_classifier()

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # collect images — works for flat folder OR subfolders
    image_paths = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in supported:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print("No images found in", images_dir)
        return

    print(f"\nFound {len(image_paths)} images. Indexing...\n")

    embeddings, metadata = [], []

    for path in image_paths:
        try:
            emb   = get_image_embedding(path, clip_model, processor)
            label = predict_label(path, clf_model, class_names)
            embeddings.append(emb)
            metadata.append({"path": path, "label": label})
            print(f"  ✓ [{label or 'N/A':12s}]  {os.path.basename(path)}")
        except Exception as e:
            print(f"  ✗ Skipped {path}: {e}")

    if not embeddings:
        print("No embeddings generated.")
        return

    embeddings_matrix = np.vstack(embeddings)
    dim   = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)

    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Indexed {len(metadata)} images.")
    print(f"  FAISS index → {INDEX_PATH}")
    print(f"  Metadata    → {META_PATH}")


if __name__ == "__main__":
    index_images()
