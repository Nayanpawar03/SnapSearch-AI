import os
import json
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ── config ──────────────────────────────────────────────
IMAGES_DIR   = "dataset"
INDEX_PATH   = "embeddings/index.faiss"
META_PATH    = "embeddings/metadata.json"
MODEL_NAME   = "openai/clip-vit-base-patch32"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────

def load_model():
    print(f"Loading CLIP model on {DEVICE}...")
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def get_image_embedding(image_path, model, processor):
    """Return a normalized 512-d embedding for one image."""
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.vision_model(**inputs)
        # pooler_output is the [CLS] token embedding → shape (1, 768 or 512)
        embedding = model.visual_projection(output.pooler_output)
    # normalize to unit vector so inner product == cosine similarity
    embedding = embedding / torch.norm(embedding, dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype("float32")


def index_images(images_dir=IMAGES_DIR):
    model, processor = load_model()

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not image_paths:
        print("No images found in", images_dir)
        return

    print(f"Found {len(image_paths)} images. Generating embeddings...")

    embeddings = []
    metadata   = []

    for path in image_paths:
        try:
            emb = get_image_embedding(path, model, processor)
            embeddings.append(emb)
            metadata.append({"path": path, "label": None})
            print(f"  ✓ {os.path.basename(path)}")
        except Exception as e:
            print(f"  ✗ Skipped {path}: {e}")

    # stack into (N, 512) matrix
    embeddings_matrix = np.vstack(embeddings)

    # build FAISS index (inner product = cosine similarity on normalized vectors)
    dim   = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)

    # save
    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Indexed {len(metadata)} images.")
    print(f"  FAISS index → {INDEX_PATH}")
    print(f"  Metadata    → {META_PATH}")


if __name__ == "__main__":
    index_images()
