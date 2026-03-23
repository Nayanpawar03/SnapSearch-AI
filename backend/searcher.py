import os
import json
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ── config ───────────────────────────────────────────────
INDEX_PATH = "embeddings/index.faiss"
META_PATH  = "embeddings/metadata.json"
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────

_model     = None
_processor = None

def load_model():
    global _model, _processor
    if _model is None:
        print(f"Loading CLIP on {DEVICE}...")
        _model     = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


def load_index():
    index    = faiss.read_index(INDEX_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)
    return index, metadata


def _normalize(tensor):
    return tensor / torch.norm(tensor, dim=-1, keepdim=True)


def get_text_embedding(query: str):
    model, processor = load_model()
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        output    = model.text_model(**inputs)
        embedding = model.text_projection(output.pooler_output)
    embedding = _normalize(embedding)
    return embedding.cpu().numpy().astype("float32")


def get_image_embedding(image_path: str):
    model, processor = load_model()
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output    = model.vision_model(**inputs)
        embedding = model.visual_projection(output.pooler_output)
    embedding = _normalize(embedding)
    return embedding.cpu().numpy().astype("float32")


def search(query, top_k=5, mode="text", label_filter=None):
    """
    Search the FAISS index.

    Args:
        query     : str (text query) or str (image path) depending on mode
        top_k     : number of results to return
        mode      : "text" or "image"
        label_filter : optional string to filter results by label

    Returns:
        list of dicts: [{"path": ..., "label": ..., "score": ...}, ...]
    """
    index, metadata = load_index()

    if mode == "text":
        query_vec = get_text_embedding(query)
    elif mode == "image":
        query_vec = get_image_embedding(query)
    else:
        raise ValueError("mode must be 'text' or 'image'")

    # FAISS search — returns (scores, indices)
    scores, indices = index.search(query_vec, top_k * 3)  # fetch extra for filtering

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        if label_filter and meta.get("label") != label_filter:
            continue
        results.append({
            "path":  meta["path"],
            "label": meta.get("label"),
            "score": float(score)
        })
        if len(results) == top_k:
            break

    return results


# ── quick CLI test ────────────────────────────────────────
if __name__ == "__main__":
    query = input("Enter search query: ")
    hits  = search(query, top_k=5, mode="text")
    print(f"\nTop {len(hits)} results:")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. [{h['score']:.4f}] {h['path']}  (label: {h['label']})")
