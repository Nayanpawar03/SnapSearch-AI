import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── config ────────────────────────────────────────────────
CLIP_MODEL  = "openai/clip-vit-base-patch32"
SBERT_MODEL = "all-MiniLM-L6-v2"       # fast, lightweight sentence transformer
META_PATH   = "embeddings/metadata.json"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────

_clip_model = _clip_processor = _sbert_model = None

def _load_models():
    global _clip_model, _clip_processor, _sbert_model
    if _clip_model is None:
        print("Loading CLIP...")
        _clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    if _sbert_model is None:
        print("Loading Sentence Transformer...")
        _sbert_model = SentenceTransformer(SBERT_MODEL)


def clip_text_embedding(text: str) -> np.ndarray:
    _load_models()
    inputs = _clip_processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = _clip_model.text_model(**inputs)
        emb = _clip_model.text_projection(out.pooler_output)
    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def sbert_embedding(text: str) -> np.ndarray:
    _load_models()
    emb = _sbert_model.encode([text], normalize_embeddings=True)
    return emb.astype("float32")


def compare_queries(query_a: str, query_b: str):
    """
    Compare two text queries using both CLIP and SBERT.
    Returns cosine similarity scores from each model.
    """
    _load_models()

    clip_a  = clip_text_embedding(query_a)
    clip_b  = clip_text_embedding(query_b)
    sbert_a = sbert_embedding(query_a)
    sbert_b = sbert_embedding(query_b)

    clip_sim  = float(cosine_similarity(clip_a,  clip_b)[0][0])
    sbert_sim = float(cosine_similarity(sbert_a, sbert_b)[0][0])

    return {
        "query_a":    query_a,
        "query_b":    query_b,
        "clip_similarity":  round(clip_sim,  4),
        "sbert_similarity": round(sbert_sim, 4),
        "interpretation": _interpret(clip_sim, sbert_sim)
    }


def _interpret(clip_sim, sbert_sim):
    diff = abs(clip_sim - sbert_sim)
    if diff < 0.05:
        return "Both models agree on semantic similarity."
    elif clip_sim > sbert_sim:
        return "CLIP sees stronger visual-semantic overlap; SBERT focuses more on linguistic meaning."
    else:
        return "SBERT sees stronger linguistic similarity; CLIP may interpret these differently visually."


def rank_labels_for_query(query: str):
    """
    Given a query, rank all known image labels by semantic similarity
    using both CLIP and SBERT side by side.
    """
    if not os.path.exists(META_PATH):
        return []

    with open(META_PATH) as f:
        metadata = json.load(f)

    labels = list({m["label"] for m in metadata if m.get("label")})
    if not labels:
        return []

    _load_models()
    query_clip  = clip_text_embedding(query)
    query_sbert = sbert_embedding(query)

    results = []
    for label in labels:
        lc = clip_text_embedding(label)
        ls = sbert_embedding(label)
        results.append({
            "label":            label,
            "clip_score":       round(float(cosine_similarity(query_clip,  lc)[0][0]), 4),
            "sbert_score":      round(float(cosine_similarity(query_sbert, ls)[0][0]), 4),
        })

    results.sort(key=lambda x: x["clip_score"], reverse=True)
    return results


if __name__ == "__main__":
    print("=== Query Comparison ===")
    r = compare_queries("bird flying in sky", "eagle soaring")
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n=== Label Ranking for 'flying animal' ===")
    ranks = rank_labels_for_query("flying animal")
    for r in ranks:
        print(f"  {r['label']:15s}  CLIP: {r['clip_score']:.4f}  SBERT: {r['sbert_score']:.4f}")
