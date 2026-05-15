# SnapSearch AI — Local Image Search Engine

## What is this?

SnapSearch AI is a local image search engine that lets you search through images on your computer using natural language or another image, no internet required, no cloud APIs, everything runs on your machine.

Think of it like Google Image Search, but for your own folders. You can type something like *"poster about operating systems"* or *"bird flying at night"* and the system will find the most visually and semantically relevant images from your indexed folder, even if those exact words don't appear anywhere in the filenames.

---

## Why does this exist?

Most people have thousands of images scattered across their computer with no good way to find them. Filename search is useless unless you named every file perfectly. Folder browsing is slow. Cloud-based tools require uploading your private images.

SnapSearch AI solves this by:
- Understanding the **meaning** of your query, not just keywords
- Running **entirely offline** on your local machine
- Supporting both **text queries** and **image queries**
- Automatically **classifying and labeling** images using a trained CNN
- Providing a clean **web UI** and a **REST API**

---

## How it works

```
Your Images
    │
    ▼
CLIP Vision Model ──► 512-d Embedding Vector
    │
    ▼
FAISS Vector Database (stored locally)
    │
    ▼
Query (text or image) ──► CLIP Embedding ──► Cosine Similarity Search
    │
    ▼
Top-K Matching Images
```

- **CLIP** (Contrastive Language-Image Pretraining by OpenAI) maps both images and text into the same vector space, enabling semantic search
- **FAISS** (Facebook AI Similarity Search) stores all image embeddings and performs fast nearest-neighbor search
- **ResNet18** CNN classifier automatically labels each image with a category during indexing
- **Sentence Transformers** provides an additional NLP layer for query analysis and comparison

---

## Features

- Natural language search — *"meme about machine learning"*, *"selfie with monkey"*
- Image-to-image search — upload an image, find visually similar ones
- Auto-labeling of images using a trained ResNet18 classifier
- Label-based filtering of search results
- Paginated carousel UI with animations and download support
- REST API via FastAPI with Swagger docs
- NLP analysis tab comparing CLIP vs Sentence Transformers

---

## Project Structure

```
SnapSearch-AI/
├── dataset/                  # Training images (organized in class subfolders)
│   ├── bird/
│   ├── certificate/
│   ├── donkey/
│   ├── forest/
│   └── horse/
├── embeddings/
│   ├── index.faiss           # FAISS vector index
│   └── metadata.json         # Image paths + labels
├── models/
│   ├── resnet18_classifier.pth
│   └── class_labels.json
├── backend/
│   ├── indexer.py            # Indexes images → CLIP embeddings → FAISS
│   ├── searcher.py           # Text/image search using FAISS
│   ├── trainer.py            # Trains ResNet18 classifier
│   ├── nlp_compare.py        # CLIP vs Sentence Transformers comparison
│   └── api.py                # FastAPI REST API
├── frontend/
│   └── app.py                # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9+
- CPU is sufficient (GPU optional for faster inference)

---

## Setup & Installation

### 1. Clone or download the project

```bash
cd SnapSearch-AI
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Train the CNN classifier

Organize your training images into class subfolders inside `dataset/`:

```
dataset/
├── bird/        ← put bird images here
├── forest/      ← put forest images here
└── ...
```

Then run:

```bash
python backend/trainer.py
```

This trains ResNet18 and saves the model to `models/`.

---

### Step 2 — Index your images

Point the indexer at any folder of images. By default it uses `dataset/`:

```bash
python backend/indexer.py
```

Or index a custom folder by editing `IMAGES_DIR` in `indexer.py`.

This generates `embeddings/index.faiss` and `embeddings/metadata.json`.

---

### Step 3 — Launch the Streamlit UI

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

From the UI you can:
- Search images using text queries
- Upload an image to find similar ones
- Index any folder on your computer from the sidebar
- Filter results by label
- Download any result image
- Compare CLIP vs Sentence Transformers in the Analysis tab

---

### Step 4 — Launch the REST API (optional)

```bash
uvicorn backend.api:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/search/text?query=...&top_k=5` | Text-to-image search |
| POST | `/search/image` | Image-to-image search (upload file) |
| POST | `/index?folder_path=...` | Index a folder |

---

## Example Queries

| Query | What it finds |
|-------|--------------|
| `bird flying at night` | Dark bird photos |
| `green trees and nature` | Forest images |
| `official document with text` | Certificates |
| `animal with hooves` | Horse or donkey images |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | CLIP (openai/clip-vit-base-patch32) |
| Vector DB | FAISS (faiss-cpu) |
| CNN Classifier | ResNet18 (torchvision) |
| NLP Comparison | Sentence Transformers (all-MiniLM-L6-v2) |
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| Deep Learning | PyTorch |
