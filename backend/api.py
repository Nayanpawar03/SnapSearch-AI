import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from backend.searcher import search

app = FastAPI(
    title="SnapSearch AI API",
    description="Semantic image search powered by CLIP + FAISS",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"message": "SnapSearch AI is running"}


@app.get("/search/text")
def search_text(
    query: str = Query(..., description="Natural language search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results"),
    label_filter: str = Query(None, description="Filter by label e.g. bird")
):
    """Search images using a text query."""
    try:
        results = search(query, top_k=top_k, mode="text", label_filter=label_filter)
        return {"query": query, "results": results}
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Index not found. Run indexer first."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/search/image")
def search_image(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=50),
    label_filter: str = Query(None)
):
    """Search images using an uploaded query image."""
    try:
        # save upload to a temp file
        suffix = os.path.splitext(file.filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        results = search(tmp_path, top_k=top_k, mode="image", label_filter=label_filter)
        os.unlink(tmp_path)
        return {"filename": file.filename, "results": results}
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Index not found. Run indexer first."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/index")
def index_folder(folder_path: str = Query(..., description="Absolute path to folder to index")):
    """Re-index a folder of images."""
    if not os.path.isdir(folder_path):
        return JSONResponse(status_code=400, content={"error": "Invalid folder path."})
    try:
        from backend.indexer import index_images
        index_images(images_dir=folder_path)
        return {"message": f"Indexed folder: {folder_path}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
