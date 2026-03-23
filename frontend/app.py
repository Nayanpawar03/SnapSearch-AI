import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from PIL import Image
import io
import json
from backend.searcher import search
from backend.indexer  import index_images

st.set_page_config(page_title="SnapSearch AI", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d0f18; }

h1 {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin-bottom: 0 !important;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: #161824; border-radius: 12px; padding: 5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #9ca3af; font-weight: 600; padding: 8px 22px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
}

/* input */
.stTextInput input {
    background: #161824 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 10px !important;
    color: #f9fafb !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
}
.stTextInput input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
}

/* search button */
div[data-testid="stButton"] > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white; border: none; border-radius: 10px;
    padding: 10px 32px; font-weight: 600; font-size: 0.95rem;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124,58,237,0.4);
}
.stButton > button:disabled {
    background: #1f2235 !important; color: #4b5563 !important;
    transform: none !important; box-shadow: none !important;
}

/* featured image panel */
.featured-panel {
    background: #161824;
    border: 1px solid #2d3148;
    border-radius: 16px;
    padding: 16px;
    position: relative;
}
.featured-panel img { border-radius: 10px; width: 100%; }

/* thumbnail strip */
.thumb-strip {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    overflow-x: auto;
    padding-bottom: 4px;
}
.thumb-strip::-webkit-scrollbar { height: 4px; }
.thumb-strip::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 4px; }

.thumb {
    flex: 0 0 72px;
    height: 72px;
    border-radius: 8px;
    overflow: hidden;
    border: 2px solid transparent;
    cursor: pointer;
    transition: border-color 0.2s, transform 0.2s;
}
.thumb.active { border-color: #7c3aed; transform: scale(1.08); }
.thumb img { width: 100%; height: 100%; object-fit: cover; }

/* score badge */
.score-badge {
    display: inline-block;
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.4);
    color: #a78bfa; border-radius: 6px;
    padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
}

/* result info */
.result-info {
    color: #9ca3af; font-size: 0.82rem; margin-top: 6px; line-height: 1.6;
}

/* nav buttons */
.nav-btn-row { display: flex; align-items: center; gap: 10px; margin-top: 14px; }

/* page dots */
.dots { color: #4b5563; font-size: 0.8rem; flex: 1; text-align: center; }
.dots .active-dot { color: #a78bfa; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: #13151f;
    border-right: 1px solid #1f2235;
}

/* fade animation */
@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.97); }
    to   { opacity: 1; transform: scale(1); }
}
.fade-in { animation: fadeIn 0.3s ease forwards; }

/* download button override */
.stDownloadButton > button {
    background: #1f2235 !important;
    border: 1px solid #7c3aed !important;
    color: #a78bfa !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    padding: 6px 16px !important;
    width: 100%;
}
.stDownloadButton > button:hover {
    background: rgba(124,58,237,0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# ── title ─────────────────────────────────────────────────
st.markdown("<h1>SnapSearch AI</h1>", unsafe_allow_html=True)
st.caption("Semantic image search powered by CLIP + FAISS")

# ── sidebar ───────────────────────────────────────────────
st.sidebar.header("⚙️ Index a Folder")
folder_path = st.sidebar.text_input("Folder path", placeholder="e.g. C:/Users/you/Pictures")
if st.sidebar.button("Index Folder"):
    if not folder_path or not os.path.isdir(folder_path):
        st.sidebar.error("Invalid folder path.")
    else:
        with st.spinner("Indexing..."):
            index_images(images_dir=folder_path)
        st.sidebar.success("Index updated!")

st.sidebar.markdown("---")
top_k     = st.sidebar.slider("Total results", 1, 50, 10)
page_size = st.sidebar.slider("Images per page", 1, 10, 5)

# ── label filter ──────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**🏷️ Filter by Label**")
_labels_path = "models/class_labels.json"
if os.path.exists(_labels_path):
    with open(_labels_path) as f:
        all_labels = json.load(f)
    selected_labels = st.sidebar.multiselect(
        "Show only these labels",
        options=all_labels,
        default=[],
        placeholder="All labels (no filter)"
    )
else:
    selected_labels = []
    st.sidebar.caption("Train a classifier to enable label filtering.")


# ── image bytes helper for download ──────────────────────
def get_image_bytes(path):
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def apply_label_filter(results):
    """Filter results by selected labels. If none selected, return all."""
    if not selected_labels:
        return results
    return [r for r in results if r.get("label") in selected_labels]


# ── gallery layout ────────────────────────────────────────
def show_gallery(results, key_prefix):
    if not results:
        st.info("No results found.")
        return

    total_pages = max(1, (len(results) + page_size - 1) // page_size)
    page_key    = f"{key_prefix}_page"
    sel_key     = f"{key_prefix}_sel"    # selected index within page
    dir_key     = f"{key_prefix}_dir"

    for k, default in [(page_key, 0), (sel_key, 0), (dir_key, "fwd")]:
        if k not in st.session_state:
            st.session_state[k] = default

    st.session_state[page_key] = min(st.session_state[page_key], total_pages - 1)
    current_page = st.session_state[page_key]

    start        = current_page * page_size
    end          = min(start + page_size, len(results))
    page_results = results[start:end]

    # clamp selected index to current page size
    st.session_state[sel_key] = min(st.session_state[sel_key], len(page_results) - 1)
    selected_idx = st.session_state[sel_key]
    selected_hit = page_results[selected_idx]

    # ── two-column layout: left = query panel, right = featured ──
    left, right = st.columns([1, 1], gap="large")

    with right:
        st.markdown('<div class="featured-panel fade-in">', unsafe_allow_html=True)

        try:
            img = Image.open(selected_hit["path"])
            st.image(img, use_container_width=True)
        except Exception:
            st.warning("Cannot load image.")

        # info row
        fname = os.path.basename(selected_hit["path"])
        label = selected_hit["label"] or "N/A"
        st.markdown(
            f'<div style="margin-top:8px;">'
            f'<span class="score-badge">score {selected_hit["score"]:.3f}</span>'
            f'<div class="result-info">🏷️ Label: {label}<br>📄 {fname}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # download button
        try:
            img_bytes = get_image_bytes(selected_hit["path"])
            st.download_button(
                label="⬇️ Download Image",
                data=img_bytes,
                file_name=fname,
                mime="image/png",
                key=f"{key_prefix}_dl_{current_page}_{selected_idx}"
            )
        except Exception:
            pass

        # thumbnail strip using columns
        st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
        thumb_cols = st.columns(len(page_results))
        for i, hit in enumerate(page_results):
            with thumb_cols[i]:
                try:
                    t = Image.open(hit["path"])
                    border = "2px solid #7c3aed" if i == selected_idx else "2px solid #2d3148"
                    st.markdown(
                        f'<div style="border-radius:8px;overflow:hidden;border:{border};'
                        f'transition:border-color 0.2s;">',
                        unsafe_allow_html=True
                    )
                    st.image(t, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    if st.button("▲", key=f"{key_prefix}_thumb_{current_page}_{i}",
                                 help=os.path.basename(hit["path"])):
                        st.session_state[sel_key] = i
                        st.rerun()
                except Exception:
                    pass
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── left column: result count + nav ──────────────────
    with left:
        st.markdown(
            f'<div style="background:#161824;border:1px solid #2d3148;border-radius:12px;'
            f'padding:14px 18px;margin-bottom:16px;">'
            f'<span style="color:#a78bfa;font-weight:600;">🎯 {len(results)} results found</span>'
            f'<div style="color:#6b7280;font-size:0.82rem;margin-top:4px;">'
            f'Page {current_page+1} of {total_pages} &nbsp;·&nbsp; '
            f'Showing {start+1}–{end}</div></div>',
            unsafe_allow_html=True
        )

        # prev / next
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("◀ Prev", key=f"{key_prefix}_prev",
                         disabled=(current_page == 0)):
                st.session_state[page_key] -= 1
                st.session_state[sel_key]   = 0
                st.session_state[dir_key]   = "back"
                st.rerun()
        with c2:
            # dot indicators
            dots_html = ""
            for p in range(total_pages):
                color = "#a78bfa" if p == current_page else "#2d3148"
                dots_html += f'<span style="color:{color};font-size:1.1rem;">⬤</span> '
            st.markdown(
                f'<div style="text-align:center;padding-top:6px;">{dots_html}</div>',
                unsafe_allow_html=True
            )
        with c3:
            if st.button("Next ▶", key=f"{key_prefix}_next",
                         disabled=(current_page >= total_pages - 1)):
                st.session_state[page_key] += 1
                st.session_state[sel_key]   = 0
                st.session_state[dir_key]   = "fwd"
                st.rerun()

        # all results list
        st.markdown(
            '<div style="color:#6b7280;font-size:0.78rem;margin-top:20px;'
            'margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em;">'
            'All results on this page</div>',
            unsafe_allow_html=True
        )
        for i, hit in enumerate(page_results):
            is_sel = i == selected_idx
            bg     = "#1f2235" if is_sel else "#161824"
            border = "#7c3aed" if is_sel else "#2d3148"
            if st.button(
                f"{'▶ ' if is_sel else '   '}{os.path.basename(hit['path'])}  [{hit['score']:.3f}]",
                key=f"{key_prefix}_list_{current_page}_{i}"
            ):
                st.session_state[sel_key] = i
                st.rerun()


# ── tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Text Search", "Image Search"])

with tab1:
    text_query = st.text_input(
        "Describe what you're looking for",
        placeholder="e.g. poster about operating systems, meme about AI",
        key="text_input"
    )
    if st.button("Search", key="text_search"):
        if not text_query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching..."):
                try:
                    st.session_state["text_results"] = search(text_query, top_k=top_k, mode="text")
                    st.session_state["text_page"] = 0
                    st.session_state["text_sel"]  = 0
                except FileNotFoundError:
                    st.error("No index found. Index a folder first.")
                    st.session_state["text_results"] = []

    if "text_results" in st.session_state:
        show_gallery(apply_label_filter(st.session_state["text_results"]), key_prefix="text")

with tab2:
    uploaded = st.file_uploader("Upload a query image", type=["jpg","jpeg","png","bmp","webp"])
    if uploaded and st.button("Search", key="image_search"):
        tmp_path = "embeddings/_query_tmp.png"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        with st.spinner("Searching..."):
            try:
                st.session_state["image_results"] = search(tmp_path, top_k=top_k, mode="image")
                st.session_state["image_page"] = 0
                st.session_state["image_sel"]  = 0
            except FileNotFoundError:
                st.error("No index found. Index a folder first.")
                st.session_state["image_results"] = []

    if "image_results" in st.session_state:
        show_gallery(apply_label_filter(st.session_state["image_results"]), key_prefix="image")
