#streamlit run app.py
#http://localhost:8501

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from search_core import ImageSearchEngine

st.set_page_config(page_title="Image Search (CLIP)", layout="wide")
st.title("🔎 Image Search Engine — CLIP")


index_dir = st.sidebar.text_input("Index directory", value="index")
model_name = st.sidebar.text_input("Model name", value="clip-ViT-B-32")

@st.cache_resource(show_spinner=False)
def load_engine(index_dir: str, model_name: str):
    return ImageSearchEngine(index_dir=index_dir, model_name=model_name)

try:
    engine = load_engine(index_dir, model_name)
    st.success("Index loaded.")
except Exception as e:
    st.error(f"Failed to load index: {e}")
    st.stop()

tab1, tab2 = st.tabs(["Text query", "Image query"])

with tab1:
    query = st.text_input("Describe what you want to find (e.g., 'a red car', 'a dog running')")
    k = st.slider("Top-K results", 1, 30, 10)
    if st.button("Search by text") and query.strip():
        results = engine.search_by_text(query, k=k)
        cols = st.columns(5)
        for i, (path, score) in enumerate(results):
            with cols[i % 5]:
                st.image(str(path), caption=f"cosine={score:.3f}", use_column_width=True)

with tab2:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    k2 = st.slider("Top-K results (image)", 1, 30, 10, key="k2")
    if uploaded is not None:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(img, caption="Query image", use_column_width=False)
        if st.button("Search by image"):
            results = engine.search_by_image(img, k=k2)
            cols = st.columns(5)
            for i, (path, score) in enumerate(results):
                with cols[i % 5]:
                    st.image(str(path), caption=f"cosine={score:.3f}", use_column_width=True)