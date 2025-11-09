import json
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


class ImageSearchEngine:
    def __init__(self, index_dir: str, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.vectors = np.load(f"{index_dir}/vectors.npy")
        with open(f"{index_dir}/meta.json", "r") as f:
            meta = json.load(f)
        self.paths = [meta[str(i)] for i in range(len(meta))]
        
        self.nn = NearestNeighbors(n_neighbors=20, metric="cosine")
        self.nn.fit(self.vectors)

    def search_by_text(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        dist, idx = self.nn.kneighbors(q, n_neighbors=k, return_distance=True)
        return [(self.paths[i], float(d)) for i, d in zip(idx[0], dist[0])]

    def search_by_image(self, img: Image.Image, k: int = 10) -> List[Tuple[str, float]]:
        vec = self.model.encode([img.convert("RGB")], convert_to_numpy=True, normalize_embeddings=True)
        dist, idx = self.nn.kneighbors(vec, n_neighbors=k, return_distance=True)
        return [(self.paths[i], float(d)) for i, d in zip(idx[0], dist[0])]