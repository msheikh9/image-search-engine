import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utils_io import load_image_paths


def embed_images(model, image_paths, batch_size=32):
    vectors = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
            except Exception:
                batch_imgs.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
                
        batch_vecs = model.encode(batch_imgs, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        vectors.append(batch_vecs)
    return np.vstack(vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="index")
    parser.add_argument("--model_name", type=str, default="clip-ViT-B-32")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading image paths…")
    image_paths = load_image_paths(images_dir)
    print(f"Found {len(image_paths)} images")

    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    print("Computing embeddings…")
    vectors = embed_images(model, image_paths)

    np.save(out_dir / "vectors.npy", vectors)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({i: str(p) for i, p in enumerate(image_paths)}, f)

    print("Done. Saved vectors.npy and meta.json.")


if __name__ == "__main__":
    main()