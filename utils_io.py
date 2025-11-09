from pathlib import Path

def load_image_paths(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(p)
    paths.sort()
    return paths