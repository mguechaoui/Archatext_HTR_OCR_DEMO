"""
Arabic Handwritten Line Segmentation 


Usage:
    1-Place the segmentaion model in seg_model/muharaf_seg_best.mlmodel   
    2-Plca your images in the IMAGE_DIR

    MODEL LINK  : https://zenodo.org/records/14295555


    3-Run the command 

    python segment_all.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path

from PIL import Image
from kraken import blla
from kraken.lib import vgsl

# ─────────────────────────────────────────────────────────────────────────────
# ✏️  SET YOUR PATHS HERE
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_DIR   = "./ready_for_kraken_books/pages_validated/book_09/images"
MODEL_PATH  = "seg_model/muharaf_seg_best.mlmodel"
OUTPUT_DIR  = "line_segmentaion_book_09"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ─────────────────────────────────────────────────────────────────────────────


def load_model(model_path):
    print(f"[*] Loading model: {model_path}")
    model = vgsl.TorchVGSLModel.load_model(model_path)
    print("[+] Model ready\n")
    return model


def segment(image, model):
    return blla.segment(image, model=model, text_direction="horizontal-rl")


def visualise(image, result, save_path):
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.imshow(image)
    ax.set_title(f"Lines detected: {len(result.lines)}", fontsize=13)
    ax.axis("off")
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(result.lines), 1)))
    for i, line in enumerate(result.lines):
        c = colors[i % len(colors)]
        if line.boundary:
            pts = np.array(line.boundary)
            poly = Polygon(pts, closed=True, linewidth=1.5,
                           edgecolor=c, facecolor=(*c[:3], 0.15))
            ax.add_patch(poly)
        if line.baseline:
            bl = np.array(line.baseline)
            ax.plot(bl[:, 0], bl[:, 1], color="red", linewidth=1.2, alpha=0.85)
            mid = bl[len(bl) // 2]
            ax.text(mid[0], mid[1] - 8, str(i + 1), fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="navy", alpha=0.75))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [+] Visualisation -> {save_path}")


def build_page_entry(result, image_path):
    return {
        "image": str(image_path),
        "total_lines": len(result.lines),
        "lines": [
            {
                "line_id": i,
                "baseline": line.baseline or [],
                "boundary_polygon": line.boundary or [],
            }
            for i, line in enumerate(result.lines)
        ],
    }


def crop_lines(image, result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.array(image)
    saved = 0
    for i, line in enumerate(result.lines):
        if not line.boundary:
            continue
        pts = np.array(line.boundary)
        x0, y0 = pts.min(axis=0).astype(int)
        x1, y1 = pts.max(axis=0).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(arr.shape[1], x1), min(arr.shape[0], y1)
        crop = arr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        Image.fromarray(crop).save(out_dir / f"line_{i:03d}.png")
        saved += 1
    print(f"    [+] Crops -> {out_dir}/  ({saved} lines)")


def load_combined(combined_path):
    """Load existing combined JSON, return as dict keyed by image path stem."""
    if not combined_path.exists():
        return {}
    try:
        data = json.loads(combined_path.read_text(encoding="utf-8"))
        return {Path(e["image"]).stem: e for e in data}
    except Exception as e:
        print(f"[!] Could not load existing combined JSON: {e} -- starting fresh")
        return {}


def save_combined(combined_path, combined_dict):
    """Write combined dict to disk atomically after every page."""
    all_pages = sorted(combined_dict.values(), key=lambda e: e["image"])
    tmp = combined_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(all_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(combined_path)


def process_image(image_path, model, out_root):
    stem          = Path(image_path).stem
    out_dir       = Path(out_root) / stem
    per_page_json = out_dir / f"{stem}_lines.json"

    # skip if already done
    if per_page_json.exists():
        print(f"    [skip] {image_path.name} -- already processed")
        entry = json.loads(per_page_json.read_text(encoding="utf-8"))
        return stem, entry

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n-- {image_path.name} --")
    image = Image.open(image_path).convert("RGB")
    print(f"    Size: {image.size}")

    result = segment(image, model)
    print(f"    Lines found: {len(result.lines)}")
    entry  = build_page_entry(result, image_path)

    visualise(image, result,  out_dir / f"{stem}_segmented.png")
    per_page_json.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [+] JSON  -> {per_page_json}")
    crop_lines(image, result, out_dir / "line_crops")

    return stem, entry


def main():
    model    = load_model(MODEL_PATH)
    img_dir  = Path(IMAGE_DIR)
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        print(f"[!] No images found in '{IMAGE_DIR}'")
        return

    combined_path = out_root / "all_pages_seg.json"

    # load whatever is already done
    combined = load_combined(combined_path)
    already  = len(combined)
    if already:
        print(f"[*] Resuming -- {already} page(s) already done\n")

    print(f"[*] {len(images)} image(s) total\n")

    failed = []
    for i, img_path in enumerate(images, 1):
        try:
            stem, entry = process_image(img_path, model, out_root)
            combined[stem] = entry
            save_combined(combined_path, combined)
            print(f"    [combined] {len(combined)}/{len(images)} pages saved")
        except Exception as e:
            print(f"    [!] Failed: {img_path.name} -- {e}")
            failed.append(img_path.name)

    print(f"\n[done] {len(combined)} pages -> {combined_path}")
    if failed:
        print(f"[!] {len(failed)} failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()