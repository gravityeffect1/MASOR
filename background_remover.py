from pathlib import Path
import argparse
import shutil
import traceback
import cv2
import numpy as np
import pandas as pd
import openslide
from skimage.filters import threshold_otsu
from PIL import Image

SUPPORTED_EXTS = {".svs", ".png", ".tif", ".tiff", ".ndpi", ".mrxs"}

MANIFEST_COLUMNS = ["slide_id", "slide_path", "image_path", "x0", "y0",
                    "target_level", "mask_level", "tile_size",
                    "tissue_frac", "blur_score", "keep"]


def collect_slides(input_path):
    p = Path(input_path)
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTS)
    return [p]


def choose_thumbnail_level(slide, max_dim=2048):
    for level in range(slide.level_count):
        w, h = slide.level_dimensions[level]
        if max(w, h) <= max_dim:
            return level
    return slide.level_count - 1


def get_target_level(slide, target_mpp):
    try:
        base_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    except Exception:
        return 0

    mpps = [base_mpp * float(ds) for ds in slide.level_downsamples]
    return int(np.argmin([abs(mpp - target_mpp) for mpp in mpps]))


def compute_tissue_mask(slide, level):
    w, h = slide.level_dimensions[level]
    rgb = np.asarray(slide.read_region((0, 0), level, (w, h)).convert("RGB"))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    sat_thr = threshold_otsu(sat)
    tissue = (sat > sat_thr) & (val < 245)

    tissue = tissue.astype(np.uint8)

    k_open = 3
    k_close = max(5, int(round(0.003 * max(w, h))))
    if k_close % 2 == 0:
        k_close += 1

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))

    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, open_kernel)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, close_kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, connectivity=8)
    clean = np.zeros_like(tissue)

    min_area = 0.001 * w * h
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 1

    return clean, w, h


def laplacian_blur_score(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def generate_tiles(slide, tissue_mask, mask_w, mask_h, mask_level, target_level,
                   tile_size=512, overlap_frac=0.0, min_tissue_frac=0.25):
    W0, H0 = slide.dimensions
    ds_mask = float(slide.level_downsamples[mask_level])
    ds_target = float(slide.level_downsamples[target_level])

    tile_level0 = int(round(tile_size * ds_target))
    tile_mask = tile_level0 / ds_mask
    step_mask = max(1, tile_mask * (1 - overlap_frac))

    tiles = []
    y = 0.0

    while y + tile_mask <= mask_h:
        x = 0.0
        while x + tile_mask <= mask_w:
            x0m, y0m = int(round(x)), int(round(y))
            x1m = int(round(x + tile_mask))
            y1m = int(round(y + tile_mask))

            tissue_frac = float(tissue_mask[y0m:y1m, x0m:x1m].mean())

            if tissue_frac >= min_tissue_frac:
                x0 = int(round(x0m * ds_mask))
                y0 = int(round(y0m * ds_mask))

                if x0 + tile_level0 <= W0 and y0 + tile_level0 <= H0:
                    tiles.append((x0, y0, tissue_frac))

            x += step_mask
        y += step_mask

    return tiles


def process_slide(svs_path, output_root, target_mpp=0.5, tile_size=512,
                  overlap_frac=0.0, min_tissue_frac=0.25, min_blur_score=100.0):
    svs_path = Path(svs_path)
    slide_id = svs_path.stem

    output_root = Path(output_root)
    image_dir = output_root / "images"
    qc_dir = output_root / "qc"
    image_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(str(svs_path))

    mask_level = choose_thumbnail_level(slide)
    target_level = get_target_level(slide, target_mpp)

    tissue_mask, mask_w, mask_h = compute_tissue_mask(slide, mask_level)

    Image.fromarray((tissue_mask * 255).astype(np.uint8)).save(
        qc_dir / f"{slide_id}_tissue_mask.png"
    )

    tiles = generate_tiles(
        slide, tissue_mask, mask_w, mask_h, mask_level, target_level,
        tile_size=tile_size,
        overlap_frac=overlap_frac,
        min_tissue_frac=min_tissue_frac,
    )

    rows = []

    for idx, (x0, y0, tissue_frac) in enumerate(tiles):
        rgb = np.asarray(
            slide.read_region((x0, y0), target_level, (tile_size, tile_size)).convert("RGB")
        )

        blur_score = laplacian_blur_score(rgb)
        keep = blur_score >= min_blur_score

        out_name = f"{slide_id}_x{x0}_y{y0}_l{target_level}.png"
        out_path = image_dir / out_name

        if keep:
            Image.fromarray(rgb).save(out_path)

        rows.append({
            "slide_id": slide_id,
            "slide_path": str(svs_path),
            "image_path": str(out_path) if keep else "",
            "x0": x0,
            "y0": y0,
            "target_level": target_level,
            "mask_level": mask_level,
            "tile_size": tile_size,
            "tissue_frac": tissue_frac,
            "blur_score": blur_score,
            "keep": int(keep),
        })

    slide.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--output", default="output")
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--min-tissue-frac", type=float, default=0.25)
    parser.add_argument("--min-blur-score", type=float, default=100.0)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    slides = collect_slides(args.input_path)

    all_rows = []
    errors = []

    for svs_path in slides:
        print(f"processing: {svs_path}")
        try:
            rows = process_slide(
                svs_path,
                output,
                target_mpp=args.target_mpp,
                tile_size=args.tile_size,
                overlap_frac=args.overlap,
                min_tissue_frac=args.min_tissue_frac,
                min_blur_score=args.min_blur_score,
            )
            all_rows.extend(rows)
            print(f"  candidate tiles: {len(rows)}")
        except Exception as e:
            print(f"  failed: {e}")
            errors.append({
                "slide_path": str(svs_path),
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })

    pd.DataFrame(all_rows, columns=MANIFEST_COLUMNS).to_csv(output / "manifest.csv", index=False)
    pd.DataFrame(errors).to_csv(output / "errors.csv", index=False)

    print("done")
    print("slides:", len(slides))
    print("tiles kept:", sum(r["keep"] for r in all_rows))
    print("manifest:", output / "manifest.csv")


if __name__ == "__main__":
    main()