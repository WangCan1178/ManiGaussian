import argparse
import os
from pathlib import Path

import numpy as np


def save_cache(out_path, clip_feat, dino_feat):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, clip_feat=clip_feat, dino_feat=dino_feat)


def main():
    """
    Cache format (per frame):
      - .npz with keys: clip_feat (H,W,D_clip), dino_feat (H,W,D_dino)

    Example:
      python scripts/generate_teacher_cache.py \
        --input_dir /path/to/frames \
        --output_dir /path/to/cache \
        --max_images 8 \
        --clip_dim 512 \
        --dino_dim 1024 \
        --feat_height 32 \
        --feat_width 32 \
        --dummy
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--clip_dim", type=int, default=512)
    parser.add_argument("--dino_dim", type=int, default=1024)
    parser.add_argument("--feat_height", type=int, default=32)
    parser.add_argument("--feat_width", type=int, default=32)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    image_paths = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    image_paths = image_paths[: args.max_images]

    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    if not args.dummy:
        raise SystemExit(
            "This is a placeholder cache generator. "
            "Run with --dummy to create random features, or replace "
            "this script with CLIP/DINO feature extraction."
        )

    for image_path in image_paths:
        clip_feat = np.random.randn(args.feat_height, args.feat_width, args.clip_dim).astype(np.float32)
        dino_feat = np.random.randn(args.feat_height, args.feat_width, args.dino_dim).astype(np.float32)
        out_name = image_path.stem + ".npz"
        save_cache(output_dir / out_name, clip_feat, dino_feat)
        print(f"saved {out_name}")


if __name__ == "__main__":
    main()
