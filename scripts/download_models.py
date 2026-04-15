"""
Download all required model weights for the Classroom Monitor pipeline.

Usage:
    python scripts/download_models.py            # Download all models
    python scripts/download_models.py --dry-run   # Show what would be downloaded

Requires: gdown, requests   (both are in requirements.txt)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# ────────────────────────────────────────────────────────────────
# Model registry
#   key   = local path relative to weights/
#   value = dict with "url" (or "gdown_id"), "size_mb" (approx)
# ────────────────────────────────────────────────────────────────
MODELS: dict[str, dict] = {
    # ── YOLO models (auto-downloaded by ultralytics on first run,
    #    but we pre-fetch them for offline setups) ──
    "yolov8x-worldv2.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt",
        "size_mb": 140,
    },
    "yolo11x-pose.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt",
        "size_mb": 113,
    },
    "yolov8n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "size_mb": 6,
    },
    "yolo26s.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "size_mb": 20,
        "rename": "yolo26s.pt",
    },

    # ── AdaFace (face recognition embedding) ──
    "adaface/adaface_ir101.onnx": {
        "gdown_id": "1f0FVAp1bER8TOGYvjqIVcNRSVnT0fLqs",
        "size_mb": 0.1,
    },
    "adaface/adaface_ir101.onnx.data": {
        "gdown_id": "1LHhp4RpkSLarTPWmB4sYQAMjNxrNCcY6",
        "size_mb": 249,
    },

    # ── TalkNet-ASD (active speaker detection) ──
    "talknet/pretrain_AVA.model": {
        "url": "https://www.robots.ox.ac.uk/~vgg/software/TalkNet/files/pretrain_AVA.model",
        "size_mb": 60,
    },
    "talknet/pretrain_TalkSet.model": {
        "url": "https://www.robots.ox.ac.uk/~vgg/software/TalkNet/files/pretrain_TalkSet.model",
        "size_mb": 60,
    },

    # ── VTP (visual speech processing) ──
    "vtp/public_train_data/feature_extractor.pth": {
        "gdown_id": "1RGaVkPW0GeMa8JYUZkAibEJ-8Iwuy4jn",
        "size_mb": 1030,
    },
    "vtp/public_train_data/ft_lrs2.pth": {
        "gdown_id": "149zURdgap5dXKMz2mBJk0MRfx90MAUDN",
        "size_mb": 883,
    },
    "vtp/public_train_data/ft_lrs3.pth": {
        "gdown_id": "1scJgDBihOI69l6a6tUhfNjGeCMZW4EhJ",
        "size_mb": 647,
    },
}

# ── BERT tokenizers (small, downloaded via HuggingFace) ──
TOKENIZERS = {
    "vtp/tokenizers/bert-base-uncased": "bert-base-uncased",
    "vtp/tokenizers/bert-large-uncased": "bert-large-uncased",
}


def download_url(url: str, dest: Path, label: str) -> bool:
    """Download a file from a URL using requests with a progress bar."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading {label} ...")
    print(f"    URL: {url}")

    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192 * 16):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
                    print(f"\r    [{bar}] {pct}%  ({downloaded // (1024*1024)}MB/{total // (1024*1024)}MB)", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n    ✗ Failed: {e}")
        return False


def download_gdown(file_id: str, dest: Path, label: str) -> bool:
    """Download a file from Google Drive using gdown."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading {label} (Google Drive) ...")

    try:
        import gdown
        gdown.download(id=file_id, output=str(dest), quiet=False, fuzzy=True)
        return dest.exists()
    except Exception as e:
        print(f"    ✗ gdown failed: {e}")
        return False


def download_tokenizer(model_name: str, dest_dir: Path) -> bool:
    """Download a HuggingFace tokenizer to a local directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading tokenizer: {model_name} ...")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(str(dest_dir))
        return True
    except Exception as e:
        print(f"    ✗ Tokenizer download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download all model weights for Classroom Monitor.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading.")
    parser.add_argument("--force", action="store_true", help="Re-download even if file already exists.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Classroom Monitor — Model Weight Downloader")
    print("=" * 60)
    print(f"  Target directory: {WEIGHTS_DIR}")
    print()

    total_mb = sum(m.get("size_mb", 0) for m in MODELS.values())
    print(f"  Total models: {len(MODELS)}  |  Estimated total size: ~{total_mb / 1024:.1f} GB")
    print()

    success = 0
    skipped = 0
    failed = 0

    # Download model files
    for rel_path, info in MODELS.items():
        dest = WEIGHTS_DIR / rel_path
        label = rel_path
        size_mb = info.get("size_mb", "?")

        if dest.exists() and not args.force:
            actual_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✓ {label} already exists ({actual_mb:.0f} MB) — skipping")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [DRY RUN] Would download: {label} (~{size_mb} MB)")
            continue

        if "gdown_id" in info:
            ok = download_gdown(info["gdown_id"], dest, label)
        elif "url" in info:
            ok = download_url(info["url"], dest, label)
        else:
            print(f"  ✗ No download source for {label}")
            failed += 1
            continue

        if ok:
            success += 1
            print(f"  ✓ {label} downloaded successfully")
        else:
            failed += 1

    # Download tokenizers
    for rel_path, model_name in TOKENIZERS.items():
        dest_dir = WEIGHTS_DIR / rel_path
        vocab_file = dest_dir / "vocab.txt"

        if vocab_file.exists() and not args.force:
            print(f"  ✓ {rel_path} already exists — skipping")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [DRY RUN] Would download tokenizer: {model_name}")
            continue

        ok = download_tokenizer(model_name, dest_dir)
        if ok:
            success += 1
        else:
            failed += 1

    print()
    print("=" * 60)
    print(f"  Done!  ✓ {success} downloaded  |  ⊘ {skipped} skipped  |  ✗ {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
