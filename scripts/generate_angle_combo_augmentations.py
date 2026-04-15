from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.augmentation_manifest import load_manifest, save_manifest, upsert_manifest_row
from scripts.generate_student_face_augmentations import (
    COMBINED_OUTPUT_DIR,
    FaceBox,
    augmentation_blur,
    augmentation_cap_block,
    augmentation_compression,
    augmentation_low_light,
    augmentation_low_resolution,
    augmentation_mask_block,
    augmentation_shadow,
    augmentation_sunglasses_block,
    augmentation_tiny_face,
    clamp_box,
    detect_primary_face,
    save_image,
    to_rgb,
)


POSE_TAGS = ("left_profile", "right_profile", "slight_tilt", "up_angle", "down_angle")
ACCESSORY_STEPS: dict[str, Callable[[Image.Image, FaceBox], Image.Image]] = {
    "cap": augmentation_cap_block,
    "mask": augmentation_mask_block,
    "sunglasses": augmentation_sunglasses_block,
}
QUALITY_STEPS: dict[str, Callable[[Image.Image, FaceBox], Image.Image]] = {
    "blur": augmentation_blur,
    "low_light": augmentation_low_light,
    "shadow": augmentation_shadow,
    "low_resolution": augmentation_low_resolution,
    "tiny_face": augmentation_tiny_face,
    "compression": augmentation_compression,
}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class PoseSource:
    student_dir: Path
    pose_path: Path
    pose_tag: str
    source_image_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic accessory/quality combinations from accepted angle augmentation images.",
    )
    parser.add_argument("--root", default="student-details", help="student-details root folder.")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "student-details" / "augmentation_manifest.csv"),
        help="Unified augmentation manifest path.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing angle-combo outputs.")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality for generated images.")
    parser.add_argument("--max-students", type=int, default=0, help="Optional cap on student folders.")
    return parser.parse_args()


def _source_stem_from_pose(path: Path) -> str:
    stem = path.stem
    if "__gen-pose--" in stem:
        return stem.split("__gen-pose--", 1)[0]
    return stem


def _pose_tag_from_path(path: Path) -> str:
    stem = path.stem
    if "__gen-pose--" not in stem:
        return ""
    tag = stem.split("__gen-pose--", 1)[1]
    return tag.strip()


def find_original_source_name(student_dir: Path, pose_path: Path) -> str:
    source_stem = _source_stem_from_pose(pose_path)
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"):
        candidate = student_dir / f"{source_stem}{ext}"
        if candidate.exists():
            return candidate.name
    return f"{source_stem}{pose_path.suffix}"


def iter_pose_sources(root: Path, max_students: int = 0) -> Iterable[PoseSource]:
    student_dirs = [path for path in sorted(root.iterdir(), key=lambda item: item.name.lower()) if path.is_dir()]
    if max_students > 0:
        student_dirs = student_dirs[:max_students]
    for student_dir in student_dirs:
        generated_dir = student_dir / "augmentations_generated"
        if not generated_dir.exists():
            continue
        for pose_path in sorted(generated_dir.iterdir(), key=lambda item: item.name.lower()):
            if not pose_path.is_file() or pose_path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            pose_tag = _pose_tag_from_path(pose_path)
            if pose_tag not in POSE_TAGS:
                continue
            yield PoseSource(
                student_dir=student_dir,
                pose_path=pose_path,
                pose_tag=pose_tag,
                source_image_name=find_original_source_name(student_dir, pose_path),
            )


def output_path_for(source: PoseSource, combo_tag: str) -> Path:
    return source.student_dir / COMBINED_OUTPUT_DIR / f"{source.pose_path.stem}__local-combo--{combo_tag}.jpg"


def generate_combo_image(source: PoseSource, combo_tag: str) -> Image.Image:
    image = to_rgb(Image.open(source.pose_path))
    face = clamp_box(detect_primary_face(image), image)
    if "_" not in combo_tag:
        raise ValueError(f"Invalid combo tag: {combo_tag}")

    if combo_tag.startswith(f"{source.pose_tag}_"):
        quality_tag = combo_tag.removeprefix(f"{source.pose_tag}_")
        step = QUALITY_STEPS[quality_tag]
        return step(image, face)

    accessory, _, pose_tag = combo_tag.partition("_")
    if pose_tag != source.pose_tag:
        raise ValueError(f"Combo tag {combo_tag} does not match pose {source.pose_tag}")
    step = ACCESSORY_STEPS[accessory]
    return step(image, face)


def combo_tags_for_pose(pose_tag: str) -> list[str]:
    tags = [f"{accessory}_{pose_tag}" for accessory in ACCESSORY_STEPS]
    tags.extend(f"{pose_tag}_{quality_tag}" for quality_tag in QUALITY_STEPS)
    return tags


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    manifest_path = Path(args.manifest).resolve()
    if not root.exists():
        raise FileNotFoundError(f"student-details root not found: {root}")

    manifest_rows = load_manifest(manifest_path)
    sources = list(iter_pose_sources(root, max_students=int(args.max_students)))
    created = 0
    existing = 0
    errors = 0

    for source in sources:
        for combo_tag in combo_tags_for_pose(source.pose_tag):
            out_path = output_path_for(source, combo_tag)
            status = "exists"
            reason = ""
            try:
                if not out_path.exists() or bool(args.overwrite):
                    image = generate_combo_image(source, combo_tag)
                    save_image(image, out_path, quality=int(args.quality))
                    status = "created"
                    created += 1
                else:
                    existing += 1
            except Exception as exc:
                status = "error"
                reason = str(exc)[:400]
                errors += 1

            row = {
                "student_folder": source.student_dir.name,
                "source_image": source.source_image_name,
                "source_pose_image": str(source.pose_path.relative_to(root)),
                "generator_type": "local_angle_combo",
                "family": "combo",
                "tag": combo_tag,
                "combination_tag": combo_tag,
                "status": status,
                "rejection_reason": reason,
                "output_image": "" if status == "error" else str(out_path.relative_to(root)),
            }
            manifest_rows = upsert_manifest_row(manifest_rows, row)
            save_manifest(manifest_path, manifest_rows)

    save_manifest(manifest_path, manifest_rows)
    print(f"Pose sources       : {len(sources)}")
    print(f"Combos created    : {created}")
    print(f"Combos existing   : {existing}")
    print(f"Combos errored    : {errors}")
    print(f"Manifest          : {manifest_path}")


if __name__ == "__main__":
    main()
