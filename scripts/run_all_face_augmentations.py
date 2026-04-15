from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.project_env import load_project_env


load_project_env(PROJECT_ROOT)

LOCAL_SCRIPT = PROJECT_ROOT / "scripts" / "generate_student_face_augmentations.py"
ANGLE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_student_face_firered_augmentations.py"
ANGLE_COMBO_SCRIPT = PROJECT_ROOT / "scripts" / "generate_angle_combo_augmentations.py"
LOCAL_BASE_VARIANTS = (
    "shadow,low_light,backlight,harsh_light,blur,low_resolution,noise,compression,"
    "tiny_face,distant_face,partial_crop,main_subject_crop"
)
ACCESSORY_VARIANTS = (
    "cap_block,mask_block,sunglasses_block,"
    "cap_low_light,cap_shadow,cap_compression,"
    "mask_low_resolution,mask_distant_face,mask_partial_crop,"
    "sunglasses_blur,sunglasses_harsh_light,sunglasses_tiny_face"
)


@dataclass(frozen=True)
class Stage:
    name: str
    command: List[str]
    log_path: Path
    required: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local deterministic, hosted angle, and block-accessory face augmentations with logging and resume support.",
    )
    parser.add_argument("--root", default="student-details", help="student-details root folder.")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "student-details" / "augmentation_manifest.csv"),
        help="Unified augmentation manifest path.",
    )
    parser.add_argument("--log-dir", default="outputs/augmentation_runs", help="Folder for stage logs.")
    parser.add_argument("--skip-local", action="store_true", help="Do not run deterministic local augmentations.")
    parser.add_argument("--skip-angle", action="store_true", help="Do not run hosted angle augmentations.")
    parser.add_argument("--skip-angle-combos", action="store_true", help="Do not generate deterministic combos from angle outputs.")
    parser.add_argument("--skip-accessory", action="store_true", help="Do not run block accessory augmentations.")
    parser.add_argument("--local-overwrite", action="store_true", help="Overwrite local deterministic outputs.")
    parser.add_argument("--local-purge-generated", action="store_true", help="Purge stale local/legacy augmentation files.")
    parser.add_argument("--angle-space", default=os.environ.get("HF_ANGLE_SPACE_ID", "linoyts/Qwen-Image-Edit-Angles"))
    parser.add_argument("--angle-max-workers", type=int, default=int(os.environ.get("ANGLE_MAX_WORKERS", "1")))
    parser.add_argument("--angle-retries", type=int, default=int(os.environ.get("AUGMENTATION_RETRIES", "3")))
    parser.add_argument("--angle-timeout", type=int, default=int(os.environ.get("AUGMENTATION_REQUEST_TIMEOUT", "240")))
    parser.add_argument("--angle-steps", type=int, default=int(os.environ.get("ANGLE_STEPS", "4")))
    parser.add_argument("--angle-quota-max-sleep", type=int, default=int(os.environ.get("AUGMENTATION_QUOTA_MAX_SLEEP", "1200")))
    parser.add_argument("--angle-quota-wait-limit", type=int, default=int(os.environ.get("AUGMENTATION_QUOTA_WAIT_LIMIT", "4")))
    parser.add_argument("--angle-max-students", type=int, default=0, help="Optional cap for angle generation.")
    parser.add_argument("--accessory-variants", default=ACCESSORY_VARIANTS, help="Comma-separated local block accessory variants.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue later stages even if one stage fails.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def run_stage(stage: Stage, *, dry_run: bool) -> int:
    print(f"[{stage.name}] starting")
    print(f"[{stage.name}] log: {stage.log_path}")
    append_log(stage.log_path, f"=== {stage.name} started {datetime.now().isoformat(timespec='seconds')} ===")
    append_log(stage.log_path, "COMMAND " + " ".join(stage.command))

    if dry_run:
        append_log(stage.log_path, "DRY RUN: command not executed")
        print(f"[{stage.name}] dry-run complete")
        return 0

    process = subprocess.Popen(
        stage.command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def pump_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            append_log(stage.log_path, line.rstrip())
            print(f"[{stage.name}] {line.rstrip()}")

    thread = threading.Thread(target=pump_output, daemon=True)
    thread.start()
    return_code = process.wait()
    thread.join(timeout=5)
    append_log(stage.log_path, f"=== {stage.name} finished rc={return_code} {datetime.now().isoformat(timespec='seconds')} ===")
    print(f"[{stage.name}] finished rc={return_code}")
    return return_code


def build_local_stage(args: argparse.Namespace, run_id: str, variants: str | None = None, name: str = "local") -> Stage:
    command = [
        sys.executable,
        str(LOCAL_SCRIPT),
        "--root",
        args.root,
    ]
    if variants:
        command.extend(["--variants", variants])
    if args.local_overwrite:
        command.append("--overwrite")
    if args.local_purge_generated:
        command.append("--purge-generated")
    return Stage(name, command, Path(args.log_dir) / run_id / f"{name}.log")


def build_angle_stage(args: argparse.Namespace, run_id: str) -> Stage:
    command = [
        sys.executable,
        str(ANGLE_SCRIPT),
        "--root",
        args.root,
        "--manifest",
        args.manifest,
        "--families",
        "pose",
        "--space",
        args.angle_space,
        "--space-kind",
        "angles",
        "--max-workers",
        str(max(1, int(args.angle_max_workers))),
        "--retries",
        str(max(1, int(args.angle_retries))),
        "--request-timeout",
        str(max(30, int(args.angle_timeout))),
        "--steps",
        str(max(1, int(args.angle_steps))),
        "--smart-mode",
        "--quota-max-sleep",
        str(max(10, int(args.angle_quota_max_sleep))),
        "--quota-wait-limit",
        str(max(0, int(args.angle_quota_wait_limit))),
    ]
    if int(args.angle_max_students) > 0:
        command.extend(["--max-students", str(int(args.angle_max_students))])
    return Stage("angle", command, Path(args.log_dir) / run_id / "angle.log")


def build_accessory_stage(args: argparse.Namespace, run_id: str) -> Stage:
    return build_local_stage(args, run_id, variants=args.accessory_variants, name="accessory_blocks")


def build_angle_combo_stage(args: argparse.Namespace, run_id: str) -> Stage:
    command = [
        sys.executable,
        str(ANGLE_COMBO_SCRIPT),
        "--root",
        args.root,
        "--manifest",
        args.manifest,
    ]
    if args.local_overwrite:
        command.append("--overwrite")
    return Stage("angle_combos", command, Path(args.log_dir) / run_id / "angle_combos.log")


def run_parallel(stages: Sequence[Stage], *, dry_run: bool, continue_on_error: bool) -> dict[str, int]:
    results: dict[str, int] = {}

    def run_and_record(stage: Stage) -> None:
        results[stage.name] = run_stage(stage, dry_run=dry_run)

    threads = [threading.Thread(target=run_and_record, args=(stage,), daemon=True) for stage in stages]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failures = {name: rc for name, rc in results.items() if rc != 0}
    if failures and not continue_on_error:
        raise SystemExit(f"Stage failure(s): {failures}")
    return results


def main() -> None:
    args = parse_args()
    run_id = timestamp()
    stages: List[Stage] = []

    if not args.skip_local:
        stages.append(build_local_stage(args, run_id, variants=LOCAL_BASE_VARIANTS))
    if not args.skip_angle:
        stages.append(build_angle_stage(args, run_id))

    parallel_results = run_parallel(stages, dry_run=bool(args.dry_run), continue_on_error=bool(args.continue_on_error))

    accessory_rc = None
    if not args.skip_accessory:
        accessory_stage = build_accessory_stage(args, run_id)
        accessory_rc = run_stage(accessory_stage, dry_run=bool(args.dry_run))
        if accessory_rc != 0 and not bool(args.continue_on_error):
            raise SystemExit(accessory_rc)

    angle_combo_rc = None
    if not args.skip_angle_combos:
        angle_combo_stage = build_angle_combo_stage(args, run_id)
        angle_combo_rc = run_stage(angle_combo_stage, dry_run=bool(args.dry_run))
        if angle_combo_rc != 0 and not bool(args.continue_on_error):
            raise SystemExit(angle_combo_rc)

    print("=== augmentation run summary ===")
    for name, rc in sorted(parallel_results.items()):
        print(f"{name}: rc={rc}")
    if accessory_rc is not None:
        print(f"accessory: rc={accessory_rc}")
    if angle_combo_rc is not None:
        print(f"angle_combos: rc={angle_combo_rc}")
    print(f"logs: {Path(args.log_dir) / run_id}")


if __name__ == "__main__":
    main()
