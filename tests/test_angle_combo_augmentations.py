import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.models.augmentation_manifest import load_manifest
from scripts.generate_angle_combo_augmentations import main as angle_combo_main


class AngleComboAugmentationTests(unittest.TestCase):
    def test_angle_pose_outputs_create_combo_manifest_rows_idempotently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            student_dir = root / "Student-001"
            generated_dir = student_dir / "augmentations_generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            (student_dir / "Photo-Student-001.jpg").parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (220, 280), (176, 152, 136)).save(student_dir / "Photo-Student-001.jpg")
            Image.new("RGB", (220, 280), (176, 152, 136)).save(
                generated_dir / "Photo-Student-001__gen-pose--left_profile.png"
            )
            manifest_path = root / "augmentation_manifest.csv"

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "generate_angle_combo_augmentations.py",
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                ]
                angle_combo_main()
                angle_combo_main()
            finally:
                sys.argv = old_argv

            rows = load_manifest(manifest_path)
            combo_rows = [row for row in rows if row["family"] == "combo"]
            self.assertEqual(len(combo_rows), 9)
            self.assertTrue(any(row["combination_tag"] == "cap_left_profile" for row in combo_rows))
            self.assertTrue(any(row["combination_tag"] == "left_profile_blur" for row in combo_rows))
            self.assertTrue(all(row.get("source_pose_image") for row in combo_rows))
            self.assertTrue((student_dir / "augmentations_combines").exists())


if __name__ == "__main__":
    unittest.main()
