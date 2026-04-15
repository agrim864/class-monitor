from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _exists(path_value: str) -> bool:
    if not path_value:
        return False
    return (PROJECT_ROOT / path_value).exists()


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    visual = config.get("visual_speech", {}) or {}

    checks = {
        "VTP CNN checkpoint": visual.get("vtp_cnn_checkpoint", ""),
        "VTP lip checkpoint": visual.get("vtp_lip_checkpoint", ""),
        "VTP trained VSD checkpoint": visual.get("vtp_vsd_checkpoint", ""),
        "TalkNet repo": visual.get("talknet_repo", ""),
        "TalkNet TalkSet checkpoint": visual.get("talknet_talkset_checkpoint", ""),
        "TalkNet AVA checkpoint": visual.get("talknet_ava_checkpoint", ""),
    }

    print("Visual speech asset check")
    for label, path_value in checks.items():
        status = "OK" if _exists(str(path_value)) else "MISSING"
        print(f"- {label}: {status} {path_value}")

    imports = ["python_speech_features", "scenedetect", "gdown"]
    print("\nTalkNet dependency check")
    for module_name in imports:
        status = "OK" if importlib.util.find_spec(module_name) else "MISSING"
        print(f"- {module_name}: {status}")

    if not _exists(str(visual.get("vtp_vsd_checkpoint", ""))):
        print(
            "\nNote: the native VTP VSD checkpoint is still absent. "
            "The pipeline can use the existing VTP encoder-motion proxy, "
            "or TalkNet can be wired as the trained active-speaker backend."
        )


if __name__ == "__main__":
    main()
