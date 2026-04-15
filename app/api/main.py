from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from fastapi import BackgroundTasks, Body, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api import local_db

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_TOPIC_CONFIG = PROJECT_ROOT / "configs" / "topic_profiles.yaml"
RUN_SUMMARY_NAME = "run_summary.json"
PIPELINE_LOG_NAME = "pipeline_log.txt"

app = FastAPI(title="Classroom Monitor API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}

_EMBEDDING_LOCK = threading.Lock()
_EMBEDDING_JOBS: dict[str, dict[str, Any]] = {}


@app.on_event("startup")
def _startup() -> None:
    local_db.init_db()


def _extract_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    prefix = "Bearer "
    if authorization.startswith(prefix):
        return authorization[len(prefix):].strip()
    return authorization.strip()


def _current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = local_db.get_user_for_token(_extract_token(authorization))
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return text[:48] or "session"


def _safe_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    return lowered in {"1", "true", "yes", "on"}


def _safe_string(value: str | None, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _today_string() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _job_response(job_id: str) -> dict[str, Any]:
    with _JOB_LOCK:
        if job_id not in _JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
        job = dict(_JOBS[job_id])
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "current_step": job["current_step"],
        "error_message": job.get("error_message"),
        "result": job.get("result"),
    }


def _set_job_state(job_id: str, **updates: Any) -> None:
    with _JOB_LOCK:
        if job_id not in _JOBS:
            return
        _JOBS[job_id].update(updates)


def _path_or_none(path: Path | None) -> str | None:
    return None if path is None or not path.exists() else str(path.resolve())


def _convert_avi_to_mp4(avi_path: Path) -> Path:
    if not avi_path.exists():
        return avi_path

    mp4_path = avi_path.with_suffix(".mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(avi_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ],
            capture_output=True,
            check=True,
        )
        if mp4_path.exists() and mp4_path.stat().st_size > 0:
            return mp4_path
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(str(avi_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        if mp4_path.exists() and mp4_path.stat().st_size > 0:
            return mp4_path
    except Exception:
        pass

    return avi_path


def _save_optional_upload(upload: UploadFile | None, target_dir: Path, fallback_name: str) -> Path | None:
    if upload is None:
        return None
    suffix = Path(upload.filename or fallback_name).suffix or Path(fallback_name).suffix
    path = target_dir / f"{Path(fallback_name).stem}{suffix}"
    with path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return path


def _run_subprocess(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    log_text = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return result.returncode, log_text


def _build_result_payload(
    run_id: str,
    run_dir: Path,
    class_date: str,
    subject_name: str,
    topic: str,
    timestamp: str,
    log_text: str,
) -> dict[str, Any]:
    face_video = _path_or_none(run_dir / "face_identity.mp4")
    face_csv = _path_or_none(run_dir / "face_identity.csv")
    attn_avi = run_dir / "output.avi"
    attention_video = _path_or_none(_convert_avi_to_mp4(attn_avi) if attn_avi.exists() else None)
    attention_csv = _path_or_none(run_dir / "attendance_report.csv")
    attendance_presence = _path_or_none(run_dir / "attendance_presence.csv")
    attention_metrics = _path_or_none(run_dir / "attention_metrics.csv")
    activity_video = _path_or_none(run_dir / "activity_tracking.mp4")
    activity_csv = _path_or_none(run_dir / "person_activity_summary.csv")
    device_use = _path_or_none(run_dir / "device_use.csv")
    hand_raise = _path_or_none(run_dir / "hand_raise.csv")
    note_taking = _path_or_none(run_dir / "note_taking.csv")
    speech_video = _path_or_none(run_dir / "speech_topics.mp4")
    speech_csv = _path_or_none(run_dir / "speech_topic_segments.csv")
    visual_speaking_video = _path_or_none(run_dir / "visual_speaking.mp4")
    visual_speaking_csv = _path_or_none(run_dir / "visual_speaking.csv")
    lip_transcript = _path_or_none(run_dir / "lip_reading_transcript.csv")
    speech_classification = _path_or_none(run_dir / "speech_topic_classification.csv")
    seat_map_png = _path_or_none(run_dir / "seat_map.png")
    seat_map_json = _path_or_none(run_dir / "seat_map.json")
    seating_timeline = _path_or_none(run_dir / "student_seating_timeline.csv")
    seat_shifts = _path_or_none(run_dir / "seat_shifts.csv")
    attendance_events = _path_or_none(run_dir / "attendance_events.csv")
    final_summary = _path_or_none(run_dir / "final_student_summary.csv")
    run_manifest = _path_or_none(run_dir / "run_manifest.csv")
    log_path = run_dir / PIPELINE_LOG_NAME

    if log_text.strip():
        log_path.write_text(log_text, encoding="utf-8")

    return {
        "run_id": run_id,
        "class_date": class_date,
        "subject_name": subject_name,
        "topic": topic,
        "timestamp": timestamp,
        "faceIdentityVideoPath": face_video,
        "faceIdentityCsvPath": face_csv,
        "attentionVideoPath": attention_video,
        "attendanceCsvPath": attention_csv,
        "attendancePresencePath": attendance_presence,
        "attentionMetricsPath": attention_metrics,
        "activityVideoPath": activity_video,
        "activityCsvPath": activity_csv,
        "deviceUsePath": device_use,
        "handRaisePath": hand_raise,
        "noteTakingPath": note_taking,
        "visualSpeakingVideoPath": visual_speaking_video,
        "visualSpeakingCsvPath": visual_speaking_csv,
        "lipReadingTranscriptPath": lip_transcript,
        "speechTopicClassificationPath": speech_classification,
        "speechVideoPath": speech_video,
        "speechCsvPath": speech_csv,
        "seatMapPngPath": seat_map_png,
        "seatMapJsonPath": seat_map_json,
        "seatingTimelinePath": seating_timeline,
        "seatShiftsPath": seat_shifts,
        "attendanceEventsPath": attendance_events,
        "finalStudentSummaryPath": final_summary,
        "runManifestPath": run_manifest,
        "logText": log_text or "Pipeline complete.",
        "logTextPath": _path_or_none(log_path if log_path.exists() else None),
    }


def _payload_artifacts(payload: dict[str, Any]) -> list[dict[str, str]]:
    labels = {
        "faceIdentityVideoPath": ("Face Identity Video", "video"),
        "faceIdentityCsvPath": ("Face Identity CSV", "csv"),
        "attentionVideoPath": ("Attention Video", "video"),
        "attendanceCsvPath": ("Attendance Report", "csv"),
        "attendancePresencePath": ("Attendance Presence", "csv"),
        "attentionMetricsPath": ("Attention Metrics", "csv"),
        "activityVideoPath": ("Activity Video", "video"),
        "activityCsvPath": ("Activity Summary", "csv"),
        "deviceUsePath": ("Device Use", "csv"),
        "handRaisePath": ("Hand Raise", "csv"),
        "noteTakingPath": ("Note Taking", "csv"),
        "visualSpeakingVideoPath": ("Visual Speaking Video", "video"),
        "visualSpeakingCsvPath": ("Visual Speaking CSV", "csv"),
        "lipReadingTranscriptPath": ("Lip Reading Transcript", "csv"),
        "speechTopicClassificationPath": ("Speech Topic Classification", "csv"),
        "speechVideoPath": ("Speech Topic Video", "video"),
        "speechCsvPath": ("Speech Topic Segments", "csv"),
        "seatMapPngPath": ("Seat Map Image", "image"),
        "seatMapJsonPath": ("Seat Map JSON", "json"),
        "seatingTimelinePath": ("Seating Timeline", "csv"),
        "seatShiftsPath": ("Seat Shifts", "csv"),
        "attendanceEventsPath": ("Attendance Events", "csv"),
        "finalStudentSummaryPath": ("Final Student Summary", "csv"),
        "runManifestPath": ("Run Manifest", "csv"),
        "logTextPath": ("Pipeline Log", "log"),
    }
    artifacts: list[dict[str, str]] = []
    for key, (label, artifact_type) in labels.items():
        path = payload.get(key)
        if path:
            artifacts.append({"label": label, "type": artifact_type, "path": str(path)})
    return artifacts


def _write_run_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / RUN_SUMMARY_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_run_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not OUTPUTS_ROOT.exists():
        return summaries
    for summary_path in OUTPUTS_ROOT.rglob(RUN_SUMMARY_NAME):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload["_summary_path"] = str(summary_path.resolve())
        summaries.append(payload)
    summaries.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return summaries


def _filter_summaries_by_subject(summaries: list[dict[str, Any]], subject: str | None) -> list[dict[str, Any]]:
    if not subject:
        return summaries
    wanted = subject.strip().lower()
    return [item for item in summaries if str(item.get("subject_name", "")).strip().lower() == wanted]


def _mock_rag_response(query: str) -> str:
    lower_query = query.lower()
    if "cnn" in lower_query or "convolutional" in lower_query:
        return (
            "A Convolutional Neural Network learns spatial patterns through shared filters. "
            "It is commonly used for images because convolution preserves local structure."
        )
    if "backpropagation" in lower_query:
        return (
            "Backpropagation computes gradients of the loss with respect to network weights by "
            "applying the chain rule from output back to input."
        )
    if "overfitting" in lower_query:
        return (
            "Overfitting happens when a model learns training-specific noise instead of general "
            "patterns, so validation performance drops even while training performance improves."
        )
    return (
        "That topic is not fully wired to a real RAG backend yet, but the study assistant endpoint is live. "
        "I can return concise DL/ML guidance while we keep the screen functional."
    )


def _process_analysis_job(
    job_id: str,
    input_path: Path,
    run_dir: Path,
    device: str,
    camera_id: str,
    run_face: bool,
    run_attention: bool,
    run_activity: bool,
    run_speech: bool,
    course_profile: str,
    seat_calibration_path: Path | None,
    topic_config_path: Path,
    class_date: str,
    subject_name: str,
    class_topic: str,
    subject_id: str | None = None,
) -> None:
    logs: list[str] = [
        f"Run ID: {job_id}",
        f"Output dir: {run_dir}",
        f"Device: {device}",
        f"Camera ID: {camera_id}",
        f"Subject: {subject_name}",
        f"Topic: {class_topic}",
        f"Class date: {class_date}",
        f"Course profile: {course_profile}",
    ]

    local_db.upsert_run(
        job_id,
        subject_name=subject_name,
        subject_id=subject_id,
        class_date=class_date,
        topic=class_topic,
        status="processing",
        progress=5,
        current_step="Preparing analysis run",
        run_dir=str(run_dir),
    )

    stages: list[tuple[str, list[str]]] = []
    if run_face:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "detectors" / "face_detector" / "run.py"),
            "--input",
            str(input_path),
            "--output",
            str(run_dir / "face_identity.mp4"),
            "--csv",
            str(run_dir / "face_identity.csv"),
            "--identity-db",
            str(PROJECT_ROOT / "detectors" / "face_detector" / "identity_db.json"),
            "--student-details-root",
            str(PROJECT_ROOT / "student-details"),
            "--unknown-output-dir",
            str(run_dir / "unknown_review"),
            "--process-fps",
            "1.0",
            "--det-size",
            "1280",
            "--tile-grid",
            "2",
            "--tile-overlap",
            "0.20",
            "--min-face",
            "12",
        ]
        stages.append(("face identity", cmd))

    if run_attention:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "detectors" / "attention_detector" / "run.py"),
            "--video",
            str(input_path),
            "--config",
            str(PROJECT_ROOT / "configs" / "config.yaml"),
            "--output-dir",
            str(run_dir),
            "--headless",
            "--sample-fps",
            "1.0",
        ]
        if device != "auto":
            cmd += ["--device", device]
        if camera_id.strip():
            cmd += ["--camera", camera_id.strip()]
        if seat_calibration_path is not None:
            cmd += ["--seat-calibration", str(seat_calibration_path)]
        stages.append(("attention/attendance", cmd))

    if run_activity:
        act_video = run_dir / "activity_tracking.mp4"
        act_csv = run_dir / "person_activity_summary.csv"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "detectors" / "activity_detector" / "run.py"),
            "--source",
            str(input_path),
            "--out",
            str(act_video),
            "--activity_out",
            str(act_csv),
            "--track_fps",
            "1.0",
            "--object_fps",
            "1.0",
            "--pose_fps",
            "1.0",
        ]
        if device != "auto":
            cmd += ["--device", device]
        if seat_calibration_path is not None:
            cmd += ["--seat_calibration", str(seat_calibration_path)]
        stages.append(("activity", cmd))

    if run_speech:
        speech_video = run_dir / "speech_topics.mp4"
        speech_csv = run_dir / "speech_topic_segments.csv"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "detectors" / "vlp_detector" / "run.py"),
            "--input",
            str(input_path),
            "--output",
            str(speech_video),
            "--csv",
            str(speech_csv),
            "--course-profile",
            course_profile,
            "--topic-config",
            str(topic_config_path),
        ]
        if device != "auto":
            cmd += ["--device", device]
        if seat_calibration_path is not None:
            cmd += ["--seat-calibration", str(seat_calibration_path)]
        stages.append(("speech", cmd))

    try:
        _set_job_state(job_id, status="processing", progress=5, current_step="Preparing analysis run")
        total_stages = max(1, len(stages))
        if not stages:
            raise RuntimeError("No pipeline stages were selected.")
        for index, (label, cmd) in enumerate(stages):
            start_progress = 5 + int(index * 80 / total_stages)
            end_progress = 5 + int((index + 1) * 80 / total_stages)
            _set_job_state(job_id, progress=start_progress, current_step=f"Running {label}")
            return_code, log_text = _run_subprocess(cmd)
            if log_text:
                logs.append(f"[{label}]\n{log_text}")
            if return_code != 0:
                _set_job_state(
                    job_id,
                    status="failed",
                    progress=end_progress,
                    current_step=f"{label} failed",
                    error_message=f"{label} detector failed",
                )
                local_db.upsert_run(
                    job_id,
                    subject_name=subject_name,
                    subject_id=subject_id,
                    class_date=class_date,
                    topic=class_topic,
                    status="failed",
                    progress=end_progress,
                    current_step=f"{label} failed",
                    error_message=f"{label} detector failed",
                    run_dir=str(run_dir),
                )
                return
            _set_job_state(job_id, progress=end_progress, current_step=f"Completed {label}")
            local_db.upsert_run(
                job_id,
                subject_name=subject_name,
                subject_id=subject_id,
                class_date=class_date,
                topic=class_topic,
                status="processing",
                progress=end_progress,
                current_step=f"Completed {label}",
                run_dir=str(run_dir),
            )

        _set_job_state(job_id, progress=95, current_step="Packaging results")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = _build_result_payload(
            run_id=job_id,
            run_dir=run_dir,
            class_date=class_date,
            subject_name=subject_name,
            topic=class_topic,
            timestamp=timestamp,
            log_text="\n\n".join(logs + ["Pipeline complete"]),
        )
        _write_run_summary(run_dir, payload)
        local_db.upsert_run(
            job_id,
            subject_name=subject_name,
            subject_id=subject_id,
            class_date=class_date,
            topic=class_topic,
            status="completed",
            progress=100,
            current_step="Completed",
            run_dir=str(run_dir),
            result=payload,
        )
        local_db.replace_artifacts(job_id, _payload_artifacts(payload))
        _set_job_state(job_id, status="completed", progress=100, current_step="Completed", result=payload)
    except Exception as exc:
        logs.append(f"[internal]\n{exc}")
        _set_job_state(
            job_id,
            status="failed",
            progress=100,
            current_step="Failed",
            error_message=str(exc),
        )
        local_db.upsert_run(
            job_id,
            subject_name=subject_name,
            subject_id=subject_id,
            class_date=class_date,
            topic=class_topic,
            status="failed",
            progress=100,
            current_step="Failed",
            error_message=str(exc),
            run_dir=str(run_dir),
        )


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "online", "service": "Classroom Monitor API", "version": "2.0.0"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/login")
async def login(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    result = local_db.login(str(payload.get("email", "")), str(payload.get("password", "")))
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return result


@app.post("/api/auth/logout")
async def logout(authorization: str | None = Header(default=None)) -> dict[str, str]:
    token = _extract_token(authorization)
    if token:
        local_db.logout(token)
    return {"status": "ok"}


@app.get("/api/auth/me")
async def me(user: dict[str, Any] = Depends(_current_user)) -> dict[str, Any]:
    return {"user": user}


def _subject_payload(row: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "code": row["code"],
        "description": row.get("description"),
        "icon_index": row.get("icon_index", 0),
        "total_students": row.get("total_students", 0),
        "instructor_id": row.get("instructor_id", user["id"]),
        "instructor_name": user.get("name", "Instructor"),
        "attendance_percentage": 0.0,
        "total_classes": 0,
    }


@app.get("/api/subjects")
async def list_subjects(user: dict[str, Any] = Depends(_current_user)) -> list[dict[str, Any]]:
    return [_subject_payload(row, user) for row in local_db.list_subjects(user)]


@app.post("/api/subjects")
async def create_subject(
    payload: dict[str, Any] = Body(...),
    user: dict[str, Any] = Depends(_current_user),
) -> dict[str, Any]:
    if user["role"] != "instructor":
        raise HTTPException(status_code=403, detail="Only instructors can create subjects")
    if not str(payload.get("name", "")).strip() or not str(payload.get("code", "")).strip():
        raise HTTPException(status_code=400, detail="Subject name and code are required")
    row = local_db.create_subject(user, payload)
    return _subject_payload(row, user)


@app.delete("/api/subjects/{subject_id}")
async def delete_subject(subject_id: str, user: dict[str, Any] = Depends(_current_user)) -> dict[str, str]:
    if not local_db.delete_subject(user, subject_id):
        raise HTTPException(status_code=404, detail="Subject not found")
    return {"status": "deleted"}


@app.post("/api/analysis/jobs")
async def create_analysis_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    device: str = Form("auto"),
    camera_id: str = Form("cam_01"),
    run_face: bool = Form(False),
    run_attention: bool = Form(True),
    run_activity: bool = Form(True),
    run_speech: bool = Form(False),
    full_pipeline: bool = Form(False),
    course_profile: str = Form("default"),
    class_date: str = Form(""),
    class_topic: str = Form("General"),
    subject_name: str = Form(""),
    subject_id: str = Form(""),
    seat_calibration: UploadFile | None = File(None),
    topic_config: UploadFile | None = File(None),
    user: dict[str, Any] = Depends(_current_user),
) -> dict[str, Any]:
    if not str(video.content_type or "").startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    run_id = uuid.uuid4().hex[:8]
    safe_date = _safe_string(class_date, _today_string())
    safe_subject = _safe_string(subject_name, class_topic)
    run_dir = OUTPUTS_ROOT / safe_date / f"{_slugify(safe_subject)}_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    input_path = run_dir / f"input{suffix}"
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    seat_calibration_path = _save_optional_upload(seat_calibration, run_dir, "seat_map_calibration.json")
    topic_config_path = _save_optional_upload(topic_config, run_dir, "topic_profiles.yaml") or DEFAULT_TOPIC_CONFIG

    with _JOB_LOCK:
        _JOBS[run_id] = {
            "status": "queued",
            "progress": 0,
            "current_step": "Queued",
            "error_message": None,
            "result": None,
            "run_dir": str(run_dir),
        }

    selected_full_pipeline = _safe_bool(full_pipeline, False)
    selected_run_face = _safe_bool(run_face, False) or selected_full_pipeline
    selected_run_attention = _safe_bool(run_attention, True) or selected_full_pipeline
    selected_run_activity = _safe_bool(run_activity, True) or selected_full_pipeline
    selected_run_speech = _safe_bool(run_speech, False) or selected_full_pipeline
    local_db.upsert_run(
        run_id,
        subject_name=safe_subject,
        subject_id=_safe_string(subject_id, "") or None,
        class_date=safe_date,
        topic=_safe_string(class_topic, "General"),
        status="queued",
        progress=0,
        current_step="Queued",
        run_dir=str(run_dir),
    )

    background_tasks.add_task(
        _process_analysis_job,
        run_id,
        input_path,
        run_dir,
        _safe_string(device, "auto"),
        _safe_string(camera_id, "cam_01"),
        selected_run_face,
        selected_run_attention,
        selected_run_activity,
        selected_run_speech,
        _safe_string(course_profile, "default"),
        seat_calibration_path,
        topic_config_path,
        safe_date,
        safe_subject,
        _safe_string(class_topic, "General"),
        _safe_string(subject_id, "") or None,
    )

    return {"job_id": run_id, "status": "queued", "progress": 0, "current_step": "Queued", "user": user["id"]}


@app.get("/api/analysis/jobs/{job_id}")
async def get_analysis_job(job_id: str, user: dict[str, Any] = Depends(_current_user)) -> dict[str, Any]:
    _ = user
    return _job_response(job_id)


@app.get("/api/dates")
async def get_available_dates(
    subject: str | None = Query(default=None),
    user: dict[str, Any] = Depends(_current_user),
) -> list[str]:
    _ = user
    summaries = local_db.list_run_summaries(subject=subject)
    if not summaries:
        summaries = _filter_summaries_by_subject(_load_run_summaries(), subject)
    dates = sorted({str(item.get("class_date", "")) for item in summaries if item.get("class_date")}, reverse=True)
    return dates


@app.get("/api/attendance/{date}")
async def get_attendance_runs(
    date: str,
    subject: str | None = Query(default=None),
    user: dict[str, Any] = Depends(_current_user),
) -> list[dict[str, Any]]:
    _ = user
    summaries = local_db.list_run_summaries(subject=subject, date=date)
    if not summaries:
        summaries = _filter_summaries_by_subject(_load_run_summaries(), subject)
    return [item for item in summaries if str(item.get("class_date", "")) == date]


@app.get("/api/files")
async def download_artifact(path: str = Query(...)) -> FileResponse:
    raw_path = Path(path)
    candidate = raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path)
    candidate = candidate.resolve()
    outputs_root = OUTPUTS_ROOT.resolve()
    if outputs_root not in candidate.parents and candidate != outputs_root:
        raise HTTPException(status_code=403, detail="Requested file is outside the outputs directory")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


@app.post("/api/query-rag")
async def query_rag(query: str = Form(...), course_id: str = Form("global")) -> dict[str, str]:
    _ = course_id
    return {"answer": _mock_rag_response(query)}


def _process_embedding_job(
    run_id: str,
    run_local: bool,
    run_angle: bool,
    run_accessory: bool,
    run_angle_combos: bool,
    run_rebuild_db: bool,
) -> None:
    logs = []

    def _log(msg: str):
        print(f"[Embedding Job {run_id}] {msg}")
        logs.append(msg)
        with _EMBEDDING_LOCK:
            _EMBEDDING_JOBS[run_id]["logs"] = logs.copy()

    _log("Starting embedding generation job...")

    try:
        if run_local or run_angle or run_accessory or run_angle_combos:
            _log("Running face augmentations...")
            # We construct the command
            cmd = ["python", "scripts/run_all_face_augmentations.py", "--continue-on-error"]
            if not run_local:
                cmd.append("--skip-local")
            if not run_angle:
                cmd.append("--skip-angle")
            if not run_accessory:
                cmd.append("--skip-accessory")
            if not run_angle_combos:
                cmd.append("--skip-angle-combos")

            _log(f"Executing: {' '.join(cmd)}")
            proc_aug = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc_aug.stdout:
                _log(line.rstrip())
            proc_aug.wait()
            if proc_aug.returncode != 0:
                _log(f"Warning: Augmentations finished with return code {proc_aug.returncode}")

        if run_rebuild_db:
            _log("Rebuilding Identity DB...")
            cmd_db = ["python", "scripts/rebuild_identity_db_from_student_details.py"]
            _log(f"Executing: {' '.join(cmd_db)}")
            proc_db = subprocess.Popen(
                cmd_db,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc_db.stdout:
                _log(line.rstrip())
            proc_db.wait()
            if proc_db.returncode != 0:
                _log(f"Warning: Identity DB rebuild finished with return code {proc_db.returncode}")

        _log("Job complete.")
        with _EMBEDDING_LOCK:
            _EMBEDDING_JOBS[run_id]["status"] = "completed"
            _EMBEDDING_JOBS[run_id]["progress"] = 100

    except Exception as e:
        _log(f"Error during job: {e}")
        with _EMBEDDING_LOCK:
            _EMBEDDING_JOBS[run_id]["status"] = "failed"
            _EMBEDDING_JOBS[run_id]["error_message"] = str(e)


@app.post("/api/embeddings/generate")
async def generate_embeddings(
    background_tasks: BackgroundTasks,
    run_local: bool = Form(False),
    run_angle: bool = Form(False),
    run_accessory: bool = Form(False),
    run_angle_combos: bool = Form(False),
    run_rebuild_db: bool = Form(False),
    user: dict[str, Any] = Depends(_current_user),
) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:8]

    with _EMBEDDING_LOCK:
        _EMBEDDING_JOBS[run_id] = {
            "status": "queued",
            "progress": 0,
            "error_message": None,
            "logs": [],
        }

    background_tasks.add_task(
        _process_embedding_job,
        run_id,
        run_local,
        run_angle,
        run_accessory,
        run_angle_combos,
        run_rebuild_db,
    )

    return {"job_id": run_id, "status": "queued"}


@app.get("/api/embeddings/jobs/{job_id}")
async def get_embedding_job(job_id: str, user: dict[str, Any] = Depends(_current_user)) -> dict[str, Any]:
    with _EMBEDDING_LOCK:
        job = _EMBEDDING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Embedding job not found")
    
    # If the job is logging but status isn't completed or failed, we can guess progress via logs length loosely
    # For now, just return dynamic logs.
    if job["status"] == "queued" and len(job["logs"]) > 0:
        with _EMBEDDING_LOCK:
            job["status"] = "processing"
            job["progress"] = min(99, len(job["logs"]) // 2)

    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "error_message": job["error_message"],
        "logs": job["logs"],
    }
