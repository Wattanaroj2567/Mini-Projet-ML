import os
import errno
import uuid
import cv2
import shutil
import time
import re
import io
import json
import pickle
from collections import defaultdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Google Drive imports
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
    
# -----------------------------
# Load config from .env
# -----------------------------
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "outputs")
TMP_DIR = os.getenv("TMP_DIR", "tmp")
CONF_DEFAULT = float(os.getenv("CONF_DEFAULT", "0.35"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Google Drive config
GDRIVE_CREDENTIALS_PATH = os.getenv(
    "GDRIVE_CREDENTIALS_PATH", "credentials/oauth-client.json")
GDRIVE_TOKEN_PATH = os.getenv(
    "GDRIVE_TOKEN_PATH", "credentials/token.pickle")
MAX_SAVED_FRAMES = int(os.getenv("MAX_SAVED_FRAMES", "20"))
FRAME_STRIDE = int(os.getenv("FRAME_STRIDE", "5"))
GDRIVE_PARENT_FOLDER_ID = os.getenv(
    "GDRIVE_PARENT_FOLDER_ID", None)  # ถ้าไม่ระบุจะเก็บใน My Drive root
GDRIVE_ENABLED = os.getenv("GDRIVE_ENABLED", "1") == "1"
GDRIVE_AUTH_PORT = int(os.getenv("GDRIVE_AUTH_PORT", "8080"))

# CORS
ALLOW_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOW_ORIGINS", "").split(",") if o.strip()]

# Frame selection
PREVIEW_ONLY_ONE = os.getenv("PREVIEW_ONLY_ONE", "1") != "0"
PREVIEW_STRATEGY = os.getenv("PREVIEW_STRATEGY", "max_unhelmet_then_conf")

# Frame saving policy
SAVE_ONLY_MIXED = os.getenv("SAVE_ONLY_MIXED", "0") != "0"
SAVE_FRAMES_WITH_HELMET = os.getenv("SAVE_FRAMES_WITH_HELMET", "1") != "0"
SAVE_FRAMES_WITHOUT_HELMET = os.getenv(
    "SAVE_FRAMES_WITHOUT_HELMET", "1") != "0"

# retention
OUTPUT_RETENTION_DAYS = int(os.getenv("OUTPUT_RETENTION_DAYS", "7"))

# canonical paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = (BASE_DIR / OUTPUTS_DIR).resolve()
TMP_DIR_PATH = (BASE_DIR / TMP_DIR).resolve()

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("helmet_api")

# -----------------------------
# Load YOLO model
# -----------------------------
model = None
load_error = None
try:
    from ultralytics import YOLO
    model_path_abs = (BASE_DIR / MODEL_PATH).resolve()
    if model_path_abs.exists():
        model = YOLO(str(model_path_abs))
        log.info(f"✓ Loaded YOLO model from {model_path_abs}")
    else:
        load_error = f"Model file not found: {model_path_abs}"
        log.error(load_error)
except Exception as e:
    load_error = f"YOLO load error: {e}"
    log.exception(load_error)

# -----------------------------
# Google Drive Service (OAuth)
# -----------------------------
SCOPES = ['https://www.googleapis.com/auth/drive.file']

drive_service = None
gdrive_error = None
user_email = None


def get_drive_service():
    """สร้าง Google Drive service ด้วย OAuth 2.0"""
    global drive_service, gdrive_error, user_email

    if not GDRIVE_ENABLED:
        return None

    creds = None
    token_path = BASE_DIR / GDRIVE_TOKEN_PATH
    creds_path = BASE_DIR / GDRIVE_CREDENTIALS_PATH

    # โหลด token ที่เคย authorize แล้ว
    if token_path.exists():
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
            if hasattr(creds, '_service_account_email'):
                user_email = creds._service_account_email

    # ถ้า token หมดอายุหรือไม่มี ให้ login ใหม่
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                log.info("✓ Refreshed Google Drive credentials")
            except Exception as e:
                log.warning(
                    f"Failed to refresh token: {e}, need to re-authorize")
                creds = None

        if not creds:
            if not creds_path.exists():
                gdrive_error = f"OAuth credentials not found: {creds_path}"
                log.error(gdrive_error)
                return None

            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_path), SCOPES)
                port = GDRIVE_AUTH_PORT
                try:
                    creds = flow.run_local_server(port=port)
                except OSError as oauth_err:
                    if port != 0 and oauth_err.errno == errno.EADDRINUSE:
                        log.warning(
                            "Port %s busy; retrying OAuth local server on an ephemeral port",
                            port,
                        )
                        creds = flow.run_local_server(port=0)
                    else:
                        raise
                log.info("✓ Completed OAuth authorization")

                # บันทึก token
                token_path.parent.mkdir(parents=True, exist_ok=True)
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
                log.info(f"✓ Saved token to {token_path}")

            except Exception as e:
                gdrive_error = f"OAuth authorization failed: {e}"
                log.exception(gdrive_error)
                return None

    try:
        service = build('drive', 'v3', credentials=creds)

        # ดึงข้อมูลผู้ใช้
        about = service.about().get(fields='user').execute()
        user_email = about.get('user', {}).get('emailAddress', 'Unknown')

        log.info(f"✓ Google Drive service ready (User: {user_email})")
        return service

    except Exception as e:
        gdrive_error = f"Failed to build Drive service: {e}"
        log.exception(gdrive_error)
        return None


# Initialize Drive service on startup
if GDRIVE_ENABLED:
    try:
        drive_service = get_drive_service()
    except Exception as e:
        gdrive_error = f"Google Drive initialization error: {e}"
        log.exception(gdrive_error)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Helmet Detection API",
    version="2.2.0",
    description="YOLO-based helmet detection with OAuth Google Drive integration"
)

if ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

router = APIRouter(prefix="/analyze/helmet-detection")


@app.get("/")
def root():
    return {"message": "Helmet Detection API running"}

# Class mapping
CLASS_WITH_HELMET = 0
CLASS_NO_HELMET = 1

# -----------------------------
# Schemas
# -----------------------------


class PredictResp(BaseModel):
    request_id: str
    source_type: str
    total_person: int = 0
    count_helmet: int = 0
    count_no_helmet: int = 0
    confidence_threshold: float
    per_frame: Optional[List[Dict]] = None
    artifacts: Dict[str, Any] = {}
    error: Optional[str] = None
    unique_person: Optional[int] = None
    unique_helmet: Optional[int] = None
    unique_no_helmet: Optional[int] = None

# -----------------------------
# Google Drive Helpers
# -----------------------------


def create_drive_folder(folder_name: str, parent_id: str = None) -> Optional[Dict]:
    """สร้างโฟลเดอร์ใน Google Drive ของผู้ใช้"""
    if not drive_service:
        log.error("Google Drive service not available")
        return None

    try:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }

        # ใช้ parent folder ถ้ามีระบุ
        if parent_id or GDRIVE_PARENT_FOLDER_ID:
            file_metadata['parents'] = [parent_id or GDRIVE_PARENT_FOLDER_ID]

        folder = drive_service.files().create(
            body=file_metadata,
            fields='id, name, webViewLink'
        ).execute()

        # แชร์โฟลเดอร์ให้เป็น public (reader) - optional
        try:
            drive_service.permissions().create(
                fileId=folder['id'],
                body={'type': 'anyone', 'role': 'reader'}
            ).execute()
        except Exception as e:
            log.warning(f"Could not make folder public: {e}")

        log.info(f"✓ Created folder: {folder['name']} (ID: {folder['id']})")
        return folder
    except Exception as e:
        log.exception(f"Failed to create folder: {e}")
        return None


def upload_image_to_drive(image_path: Path, folder_id: str, filename: str = None) -> Optional[Dict]:
    """อัปโหลดรูปภาพไป Google Drive ของผู้ใช้"""
    if not drive_service:
        return None

    try:
        if not image_path.exists():
            log.error(f"Image not found: {image_path}")
            return None

        file_name = filename or image_path.name

        with open(image_path, 'rb') as f:
            media = MediaIoBaseUpload(
                io.BytesIO(f.read()),
                mimetype='image/jpeg',
                resumable=True
            )

        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }

        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()

        log.debug(f"✓ Uploaded: {file_name} (ID: {file['id']})")
        return file
    except Exception as e:
        log.error(f"Failed to upload {image_path}: {e}")
        return None


def upload_frames_to_drive(frame_paths: List[Path], source_name: str, request_id: str) -> Dict[str, Any]:
    """อัปโหลดเฟรมทั้งหมดไป Google Drive ของผู้ใช้"""
    result = {
        'success': False,
        'folder_id': None,
        'folder_url': None,
        'uploaded_count': 0,
        'total_count': len(frame_paths)
    }

    if not frame_paths:
        log.warning("No frames to upload")
        return result

    if not drive_service:
        log.warning("Google Drive service not available")
        return result

    # สร้างชื่อโฟลเดอร์
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{source_name}_{request_id}_{timestamp}"

    # สร้างโฟลเดอร์
    folder = create_drive_folder(folder_name)
    if not folder:
        log.error("Failed to create Drive folder")
        return result

    result['folder_id'] = folder['id']
    result['folder_url'] = folder['webViewLink']

    # อัปโหลดรูปทีละรูป
    uploaded = 0
    for i, frame_path in enumerate(frame_paths, 1):
        if upload_image_to_drive(frame_path, folder['id']):
            uploaded += 1
            if i % 10 == 0:
                log.info(f"Upload progress: {i}/{len(frame_paths)}")

    result['uploaded_count'] = uploaded
    result['success'] = uploaded > 0

    log.info(
        f"✓ Upload complete: {uploaded}/{len(frame_paths)} frames to {folder_name}")
    return result

# -----------------------------
# Helpers
# -----------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    base = Path(name or "input").stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip("-._") or "input"


def count_people(result) -> Dict[str, int]:
    """นับคนจาก YOLO result"""
    total = with_helmet = no_helmet = 0
    if result.boxes is None or len(result.boxes) == 0:
        return {"total": 0, "with_helmet": 0, "no_helmet": 0}

    for box in result.boxes:
        total += 1
        cls_id = int(box.cls)
        if cls_id == CLASS_WITH_HELMET:
            with_helmet += 1
        elif cls_id == CLASS_NO_HELMET:
            no_helmet += 1

    return {"total": total, "with_helmet": with_helmet, "no_helmet": no_helmet}


def should_save_frame(counts: Dict[str, int], saved_count: int = 0) -> bool:
    """กติกาเลือกเฟรมที่จะเซฟ"""
    h = counts.get("with_helmet", 0)
    nh = counts.get("no_helmet", 0)

    # ตรวจสอบว่าเกินจำนวนสูงสุดที่อนุญาตหรือยัง
    if saved_count >= MAX_SAVED_FRAMES:
        return False

    if SAVE_ONLY_MIXED:
        return (h > 0 and nh > 0)

    return ((SAVE_FRAMES_WITH_HELMET and h > 0) or
            (SAVE_FRAMES_WITHOUT_HELMET and nh > 0))


def save_image(path: Path, image) -> Path:
    """บันทึกรูปภาพ"""
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
    try:
        os.chmod(str(path), 0o644)
    except Exception:
        pass
    return path


def frame_score(result) -> tuple:
    """ให้คะแนนเฟรมเพื่อคัด 'เฟรมดีที่สุด'"""
    if result.boxes is None or len(result.boxes) == 0:
        return (0, 0.0, 0)

    clss = result.boxes.cls.int().tolist()
    confs = result.boxes.conf.tolist()
    total = len(clss)

    no_helmet_confs = [c for c, k in zip(confs, clss) if k == CLASS_NO_HELMET]
    nh_count = len(no_helmet_confs)
    avg_nh_conf = sum(no_helmet_confs) / nh_count if nh_count else 0.0

    if PREVIEW_STRATEGY == "max_unhelmet_then_conf":
        return (nh_count, avg_nh_conf, total)
    elif PREVIEW_STRATEGY == "max_conf_no_helmet":
        return (avg_nh_conf, nh_count, total)
    else:
        return (nh_count, avg_nh_conf, total)


def cleanup_outputs(retention_days: int = OUTPUT_RETENTION_DAYS) -> None:
    """ลบไฟล์เก่าที่หมดอายุ"""
    if retention_days <= 0:
        return
    base = OUT_DIR
    if not base.is_dir():
        return
    cutoff = time.time() - (retention_days * 86400)
    deleted = 0
    for p in base.iterdir():
        try:
            if p.stat().st_mtime < cutoff:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
                deleted += 1
        except Exception:
            continue
    if deleted > 0:
        log.info(f"Cleaned up {deleted} old files/folders")

# -----------------------------
# Routes
# -----------------------------


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "gdrive_enabled": GDRIVE_ENABLED,
        "gdrive_ready": drive_service is not None,
        "gdrive_error": gdrive_error,
        "model_error": load_error,
        "user_email": user_email,
        "auth_type": "OAuth 2.0",
        "config": {
            "preview_only_one": PREVIEW_ONLY_ONE,
            "max_saved_frames": MAX_SAVED_FRAMES,
            "frame_stride": FRAME_STRIDE,
            "save_only_mixed": SAVE_ONLY_MIXED,
            "save_frames_with_helmet": SAVE_FRAMES_WITH_HELMET,
            "save_frames_without_helmet": SAVE_FRAMES_WITHOUT_HELMET,
            "parent_folder_id": GDRIVE_PARENT_FOLDER_ID,
            "conf_default": CONF_DEFAULT
        }
    }


@router.post("")
async def analyze_helmet_detection(
    file: UploadFile = File(...),
    conf: float = Form(None),
    frame_stride: int = Form(FRAME_STRIDE),  # ใช้ค่าจาก environment
    max_frames: int = Form(300),
    track: int = Form(0),
    imgsz: int = Form(640),
    tracker_name: str = Form("botsort.yaml"),
    min_track_frames: int = Form(2),
    min_box_area: int = Form(0),
    count_rule: str = Form("any-no-helmet"),
):
    if conf is None:
        conf = CONF_DEFAULT

    if model is None:
        raise HTTPException(
            status_code=503, detail=load_error or "Model not loaded")

    request_id = str(uuid.uuid4())[:8]
    safe_name = sanitize_filename(file.filename or "")
    ext = Path(file.filename or "").suffix

    log.info(
        f"[{request_id}] Processing: {file.filename} (track={track}, conf={conf})")

    ensure_dir(TMP_DIR_PATH)
    ensure_dir(OUT_DIR)

    input_path = TMP_DIR_PATH / f"{safe_name}-{request_id}{ext}"
    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        log.debug(
            f"[{request_id}] Saved upload: {input_path} ({len(content)} bytes)")
    except Exception as e:
        raise HTTPException(400, f"Failed to save upload: {e}")

    content_type = (file.content_type or "").lower()
    is_image = content_type.startswith("image/")
    is_video = content_type.startswith("video/")

    try:
        # ----------------- IMAGE -----------------
        if is_image:
            log.info(f"[{request_id}] Processing image...")
            results = model.predict(source=str(
                input_path), conf=conf, imgsz=imgsz, verbose=False)

            totals = {"total": 0, "with_helmet": 0, "no_helmet": 0}
            should_save = False

            for r in results:
                c = count_people(r)
                totals["total"] += c["total"]
                totals["with_helmet"] += c["with_helmet"]
                totals["no_helmet"] += c["no_helmet"]
                if should_save_frame(c):
                    should_save = True

            log.info(f"[{request_id}] Detected: {totals['total']} persons "
                     f"({totals['with_helmet']} helmeted, {totals['no_helmet']} unhelmeted)")

            artifacts: Dict[str, Any] = {}
            saved_paths: List[Path] = []

            if should_save:
                out_file = OUT_DIR / f"{safe_name}-{request_id}.jpg"
                results[0].save(filename=str(out_file))
                try:
                    os.chmod(str(out_file), 0o644)
                except Exception:
                    pass
                saved_paths.append(out_file)
                log.info(f"[{request_id}] Saved annotated image")

            # Upload to Google Drive
            if saved_paths and GDRIVE_ENABLED:
                upload_result = upload_frames_to_drive(
                    saved_paths, safe_name, request_id)
                artifacts["gdrive_folder_id"] = upload_result.get("folder_id")
                artifacts["gdrive_folder_url"] = upload_result.get(
                    "folder_url")
                artifacts["gdrive_uploaded"] = upload_result.get(
                    "uploaded_count", 0)
                artifacts["gdrive_total"] = upload_result.get("total_count", 0)
                artifacts["gdrive_success"] = upload_result.get(
                    "success", False)

            return PredictResp(
                request_id=request_id,
                source_type="image",
                total_person=totals["total"],
                count_helmet=totals["with_helmet"],
                count_no_helmet=totals["no_helmet"],
                confidence_threshold=conf,
                artifacts=artifacts,
            )

        # ----------------- VIDEO -----------------
        if is_video:
            log.info(
                f"[{request_id}] Processing video (stride={frame_stride}, max={max_frames})...")
            cap = cv2.VideoCapture(str(input_path))

            if not cap.isOpened():
                raise HTTPException(400, "Cannot open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            log.info(
                f"[{request_id}] Video info: {total_frames} frames, {fps:.2f} fps")

            frame_idx = 0
            processed = 0
            sum_total = sum_h = sum_nh = 0
            per_frame: List[Dict] = []

            use_tracking = (track == 1)
            unique_person = unique_helmet = unique_no_helmet = None
            track_stats = defaultdict(
                lambda: {"frames": 0, "helmet": 0, "no_helmet": 0})

            best_score = None
            best_idx = None
            best_img = None

            output_dir = OUT_DIR / f"{safe_name}-{request_id}"
            saved_frames: List[Path] = []

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if frame_idx % frame_stride != 0:
                        frame_idx += 1
                        continue

                    if processed >= max_frames:
                        log.info(f"[{request_id}] Reached max_frames limit")
                        break

                    # Predict or Track
                    if use_tracking:
                        results = model.track(
                            source=frame, conf=conf, imgsz=imgsz,
                            persist=True, tracker=tracker_name, verbose=False
                        )
                        r = results[0]
                    else:
                        results = model.predict(
                            source=frame, conf=conf, imgsz=imgsz, verbose=False)
                        r = results[0]

                    # Filter by box area
                    if min_box_area > 0 and r.boxes.xyxy is not None and len(r.boxes) > 0:
                        keep = []
                        for j in range(len(r.boxes)):
                            x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
                            area = max(0, (x2 - x1) * (y2 - y1))
                            if area >= min_box_area:
                                keep.append(j)
                        if keep and len(keep) != len(r.boxes):
                            r.boxes = r.boxes[keep]

                    # Count people
                    c = count_people(r)
                    sum_total += c["total"]
                    sum_h += c["with_helmet"]
                    sum_nh += c["no_helmet"]

                    per_frame.append({
                        "frame": frame_idx,
                        "persons": c["total"],
                        "helmet": c["with_helmet"],
                        "no_helmet": c["no_helmet"]
                    })

                    if c["total"] > 0:
                        log.debug(f"[{request_id}] Frame {frame_idx}: {c['total']} persons, "
                                  f"{c['with_helmet']} helmeted, {c['no_helmet']} unhelmeted")

                    # Save frame if meets criteria
                    if should_save_frame(c, len(saved_frames)):
                        score = frame_score(r)
                        annotated = r.plot()

                        if PREVIEW_ONLY_ONE:
                            if (best_score is None) or (score > best_score):
                                best_score = score
                                best_idx = frame_idx
                                best_img = annotated
                                log.debug(
                                    f"[{request_id}] New best frame: {frame_idx} (score={score})")
                        else:
                            ensure_dir(output_dir)
                            out_path = output_dir / \
                                f"{safe_name}_{frame_idx:06d}.jpg"
                            save_image(out_path, annotated)
                            saved_frames.append(out_path)

                    # Track statistics
                    if use_tracking and r.boxes.id is not None and len(r.boxes) > 0:
                        ids = r.boxes.id.int().tolist()
                        clss = r.boxes.cls.int().tolist()

                        if len(ids) > 0:
                            log.debug(
                                f"[{request_id}] Frame {frame_idx}: Tracked {len(ids)} objects - IDs: {ids}")

                        for tid, cls_id in zip(ids, clss):
                            stat = track_stats[tid]
                            stat["frames"] += 1
                            if cls_id == CLASS_WITH_HELMET:
                                stat["helmet"] += 1
                            elif cls_id == CLASS_NO_HELMET:
                                stat["no_helmet"] += 1

                    frame_idx += 1
                    processed += 1

                    if processed % 50 == 0:
                        log.info(
                            f"[{request_id}] Progress: {processed} frames processed")

            finally:
                cap.release()

            log.info(
                f"[{request_id}] Video processing complete: {processed} frames analyzed")
            log.info(f"[{request_id}] Total detections: {sum_total} persons, "
                     f"{sum_h} helmeted, {sum_nh} unhelmeted")

            # Save best frame if PREVIEW_ONLY_ONE
            if PREVIEW_ONLY_ONE and best_img is not None:
                ensure_dir(output_dir)
                out_file = output_dir / f"{safe_name}_{best_idx:06d}.jpg"
                save_image(out_file, best_img)
                saved_frames.append(out_file)
                log.info(f"[{request_id}] Saved best frame: {best_idx}")

            artifacts: Dict[str, Any] = {}

            # Upload to Google Drive
            if saved_frames and GDRIVE_ENABLED:
                log.info(
                    f"[{request_id}] Uploading {len(saved_frames)} frames to Google Drive...")
                upload_result = upload_frames_to_drive(
                    saved_frames, safe_name, request_id)
                # เพิ่มบรรทัดนี้หลัง saved_frames = []
                saved_frames: List[Path] = []
                log.info(
                    f"[{request_id}] Max saved frames: {MAX_SAVED_FRAMES}, Frame stride: {FRAME_STRIDE}")
                artifacts["gdrive_folder_id"] = upload_result.get("folder_id")
                artifacts["gdrive_folder_url"] = upload_result.get(
                    "folder_url")
                artifacts["gdrive_uploaded"] = upload_result.get(
                    "uploaded_count", 0)
                artifacts["gdrive_total"] = upload_result.get("total_count", 0)
                artifacts["gdrive_success"] = upload_result.get(
                    "success", False)
            elif not saved_frames:
                log.warning(
                    f"[{request_id}] No frames saved (no detections matching criteria)")

            # Calculate unique counters for tracking mode
            if use_tracking:
                filtered = {tid: s for tid, s in track_stats.items()
                            if s["frames"] >= min_track_frames}
                unique_person = len(filtered)

                if count_rule == "majority":
                    unique_no_helmet = sum(
                        1 for s in filtered.values() if s["no_helmet"] > s["helmet"])
                    unique_helmet = sum(
                        1 for s in filtered.values() if s["helmet"] > s["no_helmet"])
                else:  # any-no-helmet
                    unique_no_helmet = sum(
                        1 for s in filtered.values() if s["no_helmet"] > 0)
                    unique_helmet = sum(
                        1
                        for s in filtered.values()
                        if s["no_helmet"] == 0 and s["helmet"] > 0
                    )

            if saved_frames:
                artifacts["saved_frames"] = [str(path) for path in saved_frames]
                artifacts["saved_frames_count"] = len(saved_frames)

            return PredictResp(
                request_id=request_id,
                source_type="video",
                total_person=sum_total,
                count_helmet=sum_h,
                count_no_helmet=sum_nh,
                confidence_threshold=conf,
                per_frame=per_frame or None,
                artifacts=artifacts,
                unique_person=unique_person,
                unique_helmet=unique_helmet,
                unique_no_helmet=unique_no_helmet,
            )

        raise HTTPException(400, f"Unsupported file type: {content_type}")
    except HTTPException:
        raise
    except Exception as exc:
        log.exception(f"[{request_id}] Processing failed: {exc}")
        raise HTTPException(500, "Internal server error during processing") from exc
    finally:
        try:
            if input_path.exists():
                input_path.unlink()
        except Exception as cleanup_err:
            log.warning(f"[{request_id}] Could not remove temp file {input_path}: {cleanup_err}")


app.include_router(router)
