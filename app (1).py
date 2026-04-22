# ── Standard library ──────────────────────────────────────────────────────────
import streamlit as st
import threading
import queue
import time
import datetime
import math
import json
import csv
import io
import base64
import uuid
import urllib.request
import urllib.parse
import os

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── Hardware ──────────────────────────────────────────────────────────────────
try:
    from djitellopy import Tello
    TELLO_AVAILABLE = True
except ImportError:
    TELLO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import pymongo
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rl_canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

import streamlit.components.v1 as _stc

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(     
    page_title="UAV Navigation & Control Center",
    page_icon="𖥂🎮🕹️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL QUEUES & MJPEG FRAME BUFFER
# ══════════════════════════════════════════════════════════════════════════════
ALERT_Q: queue.Queue = queue.Queue(maxsize=50)
CMD_Q:   queue.Queue = queue.Queue(maxsize=10)

_MJPEG_LOCK     = threading.Lock()
_MJPEG_FRAME: bytes = b""
_MJPEG_META:  dict  = {}
_MJPEG_PORT   = 8889
_MJPEG_STARTED = False

# ── YOLO model (lazy-loaded) ──────────────────────────────────────────────────
_YOLO_MODEL = None
_YOLO_LOCK  = threading.Lock()

# Custom best.pt path (from your training runs)
CUSTOM_MODEL_PATHS = [
    "runs/detect/train5/weights/best.pt",
    "runs/detect/train4/weights/best.pt",
    "runs/detect/train3/weights/best.pt",
    "runs/detect/train2/weights/best.pt",
    "yolov8n.pt",  # fallback
]

def _find_model_path():
    for p in CUSTOM_MODEL_PATHS:
        if os.path.exists(p):
            return p
    return "yolov8n.pt"

def _reset_yolo_model():
    """Properly reset the global YOLO model so it will be reloaded on next inference."""
    global _YOLO_MODEL
    with _YOLO_LOCK:
        _YOLO_MODEL = None

def _get_yolo():
    """Lazy-load custom YOLOv8 model, honouring any path set in session state."""
    global _YOLO_MODEL
    if not YOLO_AVAILABLE:
        return None
    with _YOLO_LOCK:
        if _YOLO_MODEL is None:
            try:
                path = st.session_state.get("yolo_model_path") or _find_model_path()
                if not os.path.exists(path):
                    path = _find_model_path()
                _YOLO_MODEL = _YOLO(path)
                push_alert(f"🤖 YOLO loaded: {path}", "ok")
            except Exception as e:
                push_alert(f"YOLO load failed: {e}", "warn")
                return None
    return _YOLO_MODEL


def run_yolo_detection(bgr):
    """Run YOLOv8 inference. Returns (annotated_bgr, list of detection dicts)."""
    model = _get_yolo()
    if model is None or not CV2_AVAILABLE:
        return bgr, []

    results = model(bgr, imgsz=480, conf=0.35, verbose=False)
    defs    = []
    ann     = bgr.copy()

    _YOLO_COLORS = [
        (255,56,56),(255,157,151),(255,112,31),(255,178,29),(207,210,49),
        (72,249,10),(146,204,23),(61,219,134),(26,147,52),(0,212,187),
        (44,153,168),(0,194,255),(52,69,147),(100,115,255),(0,24,236),
        (132,56,255),(82,0,133),(203,56,255),(255,149,200),(255,55,199),
    ]

    for result in results:
        for box in result.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            conf_val     = float(box.conf[0])
            cls_id       = int(box.cls[0])
            label        = model.names[cls_id]
            color        = _YOLO_COLORS[cls_id % len(_YOLO_COLORS)]
            w_box, h_box = x2-x1, y2-y1

            sev = "low"
            lbl_lower = label.lower()
            if any(k in lbl_lower for k in ("crack","spalling","damage","rebar","tilt","delamination")):
                sev = "critical"
            elif any(k in lbl_lower for k in ("person","car","truck","bus","motorcycle")):
                sev = "high"
            elif any(k in lbl_lower for k in ("corrosion","rust","stain","efflorescence")):
                sev = "medium"

            defs.append({
                "type":    label,
                "conf":    round(conf_val, 2),
                "bbox":    (x1, y1, w_box, h_box),
                "severity":sev,
                "area_px": w_box * h_box,
                "contour": None,
                "source":  "yolo",
            })

            cv2.rectangle(ann, (x1,y1), (x2,y2), color, 2)
            txt          = f"{label} {conf_val:.0%}"
            (tw, th), _  = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            ty           = max(0, y1-1)
            cv2.rectangle(ann, (x1, ty-th-4), (x1+tw+4, ty), color, -1)
            cv2.putText(ann, txt, (x1+2, ty-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0,0,0), 1, cv2.LINE_AA)

    return ann, defs


def get_live_frame_b64() -> str:
    with _MJPEG_LOCK:
        frame = bytes(_MJPEG_FRAME)
    if not frame:
        return ""
    return base64.b64encode(frame).decode()


def live_camera_component(height: int = 440):
    """
    Render the live camera feed.
    Strategy: use a Streamlit image placeholder fed by the MJPEG frame buffer.
    This avoids browser localhost-fetch restrictions entirely (mixed-content /
    cross-origin blocks that silently kill the iframe fetch after 5 retries).
    The autorefresh loop (150 ms) drives re-renders; we just decode the latest
    JPEG from _MJPEG_FRAME and push it through st.image().
    """
    with _MJPEG_LOCK:
        frame = bytes(_MJPEG_FRAME)

    if frame:
        arr = np.frombuffer(frame, dtype=np.uint8)
        if CV2_AVAILABLE:
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                st.image(rgb, use_container_width=True)
                return
        # Fallback: show raw JPEG bytes directly
        st.image(frame, use_container_width=True)
    else:
        # No frame yet — show an animated offline placeholder via HTML
        b64 = ""
        html = f"""
<!DOCTYPE html><html><body style="margin:0;padding:0;background:#030710;overflow:hidden">
<div style="display:flex;align-items:center;justify-content:center;
     height:{height}px;flex-direction:column;gap:10px">
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
       color:#1a4a5a;letter-spacing:2px;text-align:center">
    📷 CAMERA OFFLINE<br>
    <span style="font-size:0.6rem;color:#0d3a50">
      Enable Simulation Mode → Start Camera
    </span>
  </div>
</div>
<style>body{{background:#030710}}</style>
</body></html>
"""
        _stc.html(html, height=height + 4, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE MODULE  (MongoDB + MySQL)
# ══════════════════════════════════════════════════════════════════════════════
def save_detection_db(db_type: str, defect_type: str, severity: str,
                       conf: float, alt_cm: int, image_path: str = ""):
    """Save a detection record to the selected database."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp":  ts,
        "type":       defect_type,
        "severity":   severity,
        "confidence": round(conf, 3),
        "altitude_cm": alt_cm,
        "image_path": image_path,
    }
    try:
        if db_type == "MongoDB" and MONGO_AVAILABLE:
            client = pymongo.MongoClient("mongodb://localhost:27017/",
                                         serverSelectionTimeoutMS=1000)
            client["crack_db"].cracks.insert_one(record)
        elif db_type == "MySQL" and MYSQL_AVAILABLE:
            conn = mysql.connector.connect(
                host="localhost", user="root", password="admin123",
                database="crack_db", connect_timeout=1)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO cracks (timestamp, type, severity, confidence, altitude_cm, image_path) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (ts, defect_type, severity, round(conf, 3), alt_cm, image_path)
            )
            conn.commit()
            conn.close()
        else:
            # In-memory fallback (always works)
            pass
    except Exception as e:
        pass  # DB optional — don't crash the app


def get_db_stats(db_type: str):
    """Returns (connected, total, critical, moderate, recent_rows)."""
    try:
        if db_type == "MongoDB" and MONGO_AVAILABLE:
            client = pymongo.MongoClient("mongodb://localhost:27017/",
                                         serverSelectionTimeoutMS=1000)
            client.admin.command('ping')
            db = client["crack_db"]
            data  = list(db.cracks.find({}, {"_id": 0}).sort("timestamp", -1).limit(10))
            total = db.cracks.count_documents({})
            crit  = db.cracks.count_documents({"severity": "critical"})
            mod   = db.cracks.count_documents({"severity": {"$in": ["high","medium"]}})
            return True, total, crit, mod, data
        elif db_type == "MySQL" and MYSQL_AVAILABLE:
            conn = mysql.connector.connect(
                host="localhost", user="root", password="",
                database="crack_db", connect_timeout=1)
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT * FROM cracks ORDER BY timestamp DESC LIMIT 10")
            data  = cur.fetchall()
            cur.execute("SELECT COUNT(*) as n FROM cracks")
            total = cur.fetchone()['n']
            cur.execute("SELECT COUNT(*) as n FROM cracks WHERE severity='critical'")
            crit  = cur.fetchone()['n']
            cur.execute("SELECT COUNT(*) as n FROM cracks WHERE severity IN ('high','medium')")
            mod   = cur.fetchone()['n']
            conn.close()
            return True, total, crit, mod, data
    except Exception:
        pass
    return False, 0, 0, 0, []


def generate_pdf_report(db_type, project, inspector, total, crit, mod,
                         defect_log, notes="") -> bytes:
    """Generate a PDF inspection report using ReportLab."""
    if not REPORTLAB_AVAILABLE:
        return b""
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=letter)
    W, H = letter

    # Header
    c.setFillColorRGB(0.02, 0.04, 0.06)
    c.rect(0, H-80, W, 80, fill=1, stroke=0)
    c.setFillColorRGB(0, 0.83, 1)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, H-45, "🚁 UAV Navigation & Control Center")
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.29, 0.56, 0.66)
    c.drawString(40, H-65, f"Project: {project}  |  Inspector: {inspector}  |  Database: {db_type}")
    c.drawString(40, H-78, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Stats
    y = H - 120
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "SUMMARY")
    c.line(40, y-5, W-40, y-5)
    y -= 25
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Total Detections: {total}")
    c.drawString(250, y, f"Critical Issues: {crit}")
    c.drawString(420, y, f"Moderate Issues: {mod}")
    y -= 30

    # Notes
    if notes:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y, "Inspector Notes:")
        y -= 16
        c.setFont("Helvetica", 10)
        for line in notes.split("\n")[:10]:
            c.drawString(50, y, line[:100])
            y -= 14
        y -= 10

    # Defect table
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "DEFECT LOG (latest 50)")
    c.line(40, y-5, W-40, y-5)
    y -= 22

    headers = ["Time", "Type", "Severity", "Conf", "Alt(cm)"]
    col_x   = [40, 110, 240, 340, 410]
    c.setFont("Helvetica-Bold", 9)
    c.setFillColorRGB(0, 0.52, 0.75)
    for hdr, cx in zip(headers, col_x):
        c.drawString(cx, y, hdr)
    y -= 14
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 8)

    for d in defect_log[-50:]:
        if y < 60:
            c.showPage()
            y = H - 60
            c.setFont("Helvetica", 8)
        sev = d.get("severity", "")
        if sev == "critical":
            c.setFillColorRGB(1, 0.28, 0.34)
        elif sev == "high":
            c.setFillColorRGB(1, 0.65, 0.01)
        else:
            c.setFillColorRGB(0, 0, 0)
        row = [
            str(d.get("time", "")),
            str(d.get("type", "")),
            str(sev),
            f"{d.get('conf', 0):.0%}",
            str(d.get("alt_cm", "")),
        ]
        for val, cx in zip(row, col_x):
            c.drawString(cx, y, val[:30])
        y -= 12

    c.save()
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
DEFECT_CLASSES = [
    "crack", "spalling", "corrosion", "delamination",
    "exposed_rebar", "water_stain", "efflorescence", "structural_tilt",
]

SEVERITY_COLOR = {
    "critical": "#ff4757",
    "high":     "#ffa502",
    "medium":   "#3742fa",
    "low":      "#2ed573",
}

SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}

DEFECT_SEVERITY_MAP = {
    "crack":            "critical",
    "exposed_rebar":    "critical",
    "spalling":         "high",
    "delamination":     "high",
    "corrosion":        "medium",
    "water_stain":      "low",
    "efflorescence":    "low",
    "structural_tilt":  "critical",
}

CAM_FILTERS = ["Normal", "Grayscale", "Edge Detection", "Night Vision", "Thermal"]

WMO_CODES = {
    0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
    45:"Foggy",48:"Icy fog",51:"Light drizzle",53:"Moderate drizzle",
    55:"Dense drizzle",61:"Slight rain",63:"Moderate rain",65:"Heavy rain",
    71:"Slight snow",73:"Moderate snow",75:"Heavy snow",80:"Slight showers",
    81:"Moderate showers",82:"Violent showers",95:"Thunderstorm",
    96:"Thunderstorm+hail",99:"Heavy thunderstorm+hail",
}

AI_PATH_MODES = ["Grid Scan", "Perimeter Loop", "Spiral Inward", "Zigzag", "Return to Home", "Custom Waypoints"]

# ══════════════════════════════════════════════════════════════════════════════
#  CSS — Futuristic Military HUD theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }
body { background: #050a0e; color: #c8d8e4; }
.stApp { background: #050a0e; }
section[data-testid="stSidebar"] { background: #080e14; }

.hud-header {
  background: linear-gradient(90deg, #050a0e 0%, #0a1628 40%, #0d1f3c 60%, #050a0e 100%);
  border-bottom: 1px solid #0d4f6e; border-top: 1px solid #0d4f6e;
  padding: 10px 24px; margin-bottom: 10px; position: relative; overflow: hidden;
}
.hud-title { font-family: 'Rajdhani', sans-serif; font-size: 1.8rem; font-weight: 700;
  letter-spacing: 3px; color: #00d4ff; text-shadow: 0 0 20px rgba(0,212,255,0.5); }
.hud-subtitle { font-size: 0.72rem; color: #4a8fa8; letter-spacing: 2px;
  text-transform: uppercase; margin-top: 2px; }
.hud-version { font-family: 'Share Tech Mono', monospace; color: #00ff88;
  font-size: 0.75rem; letter-spacing: 1px; }

.status-bar { display: flex; gap: 8px; flex-wrap: wrap; padding: 6px 0; margin-bottom: 4px; }
.s-pill { font-family: 'Share Tech Mono', monospace; font-size: 0.68rem; letter-spacing: 1px;
  padding: 4px 12px; border-radius: 3px; font-weight: 600; border: 1px solid; text-transform: uppercase; }
.s-on   { color: #00ff88; border-color: #00ff88; background: rgba(0,255,136,0.08); }
.s-off  { color: #3a5a6a; border-color: #1a3a4a; background: rgba(0,0,0,0.3); }
.s-warn { color: #ffa502; border-color: #ffa502; background: rgba(255,165,2,0.08); }
.s-crit { color: #ff4757; border-color: #ff4757; background: rgba(255,71,87,0.1);
          animation: pulse-red 1.5s ease-in-out infinite; }
@keyframes pulse-red { 0%,100%{opacity:1} 50%{opacity:0.6} }

.kpi-card { background: linear-gradient(135deg, #080e14 0%, #0a1628 100%);
  border: 1px solid #0d4f6e; border-radius: 6px; padding: 12px 14px; text-align: center;
  position: relative; overflow: hidden; }
.kpi-card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, #00d4ff, transparent); }
.kpi-val { font-family: 'Share Tech Mono', monospace; font-size: 1.6rem; font-weight: 700; }
.kpi-lbl { font-size: 0.6rem; color: #4a8fa8; text-transform: uppercase; letter-spacing: 2px; margin-top: 2px; }

.sec-hdr { font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; font-weight: 600;
  letter-spacing: 2px; color: #00d4ff; text-transform: uppercase;
  border-bottom: 1px solid #0d4f6e; padding-bottom: 5px; margin: 12px 0 8px;
  display: flex; align-items: center; gap: 6px; }

.cam-panel { background: #030710; border: 1px solid #0d4f6e; border-radius: 8px;
  overflow: hidden; box-shadow: 0 0 30px rgba(0,212,255,0.08); }
.cam-offline { display: flex; align-items: center; justify-content: center;
  height: 180px; flex-direction: column; gap: 8px; color: #1a4a5a;
  font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; letter-spacing: 1px;
  background: repeating-linear-gradient(45deg, #030710, #030710 10px, #050c14 10px, #050c14 20px); }

.alert-crit { background: rgba(255,71,87,0.08); border-left: 3px solid #ff4757;
  padding: 6px 12px; margin: 2px 0; border-radius: 0 5px 5px 0;
  font-size: 0.77rem; font-family: 'Share Tech Mono', monospace; }
.alert-warn { background: rgba(255,165,2,0.08); border-left: 3px solid #ffa502;
  padding: 6px 12px; margin: 2px 0; border-radius: 0 5px 5px 0;
  font-size: 0.77rem; font-family: 'Share Tech Mono', monospace; }
.alert-info { background: rgba(0,212,255,0.06); border-left: 3px solid #00d4ff;
  padding: 6px 12px; margin: 2px 0; border-radius: 0 5px 5px 0;
  font-size: 0.77rem; font-family: 'Share Tech Mono', monospace; }
.alert-ok   { background: rgba(0,255,136,0.06); border-left: 3px solid #00ff88;
  padding: 6px 12px; margin: 2px 0; border-radius: 0 5px 5px 0;
  font-size: 0.77rem; font-family: 'Share Tech Mono', monospace; }

.badge { display: inline-block; padding: 2px 10px; border-radius: 3px;
  font-size: 0.68rem; font-weight: 700; margin: 2px; border: 1px solid;
  letter-spacing: 1px; text-transform: uppercase; font-family: 'Share Tech Mono', monospace; }
.badge-critical { color: #ff4757; border-color: #ff4757; background: rgba(255,71,87,0.1); }
.badge-high     { color: #ffa502; border-color: #ffa502; background: rgba(255,165,2,0.1); }
.badge-medium   { color: #3742fa; border-color: #3742fa; background: rgba(55,66,250,0.1); }
.badge-low      { color: #2ed573; border-color: #2ed573; background: rgba(46,213,115,0.1); }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #080e14; border-radius: 6px;
  padding: 4px; border: 1px solid #0d4f6e; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 4px;
  padding: 5px 14px; color: #4a8fa8; font-size: 0.75rem; font-weight: 600;
  font-family: 'Rajdhani', sans-serif; letter-spacing: 1px; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0d2a4e, #0a1f3a) !important;
  color: #00d4ff !important; border: 1px solid #00d4ff !important; }

.ana-bubble-user { background: linear-gradient(135deg, #0a1628, #0d1f3c);
  border: 1px solid #0d4f6e; border-radius: 10px 10px 2px 10px;
  padding: 10px 14px; margin: 6px 0 6px 30px; font-size: 0.83rem; }
.ana-bubble-ai { background: linear-gradient(135deg, #050f1e, #080e14);
  border: 1px solid #00d4ff; border-radius: 10px 10px 10px 2px;
  padding: 10px 14px; margin: 6px 30px 6px 0; font-size: 0.83rem;
  color: #c8d8e4; box-shadow: 0 0 12px rgba(0,212,255,0.08); }
.ana-label { font-size: 0.65rem; color: #4a8fa8; margin-bottom: 2px;
  font-family: 'Share Tech Mono', monospace; letter-spacing: 1px; }

.wx-card { background: linear-gradient(135deg, #050a14, #0a1628);
  border: 1px solid #0d4f6e; border-radius: 8px; padding: 16px 20px; margin-bottom: 10px; }
.wx-temp { font-size: 2.2rem; font-weight: 800; color: #00d4ff;
  font-family: 'Share Tech Mono', monospace; }

.mini-map { background: #030710; border: 1px solid #0d4f6e; border-radius: 6px;
  padding: 8px; text-align: center; font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem; color: #4a8fa8; }

.safety-bar { display: flex; gap: 6px; align-items: center; padding: 6px 10px;
  background: #080e14; border: 1px solid #0d4f6e; border-radius: 5px; margin-bottom: 8px; }
.safety-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.safety-safe    { background: #00ff88; box-shadow: 0 0 6px #00ff88; }
.safety-caution { background: #ffa502; box-shadow: 0 0 6px #ffa502; }
.safety-danger  { background: #ff4757; box-shadow: 0 0 6px #ff4757;
                  animation: pulse-red 0.8s ease-in-out infinite; }
.safety-text { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.rec-dot { display: inline-block; width: 9px; height: 9px; background: #ff4757;
  border-radius: 50%; animation: blink 0.8s ease-in-out infinite; margin-right: 6px; }

.shot-card { background: #080e14; border: 1px solid #0d4f6e; border-radius: 6px;
  padding: 8px; margin-bottom: 10px; }
.shot-card img { width: 100%; border-radius: 4px; }
.shot-ts { font-size: 0.65rem; color: #4a8fa8; margin-top: 4px;
  font-family: 'Share Tech Mono', monospace; }

.loc-card { background: #080e14; border: 1px solid #0d4f6e; border-radius: 6px;
  padding: 12px 16px; margin-bottom: 8px; }

.stProgress > div > div { background: #00d4ff !important; border-radius: 3px; }
.stSlider > div > div > div > div { background: #00d4ff !important; }
.stTextInput > div > div > input { background: #080e14 !important;
  border: 1px solid #0d4f6e !important; color: #c8d8e4 !important;
  border-radius: 5px !important; font-family: 'Share Tech Mono', monospace !important; }

[data-testid="stMetric"] { background: #080e14; border: 1px solid #0d4f6e;
  border-radius: 6px; padding: 10px; }
[data-testid="stMetricLabel"] { color: #4a8fa8 !important; font-size: 0.7rem !important; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #00d4ff !important;
  font-family: 'Share Tech Mono', monospace !important; }

.stButton > button { background: linear-gradient(135deg, #080e14, #0a1628) !important;
  border: 1px solid #0d4f6e !important; color: #c8d8e4 !important;
  border-radius: 5px !important; font-family: 'Rajdhani', sans-serif !important;
  font-weight: 600 !important; letter-spacing: 1px !important;
  font-size: 0.78rem !important; text-transform: uppercase !important; }
.stButton > button:hover { border-color: #00d4ff !important; color: #00d4ff !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #0d2a4e, #0a1f3a) !important;
  border-color: #00d4ff !important; color: #00d4ff !important; }

/* DB status badges */
.db-ok   { color:#00ff88; font-family:'Share Tech Mono',monospace; font-size:0.72rem; }
.db-off  { color:#ff4757; font-family:'Share Tech Mono',monospace; font-size:0.72rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_SS_DEFAULTS: dict = {
    "tello":                None,
    "connected":            False,
    "flying":               False,
    "emergency_stop":       False,
    "mission_running":      False,
    "mission_phase":        "idle",
    "mission_start_time":   None,
    "mission_id":           None,
    "tel": {
        "battery": 0, "height": 0,
        "speed_x": 0, "speed_y": 0, "speed_z": 0,
        "pitch": 0, "roll": 0, "yaw": 0,
        "temp_lo": 0, "temp_hi": 0,
        "tof": 0, "baro": 0.0, "flight_time": 0,
    },
    "cam_active":           False,
    "frame_idx":            0,
    "det_enabled":          True,
    "det_classes":          list(DEFECT_CLASSES),
    "crack_sensitivity":    45,
    "min_defect_area":      120,
    "confidence_threshold": 0.35,
    "yolo_enabled":         True,
    # Survey params
    "survey_rows":          4,
    "survey_cols":          5,
    "survey_altitude":      200,
    "survey_speed":         25,
    "hover_duration":       5,
    "min_battery_rtl":      20,
    "auto_rtl":             True,
    # AI path
    "ai_path_mode":         "Grid Scan",
    "ai_path_waypoints":    [],
    "ai_path_current_wp":   0,
    "ai_safety_min_alt":    50,
    "ai_safety_max_alt":    400,
    "ai_geofence_radius":   500,
    "ai_obstacle_detect":   True,
    "ai_tof_safe_dist":     50,
    # Logs
    "defect_log":           [],
    "flight_log":           [],
    "reinspect_queue":      [],
    "session_stats": {
        "total_frames": 0,
        "defects_found": 0,
        "area_surveyed_m2": 0.0,
        "flight_distance_m": 0.0,
        "missions_completed": 0,
    },
    "pid_state": {"ex": 0, "ey": 0, "ix": 0.0, "iy": 0.0},
    "pid_gains":  {"kp": 0.40, "ki": 0.01, "kd": 0.15},
    "alerts":               [],
    "report_notes":         "",
    "project_name":         "Building Inspection",
    "inspector_name":       "",
    "building_id":          "",
    "ana_history":          [],
    "site_lat":             6.9271,
    "site_lon":             79.8612,
    "site_name":            "Colombo, Sri Lanka",
    "weather_cache":        None,
    "weather_ts":           0.0,
    "zoom_level":           1.0,
    "cam_filter":           "Normal",
    "screenshots":          [],
    "recording":            False,
    "video_frames":         [],
    "fps_ts":               [],
    "battery_ts":           [],
    "auto_reconnect":       False,
    "reconnect_count":      0,
    "stream_health":        "—",
    "move_speed":           30,
    "last_fps":             0.0,
    "safety_status":        "SAFE",
    "tof_distance":         999,
    # Database
    "db_type":              "MongoDB",
    "db_auto_save":         True,
    "db_save_count":        0,
    "yolo_model_path":      _find_model_path() if YOLO_AVAILABLE else "N/A",
    # Simulation mode
    "sim_mode":             False,
    "sim_battery":          87,
    "sim_height":           0,
    "sim_yaw":              0,
    "sim_tof":              350,
}

for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
#  ALERT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def push_alert(msg: str, level: str = "info"):
    entry = {
        "ts":    datetime.datetime.now().strftime("%H:%M:%S"),
        "msg":   msg,
        "level": level,
    }
    st.session_state["alerts"].insert(0, entry)
    st.session_state["alerts"] = st.session_state["alerts"][:60]
    try:
        ALERT_Q.put_nowait(entry)
    except queue.Full:
        pass


def drain_alert_queue():
    while True:
        try:
            entry = ALERT_Q.get_nowait()
            if entry not in st.session_state["alerts"][:5]:
                st.session_state["alerts"].insert(0, entry)
        except queue.Empty:
            break
    st.session_state["alerts"] = st.session_state["alerts"][:60]


# ══════════════════════════════════════════════════════════════════════════════
#  SAFETY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_safety() -> str:
    tel    = st.session_state.get("tel", {})
    tof    = tel.get("tof", 999)
    alt    = tel.get("height", 0)
    bat    = tel.get("battery", 100)
    flying = st.session_state.get("flying", False)
    if not flying:
        return "SAFE"
    min_alt  = st.session_state.get("ai_safety_min_alt", 50)
    max_alt  = st.session_state.get("ai_safety_max_alt", 400)
    safe_tof = st.session_state.get("ai_tof_safe_dist", 50)
    if tof < safe_tof or alt > max_alt or bat < 10:
        return "DANGER"
    if tof < safe_tof * 2 or alt > max_alt * 0.9 or bat < 20 or alt < min_alt * 0.8:
        return "CAUTION"
    return "SAFE"


def is_safe_to_move(direction: str) -> bool:
    if not st.session_state.get("ai_obstacle_detect", True):
        return True
    tel      = st.session_state.get("tel", {})
    tof      = tel.get("tof", 999)
    alt      = tel.get("height", 0)
    safe_dist = st.session_state.get("ai_tof_safe_dist", 50)
    min_alt   = st.session_state.get("ai_safety_min_alt", 50)
    max_alt   = st.session_state.get("ai_safety_max_alt", 400)
    if direction == "fwd" and tof < safe_dist:
        push_alert(f"🛡️ BLOCK: Obstacle {tof}cm ahead", "crit")
        return False
    if direction == "up" and alt >= max_alt:
        push_alert(f"🛡️ BLOCK: Max alt {max_alt}cm", "warn")
        return False
    if direction == "down" and alt <= min_alt:
        push_alert(f"🛡️ BLOCK: Min alt {min_alt}cm", "warn")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  AI PATH PLANNING
# ══════════════════════════════════════════════════════════════════════════════
def generate_ai_path(mode, rows, cols, step_cm, altitude):
    waypoints = []
    if mode == "Grid Scan":
        for r in range(rows):
            col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in col_range:
                waypoints.append({"x": c*step_cm, "y": r*step_cm, "z": altitude,
                                   "yaw": 0, "label": f"R{r+1}C{c+1}", "type": "scan"})
    elif mode == "Perimeter Loop":
        total_w, total_h = cols*step_cm, rows*step_cm
        for i, (x, y) in enumerate([(0,0),(total_w,0),(total_w,total_h),(0,total_h),(0,0)]):
            waypoints.append({"x": x, "y": y, "z": altitude, "yaw": (i*90)%360,
                               "label": f"P{i+1}", "type": "perimeter"})
    elif mode == "Spiral Inward":
        cx, cy = (cols//2)*step_cm, (rows//2)*step_cm
        for ring in range(max(cols,rows)//2, 0, -1):
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                waypoints.append({"x": int(cx+ring*step_cm*math.cos(rad)),
                                   "y": int(cy+ring*step_cm*math.sin(rad)),
                                   "z": altitude, "yaw": angle,
                                   "label": f"S{ring}-{angle}", "type": "spiral"})
    elif mode == "Zigzag":
        for r in range(rows):
            rng = range(0, cols, 2) if r%2==0 else range(cols-1, -1, -2)
            for c in rng:
                waypoints.append({"x": c*step_cm, "y": r*step_cm, "z": altitude,
                                   "yaw": 0 if r%2==0 else 180,
                                   "label": f"Z{r}{c}", "type": "zigzag"})
    elif mode == "Return to Home":
        waypoints = [
            {"x":0,"y":0,"z":100,"yaw":0,"label":"RTH-1","type":"rth"},
            {"x":0,"y":0,"z":50, "yaw":0,"label":"RTH-2","type":"rth"},
        ]
    elif mode == "Custom Waypoints":
        waypoints = st.session_state.get("ai_path_waypoints", [])
    return waypoints


def _ai_autonomous_mission_thread():
    tello   = st.session_state.get("tello")
    sim     = st.session_state.get("sim_mode", False)
    if tello is None and not sim:
        push_alert("No Tello connected.", "crit")
        st.session_state["mission_running"] = False
        return

    mode    = st.session_state.get("ai_path_mode", "Grid Scan")
    rows    = st.session_state.get("survey_rows", 4)
    cols    = st.session_state.get("survey_cols", 5)
    alt     = st.session_state.get("survey_altitude", 200)
    spd     = max(10, min(100, st.session_state.get("survey_speed", 25)))
    h_dur   = st.session_state.get("hover_duration", 5)
    max_alt = st.session_state.get("ai_safety_max_alt", 400)

    waypoints = generate_ai_path(mode, rows, cols, 80, alt)
    st.session_state["ai_path_waypoints"] = waypoints
    st.session_state["ai_path_current_wp"] = 0
    push_alert(f"🤖 AI [{mode}]: {len(waypoints)} waypoints", "ok")

    def check_abort():
        return (st.session_state.get("emergency_stop", False) or
                not st.session_state.get("mission_running", False))

    try:
        st.session_state["mission_phase"] = "takeoff"
        if mode == "Return to Home":
            push_alert("🏠 RTH initiated…", "warn")
            st.session_state["mission_phase"] = "rth"
            try:
                if tello: tello.land()
                else: st.session_state["tel"]["height"] = 0
            except Exception:
                pass
            st.session_state.update({"flying": False, "mission_running": False, "mission_phase": "idle"})
            push_alert("✅ RTH complete.", "ok")
            return

        if tello:
            tello.takeoff()
        else:
            st.session_state["tel"]["height"] = 150
        time.sleep(3)
        st.session_state["mission_phase"] = "ai_scanning"

        prev_x, prev_y = 0, 0
        for i, wp in enumerate(waypoints):
            if check_abort():
                break
            st.session_state["ai_path_current_wp"] = i
            bat = st.session_state["tel"].get("battery", 100)
            if bat <= st.session_state.get("min_battery_rtl", 20):
                push_alert(f"🔋 Battery {bat}% — RTL!", "crit")
                break
            dx, dy = wp["x"]-prev_x, wp["y"]-prev_y
            try:
                if tello:
                    if abs(dx) >= 20:
                        (tello.move_forward if dx>0 else tello.move_back)(min(500, abs(int(dx))))
                        time.sleep(1)
                    if abs(dy) >= 20:
                        (tello.move_right if dy>0 else tello.move_left)(min(500, abs(int(dy))))
                        time.sleep(1)
                else:
                    # Simulate movement — update height toward waypoint altitude
                    st.session_state["tel"]["height"] = int(wp.get("z", 150))
                    time.sleep(0.5)
            except Exception as e:
                push_alert(f"Move error: {e}", "warn")
            time.sleep(1.5)
            prev_x, prev_y = wp["x"], wp["y"]
            push_alert(f"✅ WP {i+1}/{len(waypoints)}: {wp['label']}", "info")

    except Exception as e:
        push_alert(f"AI Mission error: {e}", "crit")
    finally:
        st.session_state["mission_phase"] = "rtl"
        push_alert("🏠 Mission complete — landing…", "warn")
        try:
            if tello: tello.land()
            else: st.session_state["tel"]["height"] = 0
        except Exception:
            pass
        st.session_state.update({"flying": False, "mission_running": False, "mission_phase": "idle"})
        st.session_state["session_stats"]["missions_completed"] += 1
        push_alert("✅ Mission complete.", "ok")


# ══════════════════════════════════════════════════════════════════════════════
#  WEATHER API
# ══════════════════════════════════════════════════════════════════════════════
def fetch_weather(lat, lon):
    now = time.time()
    cached = st.session_state.get("weather_cache")
    if cached and (now - st.session_state.get("weather_ts", 0)) < 300:
        return cached
    try:
        params = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "current": ",".join(["temperature_2m","relative_humidity_2m",
                "apparent_temperature","weathercode","windspeed_10m",
                "winddirection_10m","precipitation","cloudcover","pressure_msl","visibility"]),
            "wind_speed_unit": "kmh", "timezone": "auto",
        })
        url = f"https://api.open-meteo.com/v1/forecast?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "TelloInspectorPro/4.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        c = data.get("current", {})
        result = {
            "temp":        round(c.get("temperature_2m", 0), 1),
            "feels_like":  round(c.get("apparent_temperature", 0), 1),
            "humidity":    int(c.get("relative_humidity_2m", 0)),
            "wind_kmh":    round(c.get("windspeed_10m", 0), 1),
            "wind_dir":    int(c.get("winddirection_10m", 0)),
            "cloud":       int(c.get("cloudcover", 0)),
            "pressure":    round(c.get("pressure_msl", 0), 1),
            "precip":      round(c.get("precipitation", 0), 1),
            "visibility":  round(c.get("visibility", 0) / 1000, 1),
            "code":        int(c.get("weathercode", 0)),
            "timezone":    data.get("timezone", ""),
            "time":        c.get("time", ""),
        }
        result["desc"]   = WMO_CODES.get(result["code"], "Unknown")
        result["fly_ok"] = (result["wind_kmh"] < 25 and
                            result["precip"] == 0 and
                            result["visibility"] > 1.0)
        st.session_state["weather_cache"] = result
        st.session_state["weather_ts"]    = now
        return result
    except Exception:
        return None


def _wx_icon(code):
    if code == 0: return "☀️"
    if code <= 2: return "⛅"
    if code <= 3: return "☁️"
    if code <= 48: return "🌫️"
    if code <= 67: return "🌧️"
    if code <= 77: return "🌨️"
    if code <= 82: return "🌦️"
    if code <= 99: return "⛈️"
    return "🌡️"

def _wind_arrow(deg):
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[round(deg/45) % 8]


# ══════════════════════════════════════════════════════════════════════════════
#  ANA AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
def _build_ana_system() -> str:
    tel     = st.session_state.get("tel", {})
    log     = st.session_state.get("defect_log", [])
    stats   = st.session_state.get("session_stats", {})
    weather = st.session_state.get("weather_cache")
    site    = st.session_state.get("site_name", "Unknown")
    conn    = st.session_state.get("connected", False)
    flying  = st.session_state.get("flying", False)
    phase   = st.session_state.get("mission_phase", "idle")
    safety  = evaluate_safety()
    path_mode = st.session_state.get("ai_path_mode", "Grid Scan")
    wps     = st.session_state.get("ai_path_waypoints", [])
    cur_wp  = st.session_state.get("ai_path_current_wp", 0)
    db_type = st.session_state.get("db_type", "MongoDB")
    model_path = st.session_state.get("yolo_model_path", "N/A")
    crit = sum(1 for d in log if d["severity"] == "critical")
    high = sum(1 for d in log if d["severity"] == "high")
    wx_str = ""
    if weather:
        wx_str = (f"Weather: {weather['temp']}°C, {weather['desc']}, "
                  f"wind {weather['wind_kmh']} km/h "
                  f"({'Safe to fly' if weather['fly_ok'] else 'Caution'}).")
    return f"""You are ANA (Autonomous Navigation Assistant), AI co-pilot for UAV Defect System.

## Current State
- Drone: {'Connected & ' + ('FLYING' if flying else 'GROUNDED') if conn else 'OFFLINE'}
- Safety: {safety} | Phase: {phase.upper()}
- AI path: {path_mode} | WP: {cur_wp}/{len(wps)}
- Battery: {tel.get('battery',0)}% | Alt: {tel.get('height',0)}cm | ToF: {tel.get('tof',0)}cm
- Defects: {stats.get('defects_found',0)} total ({crit} critical, {high} high)
- Database: {db_type} | YOLO model: {model_path}
- Site: {site}
{wx_str}

## Your Role
- Expert in DJI Tello / djitellopy SDK, AI path planning, structural inspection
- 8 defect classes: crack, spalling, corrosion, delamination, exposed_rebar, water_stain, efflorescence, structural_tilt
- Custom YOLOv8 model trained on crack dataset from Roboflow
- MongoDB/MySQL database integration for persistent detection storage
- Always flag DANGER/CAUTION prominently; give concise actionable advice
"""


def ana_chat(user_msg: str) -> str:
    history  = st.session_state.get("ana_history", [])
    messages = history + [{"role": "user", "content": user_msg}]
    payload  = json.dumps({
        "model":      "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "system":     _build_ana_system(),
        "messages":   messages,
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return "\n".join(b["text"] for b in data.get("content",[]) if b.get("type")=="text").strip() or "No response."
    except Exception as e:
        return f"⚠️ ANA error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA FILTER & ZOOM
# ══════════════════════════════════════════════════════════════════════════════
def apply_cam_filter(bgr, filter_name, zoom):
    if not CV2_AVAILABLE:
        return bgr
    if zoom > 1.01:
        h, w = bgr.shape[:2]
        cx, cy = w//2, h//2
        crop_w = max(64, int(w/zoom))
        crop_h = max(48, int(h/zoom))
        x1 = max(0, min(cx - crop_w//2, w - crop_w))
        y1 = max(0, min(cy - crop_h//2, h - crop_h))
        bgr = cv2.resize(bgr[y1:y1+crop_h, x1:x1+crop_w], (w, h), interpolation=cv2.INTER_LINEAR)
    if filter_name == "Grayscale":
        bgr = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif filter_name == "Edge Detection":
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq    = clahe.apply(gray)
        edges = cv2.Canny(eq, 40, 120)
        bgr   = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_name == "Night Vision":
        boosted = cv2.convertScaleAbs(bgr, alpha=2.0, beta=40)
        green   = np.zeros_like(boosted)
        green[:,:,1] = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
        bgr = green
    elif filter_name == "Thermal":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgr  = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return bgr


# ══════════════════════════════════════════════════════════════════════════════
#  CV2 DETECTION MODULES
# ══════════════════════════════════════════════════════════════════════════════
def _clahe_gray(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8)).apply(gray)

def _aspect(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    return max(w,h)/(min(w,h)+1e-5)

def _contour_conf(cnt):
    area = cv2.contourArea(cnt)
    arc  = cv2.arcLength(cnt, False)
    return round(min(0.99, arc/(area+1e-5)*4.5), 2)

def _detect_cracks(bgr, sensitivity, min_area):
    eq     = _clahe_gray(bgr)
    blur   = cv2.GaussianBlur(eq, (5,5), 0)
    lo     = max(10, sensitivity)
    edges  = cv2.Canny(blur, lo, lo*2)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,2)))
    dilated = cv2.dilate(closed, np.ones((2,2),np.uint8), iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defs = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area or _aspect(cnt) < 2.8:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        defs.append({"type":"crack","conf":_contour_conf(cnt),"bbox":(x,y,w,h),
                     "area_px":int(area),"severity":"critical","contour":cnt,"source":"cv2"})
    return defs

def _detect_corrosion(bgr, min_area):
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1   = cv2.inRange(hsv, np.array([0,80,40]),   np.array([18,255,240]))
    m2   = cv2.inRange(hsv, np.array([165,80,40]), np.array([180,255,240]))
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    mask   = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defs = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area: continue
        x,y,w,h = cv2.boundingRect(cnt)
        density = int(cv2.countNonZero(mask[y:y+h,x:x+w]))/(w*h+1e-5)
        conf    = round(min(0.97, density*2.5), 2)
        if conf < 0.50: continue
        defs.append({"type":"corrosion","conf":conf,"bbox":(x,y,w,h),
                     "area_px":int(area),"severity":"medium","contour":cnt,"source":"cv2"})
    return defs

def _detect_spalling(bgr, min_area):
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap     = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap).astype(np.float32)
    mean    = cv2.boxFilter(lap_abs, -1, (25,25))
    mean2   = cv2.boxFilter(lap_abs**2, -1, (25,25))
    var     = np.clip(mean2 - mean**2, 0, None)
    thresh  = np.percentile(var, 92)
    mask    = (var > thresh).astype(np.uint8) * 255
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defs = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area*3 or _aspect(cnt) > 8: continue
        x,y,w,h = cv2.boundingRect(cnt)
        conf = max(0.55, round(min(0.95, (area/(bgr.shape[0]*bgr.shape[1]))*200), 2))
        defs.append({"type":"spalling","conf":conf,"bbox":(x,y,w,h),
                     "area_px":int(area),"severity":"high","contour":cnt,"source":"cv2"})
    return defs

def run_detection(bgr, enabled_classes, sensitivity, min_area, conf_threshold):
    all_defs = []
    if "crack"     in enabled_classes: all_defs += _detect_cracks(bgr, sensitivity, min_area)
    if "spalling"  in enabled_classes: all_defs += _detect_spalling(bgr, min_area)
    if "corrosion" in enabled_classes: all_defs += _detect_corrosion(bgr, min_area)
    all_defs = [d for d in all_defs if d["conf"] >= conf_threshold]

    _DRAW_COLORS = {"crack":(0,30,255),"spalling":(0,140,255),"corrosion":(0,80,200)}
    annotated = bgr.copy()
    for d in all_defs:
        color = _DRAW_COLORS.get(d["type"], (255,255,255))
        x,y,w,h = d["bbox"]
        if d.get("contour") is not None:
            cv2.drawContours(annotated, [d["contour"]], -1, color, 1)
        cv2.rectangle(annotated, (x,y), (x+w,y+h), color, 2)
        cv2.putText(annotated, f"{d['type']} {d['conf']:.0%}", (x, max(0,y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return annotated, all_defs


def draw_hud(bgr, defects, tel, phase, fidx, fps=0, zoom=1.0, filt="Normal",
             recording=False, safety="SAFE"):
    if not CV2_AVAILABLE: return bgr
    h, w = bgr.shape[:2]
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, h-48), (w, h), (0,0,0), -1)
    bgr = cv2.addWeighted(overlay, 0.6, bgr, 0.4, 0)
    bat_color = (0,255,136) if tel.get("battery",0)>50 else (0,165,255) if tel.get("battery",0)>20 else (0,71,255)
    cv2.putText(bgr, f"BAT:{tel.get('battery',0)}%",  (8,  h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bat_color, 1, cv2.LINE_AA)
    cv2.putText(bgr, f"ALT:{tel.get('height',0)}cm",  (90, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,212,255), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"YAW:{tel.get('yaw',0):.0f}°",  (180,h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,212,255), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"ToF:{tel.get('tof',0)}cm",     (270,h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,212,255), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"FPS:{fps:.0f}",                (365,h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
    phase_color = (0,255,136) if phase=="ai_scanning" else (255,165,0) if "hover" in phase else (200,200,200)
    cv2.putText(bgr, phase.upper(), (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, phase_color, 1, cv2.LINE_AA)
    safety_color = (0,255,136) if safety=="SAFE" else (0,165,255) if safety=="CAUTION" else (0,71,255)
    cv2.putText(bgr, f"SAFETY:{safety}", (w-160, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, safety_color, 1, cv2.LINE_AA)
    if zoom > 1.01:
        cv2.putText(bgr, f"ZOOM {zoom:.1f}x", (w-120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,220,0), 1, cv2.LINE_AA)
    if filt != "Normal":
        cv2.putText(bgr, filt.upper(), (w-160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,255,220), 1, cv2.LINE_AA)
    if recording:
        cv2.circle(bgr, (w-15, 15), 6, (0,71,255), -1)
    if defects:
        crit_count = sum(1 for d in defects if d["severity"]=="critical")
        cv2.putText(bgr, f"{len(defects)} DEFECTS",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,71,255) if crit_count>0 else (0,165,255), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"#{fidx}", (100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1, cv2.LINE_AA)
    return bgr


# ══════════════════════════════════════════════════════════════════════════════
#  MJPEG SERVER
# ══════════════════════════════════════════════════════════════════════════════
def _mjpeg_server():
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json as _json

    class MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass
        def do_GET(self):
            if self.path.startswith("/snapshot"):
                with _MJPEG_LOCK:
                    frame = bytes(_MJPEG_FRAME)
                if frame:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(frame)
                else:
                    self.send_response(204); self.end_headers()
            elif self.path.startswith("/video_feed"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                while True:
                    with _MJPEG_LOCK:
                        frame = bytes(_MJPEG_FRAME)
                    if frame:
                        try:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                        except Exception:
                            break
                    time.sleep(0.04)
            else:
                self.send_response(404); self.end_headers()

    HTTPServer(("0.0.0.0", _MJPEG_PORT), MJPEGHandler).serve_forever()


def _ensure_mjpeg_server():
    global _MJPEG_STARTED
    if not _MJPEG_STARTED:
        threading.Thread(target=_mjpeg_server, daemon=True).start()
        _MJPEG_STARTED = True
        time.sleep(0.2)


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA + DETECTION THREAD
# ══════════════════════════════════════════════════════════════════════════════
def _camera_and_detection_thread():
    global _MJPEG_FRAME, _MJPEG_META

    tello = st.session_state.get("tello")
    if tello is None:
        # No real drone — silently fall back to the sim camera (webcam / synthetic)
        push_alert("📷 No drone connected — falling back to webcam/synthetic feed.", "warn")
        _sim_camera_thread()
        return
    try:
        tello.streamon()
        time.sleep(0.7)
        reader = tello.get_frame_read()
    except Exception as e:
        push_alert(f"Stream failed: {e}", "crit")
        st.session_state["cam_active"] = False; return

    push_alert("📷 Live stream active", "ok")
    st.session_state["stream_health"] = "OK"
    _frame_times = []
    _stall_count  = 0

    while st.session_state.get("cam_active", False):
        try:
            raw = reader.frame
            if raw is None or raw.size == 0:
                _stall_count += 1
                if _stall_count > 30:
                    st.session_state["stream_health"] = "WARN"
                time.sleep(0.03); continue
            _stall_count = 0
            st.session_state["stream_health"] = "OK"

            bgr = cv2.resize(raw.copy(), (854, 480))
            det_enabled  = st.session_state.get("det_enabled", True)
            yolo_enabled = st.session_state.get("yolo_enabled", True)
            defects      = []
            ann          = bgr.copy()

            if det_enabled and CV2_AVAILABLE:
                ann, cv2_defs = run_detection(
                    bgr,
                    enabled_classes     = st.session_state.get("det_classes", DEFECT_CLASSES),
                    sensitivity         = st.session_state.get("crack_sensitivity", 45),
                    min_area            = st.session_state.get("min_defect_area", 120),
                    conf_threshold      = st.session_state.get("confidence_threshold", 0.35),
                )
                defects.extend(cv2_defs)

            if yolo_enabled and YOLO_AVAILABLE and CV2_AVAILABLE:
                ann, yolo_defs = run_yolo_detection(ann)
                defects.extend(yolo_defs)

            filt = st.session_state.get("cam_filter", "Normal")
            zoom = st.session_state.get("zoom_level", 1.0)
            ann  = apply_cam_filter(ann, filt, zoom)

            now_t = time.time()
            _frame_times.append(now_t)
            _frame_times = [t for t in _frame_times if now_t - t < 2.0]
            fps = len(_frame_times) / 2.0
            st.session_state["last_fps"] = round(fps, 1)

            tel_snap = dict(st.session_state.get("tel", {}))
            phase    = st.session_state.get("mission_phase", "idle")
            fidx     = st.session_state.get("frame_idx", 0)
            rec      = st.session_state.get("recording", False)
            safety   = evaluate_safety()

            ann = draw_hud(ann, defects, tel_snap, phase, fidx,
                           fps=fps, zoom=zoom, filt=filt, recording=rec, safety=safety)

            bat_now = tel_snap.get("battery", 0)
            if bat_now > 0:
                hist = st.session_state.get("battery_ts", [])
                hist.append((now_t, bat_now))
                st.session_state["battery_ts"] = hist[-120:]

            ok, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                jpeg_bytes = buf.tobytes()
                with _MJPEG_LOCK:
                    _MJPEG_FRAME = jpeg_bytes
                    _MJPEG_META  = {
                        "defects": [{"type":d["type"],"severity":d["severity"],"conf":d["conf"]} for d in defects],
                        "tel": tel_snap, "frame_idx": fidx,
                    }
                if rec:
                    vf = st.session_state.get("video_frames", [])
                    vf.append(jpeg_bytes)
                    st.session_state["video_frames"] = vf[-750:]

            # Log + DB save
            if defects:
                ts  = datetime.datetime.now().strftime("%H:%M:%S")
                tel = st.session_state.get("tel", {})
                db_type   = st.session_state.get("db_type", "MongoDB")
                auto_save = st.session_state.get("db_auto_save", True)
                for d in defects:
                    entry = {
                        "id":       str(uuid.uuid4())[:8],
                        "time":     ts,
                        "type":     d["type"],
                        "severity": d["severity"],
                        "conf":     d["conf"],
                        "area_px":  d["area_px"],
                        "bbox":     str(d["bbox"]),
                        "alt_cm":   tel.get("height", 0),
                        "yaw_deg":  tel.get("yaw", 0),
                        "flight_s": tel.get("flight_time", 0),
                        "frame_idx": fidx,
                        "source":   d.get("source", "cv2"),
                    }
                    st.session_state["defect_log"].append(entry)
                    st.session_state["session_stats"]["defects_found"] += 1

                    # Auto-save to database
                    if auto_save:
                        save_detection_db(db_type, d["type"], d["severity"],
                                          d["conf"], tel.get("height", 0))
                        st.session_state["db_save_count"] = st.session_state.get("db_save_count", 0) + 1

                    if d["severity"] in ("critical", "high"):
                        st.session_state["reinspect_queue"].append({
                            "defect_id": entry["id"], "type": d["type"],
                            "severity": d["severity"], "alt_cm": tel.get("height",0),
                        })

                if len(st.session_state["defect_log"]) > 2000:
                    st.session_state["defect_log"] = st.session_state["defect_log"][-1500:]

            st.session_state["frame_idx"] += 1
            st.session_state["session_stats"]["total_frames"] += 1

        except Exception as e:
            push_alert(f"Detection error: {e}", "warn")
            st.session_state["stream_health"] = "ERROR"
        time.sleep(0.04)

    try:
        tello.streamoff()
    except Exception:
        pass
    with _MJPEG_LOCK:
        _MJPEG_FRAME = b""
    st.session_state["stream_health"] = "—"
    push_alert("📷 Camera stopped.", "info")


def _sim_camera_thread():
    """Camera thread for simulation mode: uses PC webcam or generates synthetic frames."""
    global _MJPEG_FRAME, _MJPEG_META

    cap = None
    if CV2_AVAILABLE:
        # Try indices 0-3 to find any available webcam (USB, built-in, virtual)
        for cam_idx in range(4):
            try:
                _cap = cv2.VideoCapture(cam_idx)
                if _cap.isOpened():
                    # Verify we can actually read a frame
                    _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
                    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ret, test_frame = _cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        cap = _cap
                        push_alert(f"📹 Webcam found on index {cam_idx}", "ok")
                        break
                    else:
                        _cap.release()
                else:
                    _cap.release()
            except Exception as _e:
                pass

    src = "webcam" if cap else "synthetic"
    push_alert(f"📷 [SIM] Camera active ({src})", "ok")
    st.session_state["stream_health"] = "OK"
    _frame_times = []

    while st.session_state.get("cam_active", False):
        try:
            raw = None
            if cap and cap.isOpened():
                ret, raw = cap.read()
                if not ret:
                    raw = None

            if raw is None:
                if not CV2_AVAILABLE:
                    # Pure-numpy synthetic frame (no CV2) encoded via PIL if available
                    try:
                        from PIL import Image as _PILImage
                        _arr = np.zeros((480, 854, 3), dtype=np.uint8)
                        _arr[:] = (20, 14, 8)
                        _img = _PILImage.fromarray(_arr)
                        _buf = io.BytesIO()
                        _img.save(_buf, format="JPEG", quality=70)
                        jpeg_bytes = _buf.getvalue()
                        with _MJPEG_LOCK:
                            _MJPEG_FRAME = jpeg_bytes
                            _MJPEG_META  = {"defects": [], "tel": {}, "frame_idx": 0}
                    except Exception:
                        pass
                    st.session_state["frame_idx"] = st.session_state.get("frame_idx", 0) + 1
                    st.session_state["session_stats"]["total_frames"] += 1
                    time.sleep(0.04)
                    continue
                # Generate animated synthetic frame (CV2 available)
                raw = np.zeros((480, 854, 3), dtype=np.uint8)
                raw[:] = (8, 14, 20)
                t_now = time.time()
                for i in range(0, 854, 60):
                    cv2.line(raw, (i, 0), (i, 480), (18, 38, 58), 1)
                for i in range(0, 480, 60):
                    cv2.line(raw, (0, i), (854, i), (18, 38, 58), 1)
                cx = int(427 + 120 * math.sin(t_now * 0.4))
                cy = int(240 + 70 * math.cos(t_now * 0.3))
                cv2.circle(raw, (cx, cy), 28, (0, 212, 255), -1)
                cv2.circle(raw, (cx, cy), 38, (0, 212, 255), 2)
                cv2.putText(raw, "SIMULATION MODE", (260, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 212, 255), 2, cv2.LINE_AA)
                cv2.putText(raw, "No drone connected — simulated feed",
                            (230, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (74, 143, 168), 1, cv2.LINE_AA)
                cv2.putText(raw, "Connect webcam or real drone for live video",
                            (210, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 100, 120), 1, cv2.LINE_AA)

            bgr = cv2.resize(raw, (854, 480))
            det_enabled  = st.session_state.get("det_enabled", True)
            yolo_enabled = st.session_state.get("yolo_enabled", True)
            defects      = []
            ann          = bgr.copy()

            if det_enabled and CV2_AVAILABLE:
                ann, cv2_defs = run_detection(
                    bgr,
                    enabled_classes = st.session_state.get("det_classes", DEFECT_CLASSES),
                    sensitivity     = st.session_state.get("crack_sensitivity", 45),
                    min_area        = st.session_state.get("min_defect_area", 120),
                    conf_threshold  = st.session_state.get("confidence_threshold", 0.35),
                )
                defects.extend(cv2_defs)

            if yolo_enabled and YOLO_AVAILABLE and CV2_AVAILABLE:
                ann, yolo_defs = run_yolo_detection(ann)
                defects.extend(yolo_defs)

            filt = st.session_state.get("cam_filter", "Normal")
            zoom = st.session_state.get("zoom_level", 1.0)
            ann  = apply_cam_filter(ann, filt, zoom)

            now_t = time.time()
            _frame_times.append(now_t)
            _frame_times = [ft for ft in _frame_times if now_t - ft < 2.0]
            fps = len(_frame_times) / 2.0
            st.session_state["last_fps"] = round(fps, 1)

            tel_snap = dict(st.session_state.get("tel", {}))
            phase    = st.session_state.get("mission_phase", "idle")
            fidx     = st.session_state.get("frame_idx", 0)
            rec      = st.session_state.get("recording", False)
            safety   = evaluate_safety()

            ann = draw_hud(ann, defects, tel_snap, phase, fidx,
                           fps=fps, zoom=zoom, filt=filt, recording=rec, safety=safety)

            bat_now = tel_snap.get("battery", 0)
            if bat_now > 0:
                hist = st.session_state.get("battery_ts", [])
                hist.append((now_t, bat_now))
                st.session_state["battery_ts"] = hist[-120:]

            ok, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                jpeg_bytes = buf.tobytes()
                with _MJPEG_LOCK:
                    _MJPEG_FRAME = jpeg_bytes
                    _MJPEG_META  = {
                        "defects": [{"type": d["type"], "severity": d["severity"], "conf": d["conf"]} for d in defects],
                        "tel": tel_snap, "frame_idx": fidx,
                    }
                if rec:
                    vf = st.session_state.get("video_frames", [])
                    vf.append(jpeg_bytes)
                    st.session_state["video_frames"] = vf[-750:]

            if defects:
                ts        = datetime.datetime.now().strftime("%H:%M:%S")
                tel_d     = st.session_state.get("tel", {})
                db_type   = st.session_state.get("db_type", "MongoDB")
                auto_save = st.session_state.get("db_auto_save", True)
                for d in defects:
                    entry = {
                        "id": str(uuid.uuid4())[:8], "time": ts,
                        "type": d["type"], "severity": d["severity"],
                        "conf": d["conf"], "area_px": d["area_px"],
                        "bbox": str(d["bbox"]), "alt_cm": tel_d.get("height", 0),
                        "yaw_deg": tel_d.get("yaw", 0), "flight_s": tel_d.get("flight_time", 0),
                        "frame_idx": fidx, "source": d.get("source", "cv2"),
                    }
                    st.session_state["defect_log"].append(entry)
                    st.session_state["session_stats"]["defects_found"] += 1
                    if auto_save:
                        save_detection_db(db_type, d["type"], d["severity"],
                                          d["conf"], tel_d.get("height", 0))
                        st.session_state["db_save_count"] = st.session_state.get("db_save_count", 0) + 1
                    if d["severity"] in ("critical", "high"):
                        st.session_state["reinspect_queue"].append({
                            "defect_id": entry["id"], "type": d["type"],
                            "severity": d["severity"], "alt_cm": tel_d.get("height", 0),
                        })
                if len(st.session_state["defect_log"]) > 2000:
                    st.session_state["defect_log"] = st.session_state["defect_log"][-1500:]

            st.session_state["frame_idx"] += 1
            st.session_state["session_stats"]["total_frames"] += 1

        except Exception as e:
            push_alert(f"Sim camera error: {e}", "warn")
            st.session_state["stream_health"] = "ERROR"
        time.sleep(0.04)

    if cap:
        cap.release()
    with _MJPEG_LOCK:
        _MJPEG_FRAME = b""
    st.session_state["stream_health"] = "—"
    push_alert("📷 Simulation camera stopped.", "info")


def start_camera():
    if st.session_state.get("cam_active"):
        return
    _ensure_mjpeg_server()
    st.session_state.update({"cam_active": True, "video_frames": [], "battery_ts": []})
    if st.session_state.get("sim_mode"):
        if not CV2_AVAILABLE:
            push_alert("⚠️ opencv-python not installed — install it for webcam/detection support.", "warn")
            # Still start a minimal synthetic-only thread
        threading.Thread(target=_sim_camera_thread, daemon=True).start()
    else:
        if not CV2_AVAILABLE:
            push_alert("opencv-python not installed — camera unavailable.", "crit")
            st.session_state["cam_active"] = False
            return
        threading.Thread(target=_camera_and_detection_thread, daemon=True).start()

def stop_camera():
    st.session_state.update({"cam_active": False, "recording": False})


# ══════════════════════════════════════════════════════════════════════════════
#  SCREENSHOT / RECORDING
# ══════════════════════════════════════════════════════════════════════════════
def capture_screenshot():
    with _MJPEG_LOCK:
        frame = bytes(_MJPEG_FRAME)
        meta  = dict(_MJPEG_META)
    if not frame:
        push_alert("No frame to capture.", "warn"); return
    ts  = datetime.datetime.now().strftime("%H%M%S")
    # Also save to disk in captures/ folder
    os.makedirs("captures", exist_ok=True)
    fname = f"captures/crack_{ts}.jpg"
    try:
        with open(fname, "wb") as f:
            f.write(frame)
    except Exception:
        pass
    b64 = base64.b64encode(frame).decode()
    entry = {"id": str(uuid.uuid4())[:8], "ts": ts, "b64": b64,
             "defects": meta.get("defects",[]), "tel": meta.get("tel",{})}
    shots = st.session_state.get("screenshots", [])
    shots.insert(0, entry)
    st.session_state["screenshots"] = shots[:50]
    push_alert(f"📸 Screenshot saved: {fname}", "ok")


def battery_eta() -> str:
    hist = st.session_state.get("battery_ts", [])
    if len(hist) < 6: return "—"
    t0, b0 = hist[0]; t1, b1 = hist[-1]
    elapsed = t1 - t0
    if elapsed < 5 or b0 <= b1: return "—"
    drain_per_sec = (b0 - b1) / elapsed
    if drain_per_sec <= 0: return "—"
    seconds_left = b1 / drain_per_sec
    m, s = divmod(int(seconds_left), 60)
    return f"{m}:{s:02d}"


def start_recording():
    st.session_state.update({"recording": True, "video_frames": []})
    push_alert("🔴 Recording started", "ok")

def stop_recording():
    st.session_state["recording"] = False
    push_alert(f"⏹️ Recording stopped — {len(st.session_state.get('video_frames',[]))} frames", "info")

def build_mjpeg_download() -> bytes:
    frames = st.session_state.get("video_frames", [])
    buf = io.BytesIO()
    for jpeg in frames:
        buf.write(b"--jpgboundary\r\nContent-Type: image/jpeg\r\n\r\n")
        buf.write(jpeg); buf.write(b"\r\n")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  TELEMETRY THREAD
# ══════════════════════════════════════════════════════════════════════════════
def _telemetry_thread():
    tello = st.session_state.get("tello")
    if tello is None: return
    while st.session_state.get("connected"):
        if st.session_state.get("emergency_stop"): break
        try:
            tel = st.session_state["tel"]
            try: tel["battery"]     = tello.get_battery()
            except: pass
            try: tel["tof"]         = tello.get_distance_tof()
            except: pass
            if st.session_state.get("flying"):
                try: tel["height"]      = tello.get_height()
                except: pass
                try: tel["speed_x"]     = tello.get_speed_x()
                except: pass
                try: tel["speed_y"]     = tello.get_speed_y()
                except: pass
                try: tel["yaw"]         = tello.get_yaw()
                except: pass
                try: tel["pitch"]       = tello.get_pitch()
                except: pass
                try: tel["roll"]        = tello.get_roll()
                except: pass
                try: tel["flight_time"] = tello.get_flight_time()
                except: pass
                st.session_state["safety_status"] = evaluate_safety()
                st.session_state["tof_distance"]   = tel["tof"]
                entry = {"time": datetime.datetime.now().isoformat(),
                         "battery": tel["battery"], "height": tel["height"],
                         "yaw": tel["yaw"], "speed_x": tel["speed_x"], "speed_y": tel["speed_y"]}
                st.session_state["flight_log"].append(entry)
                if len(st.session_state["flight_log"]) > 3000:
                    st.session_state["flight_log"] = st.session_state["flight_log"][-2000:]
                spd = math.hypot(tel["speed_x"], tel["speed_y"])
                st.session_state["session_stats"]["flight_distance_m"] += spd / 100.0
                bat = tel["battery"]
                if 0 < bat <= st.session_state.get("min_battery_rtl", 20):
                    push_alert(f"🔋 Battery {bat}% — RTL!", "crit")
                    if st.session_state.get("auto_rtl", True):
                        _do_land(); break
        except Exception as e:
            push_alert(f"Telemetry: {e}", "warn")
        time.sleep(1)


# ══════════════════════════════════════════════════════════════════════════════
#  FLIGHT COMMAND HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _sim_telemetry_thread():
    """Simulate realistic drone telemetry without real hardware."""
    t_start = time.time()
    while st.session_state.get("connected") and st.session_state.get("sim_mode"):
        if st.session_state.get("emergency_stop"):
            break
        try:
            tel     = st.session_state["tel"]
            elapsed = time.time() - t_start
            # Battery drains ~1% per 60s
            tel["battery"] = max(0, 87 - int(elapsed / 60))
            if st.session_state.get("flying"):
                # Smoothly converge height toward survey altitude
                target_h = st.session_state.get("survey_altitude", 150)
                cur_h    = tel.get("height", 0)
                tel["height"]      = int(cur_h * 0.88 + target_h * 0.12)
                tel["tof"]         = max(30, 350 - tel["height"] + int(math.sin(elapsed * 0.5) * 10))
                tel["speed_x"]     = int(math.sin(elapsed * 0.8) * 20)
                tel["speed_y"]     = int(math.cos(elapsed * 0.6) * 15)
                tel["pitch"]       = int(math.sin(elapsed * 0.3) * 4)
                tel["roll"]        = int(math.cos(elapsed * 0.4) * 3)
                tel["flight_time"] = int(elapsed)
                tel["temp_lo"]     = 30; tel["temp_hi"] = 35
                st.session_state["safety_status"] = evaluate_safety()
                st.session_state["tof_distance"]  = tel["tof"]
                entry = {"time": datetime.datetime.now().isoformat(),
                         "battery": tel["battery"], "height": tel["height"],
                         "yaw": tel.get("yaw", 0), "speed_x": tel["speed_x"], "speed_y": tel["speed_y"]}
                st.session_state["flight_log"].append(entry)
                if len(st.session_state["flight_log"]) > 3000:
                    st.session_state["flight_log"] = st.session_state["flight_log"][-2000:]
                spd = math.hypot(tel["speed_x"], tel["speed_y"])
                st.session_state["session_stats"]["flight_distance_m"] += spd / 100.0
                bat = tel["battery"]
                if 0 < bat <= st.session_state.get("min_battery_rtl", 20):
                    push_alert(f"🔋 [SIM] Battery {bat}% — RTL!", "crit")
                    if st.session_state.get("auto_rtl", True):
                        _do_land(); break
            else:
                tel["height"] = max(0, tel.get("height", 0) - 5)
                tel["tof"]    = 350
                tel["speed_x"] = 0; tel["speed_y"] = 0
        except Exception as e:
            push_alert(f"Sim telemetry: {e}", "warn")
        time.sleep(1)


def _do_connect(sim: bool = False):
    if sim:
        st.session_state.update({
            "tello": None, "connected": True, "sim_mode": True,
            "emergency_stop": False,
        })
        st.session_state["tel"]["battery"] = 87
        st.session_state["tel"]["tof"]     = 350
        st.session_state["tel"]["height"]  = 0
        st.session_state["reconnect_count"] += 1
        push_alert("🖥️ SIMULATION MODE active — No drone required", "ok")
        push_alert("📡 Simulated telemetry running", "info")
        threading.Thread(target=_sim_telemetry_thread, daemon=True).start()
        return True

    if not TELLO_AVAILABLE:
        push_alert("djitellopy not installed.", "crit"); return False
    if not CV2_AVAILABLE:
        push_alert("opencv-python not installed.", "crit"); return False
    try:
        t = Tello()
        t.connect()
        bat = t.get_battery()
        st.session_state.update({"tello": t, "connected": True, "sim_mode": False})
        st.session_state["tel"]["battery"] = bat
        st.session_state["reconnect_count"] += 1
        push_alert(f"✅ Tello connected. Battery: {bat}%", "ok")
        threading.Thread(target=_telemetry_thread, daemon=True).start()
        return True
    except Exception as e:
        push_alert(f"Connection failed: {e}", "crit"); return False

def _do_disconnect():
    stop_camera()
    t = st.session_state.get("tello")
    if t:
        try: t.end()
        except: pass
    st.session_state.update({"tello": None, "connected": False, "flying": False,
                               "mission_running": False, "mission_phase": "idle",
                               "sim_mode": False})
    push_alert("Disconnected.", "info")

def _do_takeoff():
    if st.session_state.get("sim_mode"):
        st.session_state.update({"flying": True, "emergency_stop": False,
                                   "mission_start_time": datetime.datetime.now()})
        st.session_state["tel"]["height"] = 150
        st.session_state["tel"]["tof"]    = 200
        push_alert("🛫 [SIM] Takeoff! Climbing to 150cm…", "ok")
        return
    t = st.session_state.get("tello")
    if t is None: push_alert("Not connected.", "crit"); return
    try:
        t.takeoff()
        st.session_state.update({"flying": True, "emergency_stop": False,
                                   "mission_start_time": datetime.datetime.now()})
        push_alert("🛫 Takeoff!", "ok")
    except Exception as e:
        push_alert(f"Takeoff failed: {e}", "crit")

def _do_land():
    if st.session_state.get("sim_mode"):
        st.session_state["tel"]["height"] = 0
        st.session_state["tel"]["tof"]    = 350
        st.session_state.update({"flying": False, "mission_running": False, "mission_phase": "idle"})
        push_alert("🛬 [SIM] Landed.", "ok")
        return
    t = st.session_state.get("tello")
    if t:
        try: t.land()
        except: pass
    st.session_state.update({"flying": False, "mission_running": False, "mission_phase": "idle"})
    push_alert("🛬 Landed.", "ok")

def _do_emergency():
    if st.session_state.get("sim_mode"):
        stop_camera()
        st.session_state["tel"]["height"] = 0
        st.session_state.update({"emergency_stop": True, "flying": False,
                                   "mission_running": False, "mission_phase": "idle"})
        push_alert("🚨 [SIM] EMERGENCY STOP — Motors cut!", "crit")
        return
    t = st.session_state.get("tello")
    if t:
        try: t.emergency()
        except: pass
    stop_camera()
    st.session_state.update({"emergency_stop": True, "flying": False,
                               "mission_running": False, "mission_phase": "idle"})
    push_alert("🚨 EMERGENCY STOP — MOTORS CUT!", "crit")

def _do_move(direction, dist=None):
    spd = dist or st.session_state.get("move_speed", 30)
    if st.session_state.get("sim_mode"):
        tel = st.session_state["tel"]
        if   direction == "up":    tel["height"] = min(400, tel.get("height", 0) + spd)
        elif direction == "down":  tel["height"] = max(20,  tel.get("height", 0) - spd)
        elif direction == "cw":    tel["yaw"]    = (tel.get("yaw", 0) + 45) % 360
        elif direction == "ccw":   tel["yaw"]    = (tel.get("yaw", 0) - 45) % 360
        elif direction in ("fwd","back","left","right"):
            tel["speed_x"] = spd if direction in ("fwd","right") else -spd
        push_alert(f"↗️ [SIM] {direction.upper()} {spd}cm", "info")
        return
    t   = st.session_state.get("tello")
    if t is None: return
    if not is_safe_to_move(direction): return
    try:
        {"up":    lambda: t.move_up(spd),
         "down":  lambda: t.move_down(spd),
         "fwd":   lambda: t.move_forward(spd),
         "back":  lambda: t.move_back(spd),
         "left":  lambda: t.move_left(spd),
         "right": lambda: t.move_right(spd),
         "cw":    lambda: t.rotate_clockwise(45),
         "ccw":   lambda: t.rotate_counter_clockwise(45),
        }[direction]()
        push_alert(f"↗️ {direction.upper()} {spd}cm", "info")
    except Exception as e:
        push_alert(f"Move error: {e}", "warn")

def start_ai_mission():
    if not st.session_state.get("flying"):
        push_alert("Drone must be airborne first.", "warn"); return
    st.session_state.update({"mission_running": True, "emergency_stop": False,
                               "mission_id": str(uuid.uuid4())[:8],
                               "reinspect_queue": [], "ai_path_current_wp": 0})
    push_alert(f"🤖 Launching AI [{st.session_state['ai_path_mode']}] mission…", "ok")
    threading.Thread(target=_ai_autonomous_mission_thread, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def kpi(val, label, sub="", color="#00d4ff"):
    sub_html = f'<div style="font-size:0.65rem;color:#3a6a7a;margin-top:2px">{sub}</div>' if sub else ""
    return (f'<div class="kpi-card">'
            f'<div class="kpi-val" style="color:{color}">{val}</div>'
            f'<div class="kpi-lbl">{label}</div>{sub_html}</div>')

def pill(text, kind="off"):
    return f'<span class="s-pill s-{kind}">{text}</span>'

def mission_elapsed():
    t0 = st.session_state.get("mission_start_time")
    if not t0: return "--:--:--"
    d = datetime.datetime.now() - t0
    m, s = divmod(int(d.total_seconds()), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def severity_badge(sev):
    return f'<span class="badge badge-{sev}">{sev.upper()}</span>'

def export_defect_csv():
    rows = st.session_state["defect_log"]
    if not rows: return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
    return buf.getvalue()

def export_full_report():
    stats = st.session_state["session_stats"]
    return json.dumps({
        "project":       st.session_state.get("project_name",""),
        "inspector":     st.session_state.get("inspector_name",""),
        "building_id":   st.session_state.get("building_id",""),
        "mission_id":    st.session_state.get("mission_id",""),
        "ai_path_mode":  st.session_state.get("ai_path_mode",""),
        "database":      st.session_state.get("db_type",""),
        "yolo_model":    st.session_state.get("yolo_model_path",""),
        "report_time":   datetime.datetime.now().isoformat(),
        "session_stats": stats,
        "defect_log":    st.session_state["defect_log"],
        "flight_log":    st.session_state["flight_log"][-200:],
        "alerts":        st.session_state["alerts"][:100],
        "notes":         st.session_state.get("report_notes",""),
    }, indent=2)

def ai_path_minimap_svg(waypoints, current_wp, rows, cols):
    if not waypoints:
        return '<div class="mini-map" style="height:80px">No path planned yet</div>'
    W, H, margin = 240, 140, 12
    xs = [w["x"] for w in waypoints]; ys = [w["y"] for w in waypoints]
    min_x, max_x = min(xs), max(xs)+1
    min_y, max_y = min(ys), max(ys)+1
    def nx(x): return margin + (x-min_x)/max(max_x-min_x,1)*(W-2*margin)
    def ny(y): return margin + (y-min_y)/max(max_y-min_y,1)*(H-2*margin)
    lines = "".join(f'<line x1="{nx(waypoints[i-1]["x"]):.0f}" y1="{ny(waypoints[i-1]["y"]):.0f}" x2="{nx(waypoints[i]["x"]):.0f}" y2="{ny(waypoints[i]["y"]):.0f}" stroke="#0d4f6e" stroke-width="1"/>' for i in range(1, len(waypoints)))
    dots  = "".join(f'<circle cx="{nx(wp["x"]):.0f}" cy="{ny(wp["y"]):.0f}" r="{4 if i==current_wp else 2.5}" fill="{"#00d4ff" if i==current_wp else "#00ff88" if i<current_wp else "#1a3a4a"}"/>' for i, wp in enumerate(waypoints))
    return f"""
<div class="mini-map">
  <svg width="{W}" height="{H}" style="display:block;margin:0 auto">
    <rect width="{W}" height="{H}" fill="#030710" rx="4"/>
    {lines}{dots}
  </svg>
  <div style="margin-top:4px;font-size:0.6rem;color:#4a8fa8">
    WP {current_wp}/{len(waypoints)} · {waypoints[current_wp]['label'] if current_wp < len(waypoints) else 'DONE'}
  </div>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
drain_alert_queue()

if st.session_state.get("cam_active"):
    if AUTOREFRESH_AVAILABLE:
        _st_autorefresh(interval=120, key="cam_refresh")
    else:
        # Fallback: inject a meta-refresh so the page still updates
        _stc.html(
            '<script>setTimeout(function(){window.parent.location.reload();},150);</script>',
            height=0
        )

# ── Header ────────────────────────────────────────────────────────────────────
tel    = st.session_state["tel"]
stats  = st.session_state["session_stats"]
conn   = st.session_state["connected"]
flying = st.session_state["flying"]
mission = st.session_state["mission_running"]
phase  = st.session_state["mission_phase"]
bat    = tel.get("battery", 0)
safety = st.session_state.get("safety_status", "SAFE")
model_loaded = os.path.exists(st.session_state.get("yolo_model_path", ""))

st.markdown(f"""
<div class="hud-header">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div class="hud-title">🚁 TELLO INSPECTOR PRO
        <span style="font-size:1rem;color:#00ff88">v4</span>
      </div>
      <div class="hud-subtitle">Custom YOLOv8 · CV2 Detection · MongoDB/MySQL · AI Path · ANA Assistant{"  ·  🖥️ SIMULATION MODE" if st.session_state.get("sim_mode") else ""}</div>
    </div>
    <div style="text-align:right">
      <div class="hud-version">SYS: {datetime.datetime.now().strftime('%H:%M:%S')}</div>
      <div class="hud-version" style="color:{'#00ff88' if model_loaded else '#ffa502'}">
        MODEL: {'✅ ' + os.path.basename(st.session_state.get('yolo_model_path','')) if model_loaded else '⚠️ yolov8n fallback'}
      </div>
      <div class="hud-version" style="color:{'#00ff88' if safety=='SAFE' else '#ffa502' if safety=='CAUTION' else '#ff4757'}">
        SAFETY: {safety}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Status pills ──────────────────────────────────────────────────────────────
sb = st.columns(9)
sb[0].markdown(pill("CONNECTED" if conn else "OFFLINE", "on" if conn else "off"), unsafe_allow_html=True)
sb[1].markdown(pill("FLYING" if flying else "GROUNDED", "on" if flying else "off"), unsafe_allow_html=True)
sb[2].markdown(pill(phase.upper() if mission else "IDLE", "on" if mission else "off"), unsafe_allow_html=True)
sb[3].markdown(pill(f"BAT {bat:.0f}%", "on" if bat>50 else "warn" if bat>20 else "crit"), unsafe_allow_html=True)
sb[4].markdown(pill(f"ALT {tel.get('height',0)}cm", "on" if flying else "off"), unsafe_allow_html=True)
sb[5].markdown(pill(f"TOF {tel.get('tof',0)}cm", "on" if tel.get('tof',0)>100 else "warn" if tel.get('tof',0)>50 else "crit"), unsafe_allow_html=True)
sb[6].markdown(pill(f"DEFECTS {stats['defects_found']}", "crit" if stats["defects_found"]>0 else "off"), unsafe_allow_html=True)
_health = st.session_state.get("stream_health","—")
sb[7].markdown(pill(f"STREAM {_health}", {"OK":"on","WARN":"warn","ERROR":"crit"}.get(_health,"off")), unsafe_allow_html=True)
db_connected, db_total, _, _, _ = get_db_stats(st.session_state.get("db_type","MongoDB"))
sb[8].markdown(pill(f"DB {st.session_state.get('db_type','MongoDB')[:3]}", "on" if db_connected else "off"), unsafe_allow_html=True)

st.markdown("<hr style='border-color:#0d4f6e;margin:5px 0 8px'>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
(tab_ctrl, tab_cam, tab_auto, tab_defects,
 tab_telem, tab_analytics, tab_db,
 tab_config, tab_report, tab_gallery, tab_weather, tab_ana) = st.tabs([
    "🎮 CONTROL",
    "📷 LIVE CAMERA",
    "🤖 AI AUTOPILOT",
    "🔍 DEFECTS",
    "📡 TELEMETRY",
    "📊 ANALYTICS",
    "🗄️ DATABASE",
    "⚙️ CONFIG",
    "📋 REPORT",
    "📸 GALLERY",
    "🌤️ WEATHER",
    "🤖 ANA AI",
])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — CONTROL
# ════════════════════════════════════════════════════════════════════════════
with tab_ctrl:
    ctrl_left, ctrl_mid, ctrl_right = st.columns([1, 1, 1], gap="medium")

    with ctrl_left:
        st.markdown('<div class="sec-hdr">📶 CONNECTION</div>', unsafe_allow_html=True)
        if not conn:
            st.markdown("""
<div style="background:#080e14;border:1px solid #0d4f6e;border-radius:6px;padding:10px 14px;font-size:0.78rem;margin-bottom:8px">
  <b style="color:#00d4ff">Real Drone Setup:</b><br>
  1. Power on Tello (LED blinks yellow)<br>
  2. Connect PC to <code style="color:#00d4ff">TELLO-XXXXXX</code> WiFi<br>
  3. Click Connect ↓
</div>
""", unsafe_allow_html=True)
            if st.button("🔗  CONNECT TO TELLO", key="btn_connect", use_container_width=True, type="primary"):
                if _do_connect(): st.rerun()
            st.markdown("<div style='text-align:center;color:#3a6a7a;font-size:0.7rem;padding:4px'>— or —</div>", unsafe_allow_html=True)
            if st.button("🖥️  SIMULATE (No Drone)", key="btn_sim", use_container_width=True):
                if _do_connect(sim=True): st.rerun()
            st.caption("Simulation uses your PC webcam (or synthetic video) with full detection, AI path planning, and telemetry.")
        else:
            sim_badge = ' <span style="background:#0d2a4e;border:1px solid #ffa502;border-radius:3px;padding:1px 6px;font-size:0.65rem;color:#ffa502;font-family:monospace">SIM</span>' if st.session_state.get("sim_mode") else ""
            st.success(f"🟢 {'[SIM] ' if st.session_state.get('sim_mode') else ''}Online — {bat:.0f}% bat | ETA: {battery_eta()}")
            st.markdown(f'<div style="font-size:0.72rem;color:#4a8fa8;margin-bottom:6px">Mode: <b style="color:{"#ffa502" if st.session_state.get("sim_mode") else "#00ff88"}">{"🖥️ SIMULATION" if st.session_state.get("sim_mode") else "🚁 REAL DRONE"}</b></div>', unsafe_allow_html=True)
            st.progress(int(bat)/100)
            c1, c2 = st.columns(2)
            c1.metric("Battery", f"{bat:.0f}%")
            c2.metric("ETA", battery_eta())
            if st.button("🔌 Disconnect", key="btn_disconnect", use_container_width=True):
                _do_disconnect(); st.rerun()

        st.markdown('<div class="sec-hdr">✈️ FLIGHT</div>', unsafe_allow_html=True)
        fb1, fb2 = st.columns(2)
        with fb1:
            if st.button("🛫 TAKEOFF", key="btn_takeoff", use_container_width=True, type="primary",
                         disabled=not conn or flying):
                _do_takeoff(); st.rerun()
            if st.button("🏠 RTL", key="btn_rtl", use_container_width=True, disabled=not flying):
                _do_land(); st.rerun()
        with fb2:
            if st.button("🛬 LAND", key="btn_land", use_container_width=True, disabled=not flying):
                _do_land(); st.rerun()
            if st.button("🚨 E-STOP", key="btn_estop", use_container_width=True, disabled=not conn):
                _do_emergency(); st.rerun()

        st.markdown('<div class="sec-hdr">📷 CAMERA</div>', unsafe_allow_html=True)
        if not st.session_state["cam_active"]:
            if not conn:
                st.caption("⚠️ Connect Tello or click **Simulate** above to enable camera.")
            cam_btn_label = "▶️ START CAMERA" if conn else "▶️ START SIM CAMERA"
            if st.button(cam_btn_label, key="btn_cam_on", use_container_width=True,
                         type="primary", disabled=not conn):
                start_camera(); st.rerun()
        else:
            if st.button("⏹️ STOP CAMERA", key="btn_cam_off", use_container_width=True):
                stop_camera(); st.rerun()

        cam_c1, cam_c2 = st.columns(2)
        with cam_c1:
            if st.button("📸 Screenshot", key="btn_snap", use_container_width=True,
                         disabled=not st.session_state["cam_active"]):
                capture_screenshot(); st.rerun()
        with cam_c2:
            lbl = "⏸️ Pause Det" if st.session_state["det_enabled"] else "▶️ Resume Det"
            if st.button(lbl, key="btn_det_toggle", use_container_width=True,
                         disabled=not st.session_state["cam_active"]):
                st.session_state["det_enabled"] = not st.session_state["det_enabled"]; st.rerun()

        if not st.session_state.get("recording"):
            if st.button("🔴 Start Recording", key="btn_rec", use_container_width=True,
                         disabled=not st.session_state["cam_active"]):
                start_recording(); st.rerun()
        else:
            st.markdown('<span class="rec-dot"></span>**RECORDING…**', unsafe_allow_html=True)
            if st.button("⏹️ Stop Recording", key="btn_rec_stop", use_container_width=True):
                stop_recording(); st.rerun()
        vf = st.session_state.get("video_frames", [])
        if vf:
            st.download_button("⬇️ Download Recording", data=build_mjpeg_download(),
                               file_name=f"tello_{datetime.datetime.now().strftime('%H%M%S')}.mjpeg",
                               mime="multipart/x-mixed-replace", key="dl_vid", use_container_width=True)

    with ctrl_mid:
        st.markdown('<div class="sec-hdr">🕹️ FLIGHT CONTROLS</div>', unsafe_allow_html=True)
        dis = not flying or mission
        speed_val = st.select_slider("Move Speed (cm)", options=[20,30,40,50,80,100],
                                     value=st.session_state.get("move_speed",30), key="spd_slider")
        st.session_state["move_speed"] = speed_val

        st.markdown(f"""
<div style="background:#080e14;border:1px solid #0d4f6e;border-radius:8px;padding:12px;margin:4px 0;text-align:center">
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#4a8fa8;letter-spacing:2px;margin-bottom:8px">D-PAD CONTROLS</div>
  <div style="display:inline-grid;grid-template-columns:repeat(3,44px);gap:3px">
    <div></div>
    <div style="background:linear-gradient(135deg,#080e14,#0d2040);border:1px solid #00d4ff;border-radius:5px;height:44px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;color:#00d4ff">▲</div>
    <div></div>
    <div style="background:linear-gradient(135deg,#080e14,#0d2040);border:1px solid #0d4f6e;border-radius:5px;height:44px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;color:#c8d8e4">◀</div>
    <div style="background:#030710;border:1px solid #1a3a4a;border-radius:5px;height:44px;display:flex;align-items:center;justify-content:center;color:#1a3a4a;font-size:0.5rem">●</div>
    <div style="background:linear-gradient(135deg,#080e14,#0d2040);border:1px solid #0d4f6e;border-radius:5px;height:44px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;color:#c8d8e4">▶</div>
    <div></div>
    <div style="background:linear-gradient(135deg,#080e14,#0d2040);border:1px solid #0d4f6e;border-radius:5px;height:44px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;color:#c8d8e4">▼</div>
    <div></div>
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#3a6a7a;margin:6px 0 4px">ALT &amp; YAW</div>
  <div style="display:inline-grid;grid-template-columns:repeat(4,44px);gap:3px">
    <div style="background:rgba(0,255,136,0.06);border:1px solid #00ff88;border-radius:5px;height:40px;display:flex;align-items:center;justify-content:center;font-size:1rem;color:#00ff88">⬆</div>
    <div style="background:rgba(255,71,87,0.06);border:1px solid #ff4757;border-radius:5px;height:40px;display:flex;align-items:center;justify-content:center;font-size:1rem;color:#ff4757">⬇</div>
    <div style="background:rgba(255,165,2,0.06);border:1px solid #ffa502;border-radius:5px;height:40px;display:flex;align-items:center;justify-content:center;font-size:1rem;color:#ffa502">↺</div>
    <div style="background:rgba(255,165,2,0.06);border:1px solid #ffa502;border-radius:5px;height:40px;display:flex;align-items:center;justify-content:center;font-size:1rem;color:#ffa502">↻</div>
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.55rem;color:#3a6a7a;margin-top:4px">STEP: {speed_val}cm · ROT: 45°</div>
</div>
""", unsafe_allow_html=True)

        r1 = st.columns(4)
        with r1[0]:
            if st.button("▲ FWD",  key="btn_fwd",  use_container_width=True, disabled=dis): _do_move("fwd")
        with r1[1]:
            if st.button("▼ BACK", key="btn_back", use_container_width=True, disabled=dis): _do_move("back")
        with r1[2]:
            if st.button("◀ LEFT", key="btn_left", use_container_width=True, disabled=dis): _do_move("left")
        with r1[3]:
            if st.button("▶ RIGHT",key="btn_right",use_container_width=True, disabled=dis): _do_move("right")
        r2 = st.columns(4)
        with r2[0]:
            if st.button("⬆ UP",  key="btn_up",  use_container_width=True, disabled=dis): _do_move("up")
        with r2[1]:
            if st.button("⬇ DOWN",key="btn_down",use_container_width=True, disabled=dis): _do_move("down")
        with r2[2]:
            if st.button("↺ CCW", key="btn_ccw", use_container_width=True, disabled=dis): _do_move("ccw")
        with r2[3]:
            if st.button("↻ CW",  key="btn_cw",  use_container_width=True, disabled=dis): _do_move("cw")

        safety_now = evaluate_safety()
        s_color = "safety-safe" if safety_now=="SAFE" else "safety-caution" if safety_now=="CAUTION" else "safety-danger"
        st.markdown(f"""
<div class="safety-bar" style="margin-top:8px">
  <div class="safety-dot {s_color}"></div>
  <div class="safety-text" style="color:{'#00ff88' if safety_now=='SAFE' else '#ffa502' if safety_now=='CAUTION' else '#ff4757'}">
    {safety_now}
  </div>
  <div style="color:#4a8fa8;font-family:'Share Tech Mono',monospace;font-size:0.65rem;margin-left:8px">
    ToF: {tel.get('tof',0)}cm · Alt: {tel.get('height',0)}cm
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">🔔 LIVE ALERTS</div>', unsafe_allow_html=True)
        for a in st.session_state["alerts"][:8]:
            css = {"crit":"alert-crit","warn":"alert-warn","ok":"alert-ok"}.get(a["level"],"alert-info")
            st.markdown(f'<div class="{css}">[{a["ts"]}] {a["msg"]}</div>', unsafe_allow_html=True)

    with ctrl_right:
        st.markdown('<div class="sec-hdr">📷 LIVE FEED</div>', unsafe_allow_html=True)
        if st.session_state.get("cam_active"):
            live_camera_component(height=260)
            with _MJPEG_LOCK:
                meta = dict(_MJPEG_META)
            live_def = meta.get("defects", [])
            if live_def:
                badge_html = ""
                for d in sorted(live_def, key=lambda x: SEVERITY_RANK.get(x["severity"],0), reverse=True):
                    badge_html += severity_badge(d["severity"]) + f" {d['type']} {d['conf']:.0%}<br>"
                st.markdown(badge_html, unsafe_allow_html=True)
            st.caption(f"FPS: {st.session_state.get('last_fps',0):.1f} · Frame #{st.session_state['frame_idx']}")
        else:
            st.markdown("""
<div class="cam-panel">
  <div class="cam-offline">
    <div style="font-size:1.5rem">📷</div>
    <div>CAMERA OFFLINE</div>
    <div style="color:#1a3a4a">Connect drone &amp; start camera</div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">🎛️ CAMERA</div>', unsafe_allow_html=True)
        z_col, f_col = st.columns(2)
        with z_col:
            z = st.slider("Zoom", 1.0, 4.0, st.session_state["zoom_level"], 0.25, key="z_ctrl")
            st.session_state["zoom_level"] = z
        with f_col:
            fi = st.selectbox("Filter", CAM_FILTERS, index=CAM_FILTERS.index(st.session_state["cam_filter"]), key="f_ctrl")
            st.session_state["cam_filter"] = fi

        st.markdown('<div class="sec-hdr">🗄️ DATABASE</div>', unsafe_allow_html=True)
        db_sel = st.selectbox("Active DB", ["MongoDB","MySQL"], key="db_ctrl_sel")
        st.session_state["db_type"] = db_sel
        st.markdown(f"""
<div style="background:#080e14;border:1px solid #0d4f6e;border-radius:5px;padding:8px;font-family:'Share Tech Mono',monospace;font-size:0.68rem;line-height:1.7">
  <div>Status: <span class="{'db-ok' if db_connected else 'db-off'}">{'● Connected' if db_connected else '○ Offline (using memory)'}</span></div>
  <div>Saved records: <span style="color:#00d4ff">{st.session_state.get('db_save_count',0)}</span></div>
  <div>Auto-save: <span style="color:#00ff88">{'ON' if st.session_state.get('db_auto_save') else 'OFF'}</span></div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — FULL LIVE CAMERA
# ════════════════════════════════════════════════════════════════════════════
with tab_cam:
    cam_l, cam_r = st.columns([3, 1], gap="medium")
    with cam_r:
        st.markdown('<div class="sec-hdr">🎛️ CONTROLS</div>', unsafe_allow_html=True)
        if not st.session_state["cam_active"]:
            if not conn:
                st.caption("⚠️ Connect Tello or use Simulate mode first.")
            if st.button("▶️ Start Camera", key="btn_cam2_on", use_container_width=True,
                         type="primary", disabled=not conn):
                start_camera(); st.rerun()
        else:
            if st.button("⏹️ Stop Camera", key="btn_cam2_off", use_container_width=True):
                stop_camera(); st.rerun()
        if st.button("📸 Capture", key="btn_snap2", use_container_width=True, disabled=not st.session_state["cam_active"]):
            capture_screenshot(); st.rerun()
        if st.button("🔄 Refresh", key="btn_refresh2", use_container_width=True): st.rerun()
        zoom_v = st.slider("Zoom", 1.0, 4.0, st.session_state["zoom_level"], 0.25, key="zoom_sl2")
        st.session_state["zoom_level"] = zoom_v
        filt_v = st.selectbox("Filter", CAM_FILTERS, index=CAM_FILTERS.index(st.session_state["cam_filter"]), key="filt_sl2")
        st.session_state["cam_filter"] = filt_v
        st.markdown('<div class="sec-hdr">🔍 DETECTION</div>', unsafe_allow_html=True)
        det_on = st.toggle("CV2 Detection", value=st.session_state["det_enabled"], key="det_tog")
        st.session_state["det_enabled"] = det_on
        yolo_lbl = "🤖 YOLO (best.pt)" + (" ✅" if YOLO_AVAILABLE else " (install ultralytics)")
        yolo_on = st.toggle(yolo_lbl, value=st.session_state.get("yolo_enabled",True), key="yolo_tog", disabled=not YOLO_AVAILABLE)
        st.session_state["yolo_enabled"] = yolo_on
        if YOLO_AVAILABLE:
            model_path = st.session_state.get("yolo_model_path","")
            st.caption(f"Model: {model_path}")
        st.markdown('<div class="sec-hdr">📊 STATS</div>', unsafe_allow_html=True)
        st.metric("FPS",     f"{st.session_state.get('last_fps',0):.1f}")
        st.metric("Frame #", st.session_state["frame_idx"])
        st.metric("Defects", stats["defects_found"])
        st.metric("DB saved",st.session_state.get("db_save_count",0))
    with cam_l:
        if not st.session_state["cam_active"]:
            st.info("📷 Camera is off. Connect Tello and press ▶️ Start Camera.")
            st.markdown("""<div class="cam-panel"><div class="cam-offline">
  <div style="font-size:2rem">📷</div><div>CAMERA OFFLINE</div>
  <div style="color:#1a3a4a;font-size:0.75rem">Connect drone &amp; start camera</div>
</div></div>""", unsafe_allow_html=True)
        else:
            live_camera_component(height=440)
            extras = []
            if st.session_state["zoom_level"] > 1.0: extras.append(f"🔭 {st.session_state['zoom_level']:.1f}×")
            if st.session_state["cam_filter"] != "Normal": extras.append(f"🎨 {st.session_state['cam_filter']}")
            if st.session_state.get("recording"): extras.append("🔴 REC")
            extras.append("🤖 YOLO ON" if (YOLO_AVAILABLE and st.session_state.get("yolo_enabled")) else "🤖 YOLO N/A")
            st.caption(f"🟢 Live · Frame #{st.session_state['frame_idx']} · FPS {st.session_state.get('last_fps',0):.1f}" + (" · " + " · ".join(extras) if extras else ""))
            with _MJPEG_LOCK:
                live_meta = dict(_MJPEG_META)
            ld = live_meta.get("defects", [])
            if ld:
                bh = "<b>Live detections:</b> "
                for d in sorted(ld, key=lambda x: SEVERITY_RANK.get(x["severity"],0), reverse=True):
                    bh += severity_badge(d["severity"]) + f" {d['type']} {d['conf']:.0%} &nbsp;"
                st.markdown(bh, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — AI AUTOPILOT
# ════════════════════════════════════════════════════════════════════════════
with tab_auto:
    auto_l, auto_r = st.columns([2, 1], gap="medium")
    with auto_l:
        st.markdown('<div class="sec-hdr">🤖 AI PATH PLANNING</div>', unsafe_allow_html=True)
        st.markdown("**Select Path Mode:**")
        mode_cols = st.columns(len(AI_PATH_MODES))
        for i, mode in enumerate(AI_PATH_MODES):
            with mode_cols[i]:
                is_active = st.session_state["ai_path_mode"] == mode
                style = "border-color:#00d4ff;color:#00d4ff;background:rgba(0,212,255,0.08)" if is_active else "border-color:#0d4f6e;color:#4a8fa8"
                st.markdown(f'<div style="background:#080e14;border:1px solid;border-radius:5px;padding:6px;text-align:center;font-family:\'Share Tech Mono\',monospace;font-size:0.6rem;{style}">{mode}</div>', unsafe_allow_html=True)
                if st.button(f"Select", key=f"pm_{i}", use_container_width=True):
                    st.session_state["ai_path_mode"] = mode; st.rerun()

        selected_mode = st.session_state["ai_path_mode"]
        mode_info = {
            "Grid Scan":        "🔲 Serpentine boustrophedon sweep — maximum coverage.",
            "Perimeter Loop":   "🔁 Safe perimeter loop around the building facade.",
            "Spiral Inward":    "🌀 Converging spiral from outside in.",
            "Zigzag":           "⚡ Aggressive zigzag for fastest area coverage.",
            "Return to Home":   "🏠 Immediately descends and lands.",
            "Custom Waypoints": "📍 Use manually saved custom waypoints.",
        }
        st.info(f"**{selected_mode}:** {mode_info.get(selected_mode,'')}")

        p1, p2, p3 = st.columns(3)
        with p1: st.session_state["survey_rows"] = st.number_input("Rows", 1, 10, st.session_state["survey_rows"], key="rows_ai")
        with p2: st.session_state["survey_cols"] = st.number_input("Columns", 1, 10, st.session_state["survey_cols"], key="cols_ai")
        with p3: st.session_state["survey_altitude"] = st.number_input("Altitude (cm)", 50, 400, st.session_state["survey_altitude"], key="alt_ai")
        p4, p5 = st.columns(2)
        with p4: st.session_state["survey_speed"] = st.slider("Speed (cm/s)", 10, 100, st.session_state["survey_speed"], key="spd_ai")
        with p5: st.session_state["hover_duration"] = st.slider("Defect hover (s)", 2, 15, st.session_state["hover_duration"], key="hdur_ai")

        if st.button("👁️ Preview Path", key="btn_preview_path", use_container_width=True):
            wps = generate_ai_path(selected_mode, st.session_state["survey_rows"],
                                   st.session_state["survey_cols"], 80, st.session_state["survey_altitude"])
            st.session_state["ai_path_waypoints"] = wps
            st.session_state["ai_path_current_wp"] = 0
            st.success(f"✅ {len(wps)} waypoints generated for {selected_mode}")

        wps = st.session_state.get("ai_path_waypoints", [])
        cur_wp = st.session_state.get("ai_path_current_wp", 0)
        if wps:
            st.markdown(ai_path_minimap_svg(wps, cur_wp, st.session_state["survey_rows"], st.session_state["survey_cols"]), unsafe_allow_html=True)

        mc1, mc2 = st.columns(2)
        with mc1:
            if st.button("🚀 START AI MISSION", key="btn_ai_start", use_container_width=True, type="primary",
                         disabled=not flying or mission):
                start_ai_mission(); st.rerun()
        with mc2:
            if st.button("⬛ ABORT MISSION", key="btn_ai_abort", use_container_width=True, disabled=not mission):
                st.session_state["mission_running"] = False
                push_alert("Mission aborted.", "warn"); st.rerun()

        if mission:
            st.progress(cur_wp / max(len(wps), 1))
            st.caption(f"Progress: {cur_wp}/{len(wps)} · Phase: {phase.upper()}")

    with auto_r:
        st.markdown('<div class="sec-hdr">🛡️ SAFETY</div>', unsafe_allow_html=True)
        st.session_state["ai_obstacle_detect"] = st.toggle("Obstacle detect (ToF)", value=st.session_state["ai_obstacle_detect"], key="obs_tog")
        st.session_state["ai_tof_safe_dist"] = st.slider("Min ToF dist (cm)", 20, 200, st.session_state["ai_tof_safe_dist"], key="tof_safe")
        st.session_state["ai_safety_min_alt"] = st.slider("Min altitude (cm)", 20, 150, st.session_state["ai_safety_min_alt"], key="min_alt")
        st.session_state["ai_safety_max_alt"] = st.slider("Max altitude (cm)", 100, 500, st.session_state["ai_safety_max_alt"], key="max_alt")
        st.session_state["min_battery_rtl"] = st.slider("RTL battery %", 5, 40, st.session_state["min_battery_rtl"], key="rtl_bat")
        st.session_state["auto_rtl"] = st.toggle("Auto-RTL on low battery", value=st.session_state["auto_rtl"], key="auto_rtl_tog")

        st.markdown('<div class="sec-hdr">📊 MISSION STATUS</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:#080e14;border:1px solid #0d4f6e;border-radius:6px;padding:12px;font-family:'Share Tech Mono',monospace;font-size:0.72rem;line-height:1.8">
  <div>Phase: <span style="color:#00d4ff">{phase.upper()}</span></div>
  <div>Mode: <span style="color:#ffa502">{st.session_state['ai_path_mode']}</span></div>
  <div>WP: <span style="color:#00d4ff">{cur_wp}/{len(wps)}</span></div>
  <div>Missions: <span style="color:#00ff88">{stats['missions_completed']}</span></div>
  <div>Elapsed: {mission_elapsed()}</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — DEFECT LOG
# ════════════════════════════════════════════════════════════════════════════
with tab_defects:
    log = st.session_state["defect_log"]
    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:10px">
  {kpi(stats['defects_found'],"Total",color="#ff4757")}
  {kpi(sum(1 for d in log if d['severity']=='critical'),"Critical",color="#ff4757")}
  {kpi(sum(1 for d in log if d['severity']=='high'),"High",color="#ffa502")}
  {kpi(sum(1 for d in log if d.get('source')=='yolo'),"YOLO Det.",color="#00d4ff")}
  {kpi(st.session_state.get('db_save_count',0),"DB Saved",color="#00ff88")}
</div>
""", unsafe_allow_html=True)

    if log:
        st.download_button("⬇️ Export CSV", data=export_defect_csv(),
                           file_name=f"defects_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", key="dl_csv")
        df = pd.DataFrame(log[-200:])
        for col in ["time","type","severity","conf","alt_cm","source"]:
            if col not in df.columns: df[col] = "—"
        st.dataframe(df[["time","type","severity","conf","alt_cm","source","yaw_deg"]], use_container_width=True, height=420)
    else:
        st.info("No defects detected yet. Start the camera and fly to begin scanning.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — TELEMETRY
# ════════════════════════════════════════════════════════════════════════════
with tab_telem:
    t1, t2 = st.columns(2, gap="medium")
    with t1:
        st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:8px">
  {kpi(f"{tel.get('battery',0)}%","Battery",color="#00ff88")}
  {kpi(f"{tel.get('height',0)} cm","Altitude",color="#00d4ff")}
  {kpi(f"{tel.get('tof',0)} cm","ToF Distance",color="#ffa502")}
  {kpi(f"{tel.get('yaw',0):.0f}°","Yaw",color="#00d4ff")}
  {kpi(f"{tel.get('pitch',0):.0f}°","Pitch",color="#c8d8e4")}
  {kpi(f"{tel.get('roll',0):.0f}°","Roll",color="#c8d8e4")}
</div>
""", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Flight Time", f"{tel.get('flight_time',0)}s")
        c2.metric("Distance (m)", f"{stats['flight_distance_m']:.1f}")
        c3.metric("Temp", f"{tel.get('temp_lo',0)}-{tel.get('temp_hi',0)}°C")
    with t2:
        fl = st.session_state.get("flight_log", [])
        if fl and PLOTLY_AVAILABLE:
            df_fl = pd.DataFrame(fl[-300:])
            if "height" in df_fl.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df_fl["height"], name="Altitude (cm)", line=dict(color="#00d4ff")))
                if "battery" in df_fl.columns:
                    fig.add_trace(go.Scatter(y=df_fl["battery"], name="Battery %", line=dict(color="#00ff88"), yaxis="y2"))
                fig.update_layout(height=240, paper_bgcolor="#050a0e", plot_bgcolor="#080e14",
                                  font=dict(color="#c8d8e4", size=10),
                                  yaxis=dict(gridcolor="#0d4f6e"),
                                  yaxis2=dict(overlaying="y", side="right"),
                                  margin=dict(l=30,r=30,t=20,b=20),
                                  legend=dict(bgcolor="#080e14"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Flight log chart will appear once flying.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    log = st.session_state["defect_log"]
    if not log:
        st.info("No data yet — start a mission.")
    else:
        df = pd.DataFrame(log)
        a1, a2 = st.columns(2, gap="medium")
        with a1:
            if PLOTLY_AVAILABLE and "type" in df.columns:
                fig = px.pie(df, names="type", title="Defects by Type",
                             color_discrete_sequence=["#ff4757","#ffa502","#00d4ff","#00ff88","#7bed9f","#eccc68","#a29bfe","#fd79a8"])
                fig.update_layout(height=280, paper_bgcolor="#050a0e", font=dict(color="#c8d8e4"), margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)
        with a2:
            if PLOTLY_AVAILABLE and "severity" in df.columns:
                fig2 = px.bar(df.groupby("severity").size().reset_index(name="count"),
                              x="severity", y="count", title="Defects by Severity",
                              color="severity",
                              color_discrete_map={"critical":"#ff4757","high":"#ffa502","medium":"#3742fa","low":"#2ed573"})
                fig2.update_layout(height=280, paper_bgcolor="#050a0e", plot_bgcolor="#080e14",
                                   font=dict(color="#c8d8e4"), showlegend=False, margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig2, use_container_width=True)

        # Source breakdown
        if PLOTLY_AVAILABLE and "source" in df.columns:
            fig3 = px.pie(df, names="source", title="Detection Source (YOLO vs CV2)",
                          color_discrete_sequence=["#00d4ff","#00ff88"])
            fig3.update_layout(height=220, paper_bgcolor="#050a0e", font=dict(color="#c8d8e4"), margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 7 — DATABASE
# ════════════════════════════════════════════════════════════════════════════
with tab_db:
    db_l, db_r = st.columns([2, 1], gap="medium")
    with db_l:
        st.markdown('<div class="sec-hdr">🗄️ DATABASE STATUS</div>', unsafe_allow_html=True)
        db_type = st.selectbox("Database Engine", ["MongoDB", "MySQL"],
                               index=["MongoDB","MySQL"].index(st.session_state.get("db_type","MongoDB")),
                               key="db_main_sel")
        st.session_state["db_type"] = db_type

        connected, total, crit, mod, recent = get_db_stats(db_type)
        if connected:
            st.success(f"✅ {db_type} connected · {total} total records")
            st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0">
  {kpi(total,"Total Records",color="#00d4ff")}
  {kpi(crit,"Critical",color="#ff4757")}
  {kpi(mod,"Moderate",color="#ffa502")}
</div>
""", unsafe_allow_html=True)
            if recent:
                st.markdown('<div class="sec-hdr">🕐 RECENT RECORDS</div>', unsafe_allow_html=True)
                df_recent = pd.DataFrame(recent)
                st.dataframe(df_recent, use_container_width=True, height=300)
        else:
            st.warning(f"⚠️ {db_type} not connected. Detections are stored in session memory only.")
            if db_type == "MongoDB":
                st.info("Start MongoDB: `mongod --dbpath /data/db`")
            else:
                st.info("Start MySQL: `mysql -u root` then `CREATE DATABASE crack_db;`\n\n"
                        "Create table:\n```sql\nCREATE TABLE cracks (id INT AUTO_INCREMENT PRIMARY KEY, timestamp VARCHAR(30), type VARCHAR(50), severity VARCHAR(20), confidence FLOAT, altitude_cm INT, image_path VARCHAR(200));\n```")

        st.markdown('<div class="sec-hdr">⚙️ SETTINGS</div>', unsafe_allow_html=True)
        st.session_state["db_auto_save"] = st.toggle("Auto-save detections to DB", value=st.session_state.get("db_auto_save", True), key="db_autosave")
        st.caption(f"Saved this session: {st.session_state.get('db_save_count',0)} records")

        if st.button("💾 Manual Save All Session Detections", key="btn_save_all"):
            for d in st.session_state.get("defect_log", []):
                save_detection_db(db_type, d["type"], d["severity"],
                                  d.get("conf", 0), d.get("alt_cm", 0))
            st.session_state["db_save_count"] = len(st.session_state.get("defect_log", []))
            st.success(f"Saved {len(st.session_state.get('defect_log',[]))} records to {db_type}")

    with db_r:
        st.markdown('<div class="sec-hdr">📁 CAPTURES FOLDER</div>', unsafe_allow_html=True)
        captures = sorted([f for f in os.listdir("captures") if f.endswith(".jpg")], reverse=True) if os.path.exists("captures") else []
        st.metric("Saved Images", len(captures))
        if captures:
            st.caption("Latest captures:")
            for f in captures[:5]:
                st.text(f"📷 {f}")

        st.markdown('<div class="sec-hdr">🔌 REQUIREMENTS</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:#080e14;border:1px solid #0d4f6e;border-radius:6px;padding:10px;font-family:'Share Tech Mono',monospace;font-size:0.68rem;line-height:1.8">
  <div>pymongo: <span class="{'db-ok' if MONGO_AVAILABLE else 'db-off'}">{'✅' if MONGO_AVAILABLE else '❌ pip install pymongo'}</span></div>
  <div>mysql-connector: <span class="{'db-ok' if MYSQL_AVAILABLE else 'db-off'}">{'✅' if MYSQL_AVAILABLE else '❌ pip install mysql-connector-python'}</span></div>
  <div>reportlab: <span class="{'db-ok' if REPORTLAB_AVAILABLE else 'db-off'}">{'✅' if REPORTLAB_AVAILABLE else '❌ pip install reportlab'}</span></div>
  <div>ultralytics: <span class="{'db-ok' if YOLO_AVAILABLE else 'db-off'}">{'✅' if YOLO_AVAILABLE else '❌ pip install ultralytics'}</span></div>
  <div>opencv: <span class="{'db-ok' if CV2_AVAILABLE else 'db-off'}">{'✅' if CV2_AVAILABLE else '❌ pip install opencv-python'}</span></div>
  <div>djitellopy: <span class="{'db-ok' if TELLO_AVAILABLE else 'db-off'}">{'✅' if TELLO_AVAILABLE else '❌ pip install djitellopy'}</span></div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 8 — CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
with tab_config:
    cf1, cf2 = st.columns(2, gap="medium")
    with cf1:
        st.markdown('<div class="sec-hdr">🤖 YOLO MODEL</div>', unsafe_allow_html=True)
        st.info(f"Current model: `{st.session_state.get('yolo_model_path','')}`")

        custom_path = st.text_input(
            "Custom model path",
            value=st.session_state.get("yolo_model_path",""),
            key="model_path_cfg"
        )

        if st.button("🔄 Reload Model", key="btn_reload_model"):
            _reset_yolo_model()
            st.session_state["yolo_model_path"] = custom_path
            push_alert(f"YOLO model path updated: {custom_path}", "info")
            st.rerun()
        st.markdown('<div class="sec-hdr">🔍 DETECTION SETTINGS</div>', unsafe_allow_html=True)
        st.session_state["crack_sensitivity"]    = st.slider("Crack sensitivity", 10, 80, st.session_state["crack_sensitivity"], key="sens_cfg")
        st.session_state["min_defect_area"]      = st.slider("Min defect area (px)", 30, 500, st.session_state["min_defect_area"], key="area_cfg")
        st.session_state["confidence_threshold"] = st.slider("Confidence threshold", 0.2, 0.9, st.session_state["confidence_threshold"], 0.05, key="conf_cfg")

        st.markdown('<div class="sec-hdr">🎛️ PID GAINS</div>', unsafe_allow_html=True)
        pid = st.session_state["pid_gains"]
        pid["kp"] = st.slider("Kp", 0.1, 1.0, pid["kp"], 0.05, key="kp_cfg")
        pid["ki"] = st.slider("Ki", 0.0, 0.1, pid["ki"], 0.005, key="ki_cfg")
        pid["kd"] = st.slider("Kd", 0.0, 0.5, pid["kd"], 0.01, key="kd_cfg")

    with cf2:
        st.markdown('<div class="sec-hdr">🏗️ PROJECT INFO</div>', unsafe_allow_html=True)
        st.session_state["project_name"]   = st.text_input("Project", st.session_state["project_name"], key="pname_cfg")
        st.session_state["building_id"]    = st.text_input("Building ID", st.session_state["building_id"], key="bid_cfg")
        st.session_state["inspector_name"] = st.text_input("Inspector", st.session_state["inspector_name"], key="iname_cfg")

        st.markdown('<div class="sec-hdr">📍 SITE LOCATION</div>', unsafe_allow_html=True)
        st.session_state["site_name"] = st.text_input("Site name", st.session_state["site_name"], key="sname_cfg")
        lc, rc = st.columns(2)
        st.session_state["site_lat"] = lc.number_input("Latitude", value=st.session_state["site_lat"], format="%.6f", key="lat_cfg")
        st.session_state["site_lon"] = rc.number_input("Longitude", value=st.session_state["site_lon"], format="%.6f", key="lon_cfg")

        st.markdown('<div class="sec-hdr">🗄️ DATABASE CONFIG</div>', unsafe_allow_html=True)
        st.info("MongoDB: default `mongodb://localhost:27017/`\nMySQL: default `root@localhost/crack_db`")

        if st.button("🗑️ Clear all session data", key="btn_clear", type="secondary"):
            import copy as _copy
            st.session_state["defect_log"]    = []
            st.session_state["flight_log"]    = []
            st.session_state["alerts"]        = []
            st.session_state["screenshots"]   = []
            st.session_state["session_stats"] = _copy.deepcopy(_SS_DEFAULTS["session_stats"])
            st.session_state["db_save_count"] = 0
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  TAB 9 — REPORT
# ════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown('<div class="sec-hdr">📋 INSPECTION REPORT</div>', unsafe_allow_html=True)
    st.session_state["report_notes"] = st.text_area("Inspector notes", st.session_state.get("report_notes",""), height=120, key="notes_rep")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.download_button("⬇️ Download JSON Report", data=export_full_report(),
                           file_name=f"inspection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                           mime="application/json", key="dl_json", use_container_width=True)
    with r2:
        st.download_button("⬇️ Download Defect CSV",
                           data=export_defect_csv() or "No defects",
                           file_name=f"defects_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", key="dl_csv_rep", use_container_width=True)
    with r3:
        if REPORTLAB_AVAILABLE:
            pdf_bytes = generate_pdf_report(
                st.session_state.get("db_type","MongoDB"),
                st.session_state.get("project_name",""),
                st.session_state.get("inspector_name",""),
                stats["defects_found"],
                sum(1 for d in st.session_state["defect_log"] if d["severity"]=="critical"),
                sum(1 for d in st.session_state["defect_log"] if d["severity"] in ("high","medium")),
                st.session_state["defect_log"],
                st.session_state.get("report_notes","")
            )
            if pdf_bytes:
                st.download_button("⬇️ Download PDF Report", data=pdf_bytes,
                                   file_name=f"inspection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                   mime="application/pdf", key="dl_pdf", use_container_width=True)
        else:
            st.button("⬇️ PDF (install reportlab)", disabled=True, use_container_width=True)
            st.caption("`pip install reportlab`")

    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0">
  {kpi(stats['defects_found'],"Defects Found",color="#ff4757")}
  {kpi(f"{stats['area_surveyed_m2']:.1f}","Area m²",color="#00d4ff")}
  {kpi(f"{stats['flight_distance_m']:.1f}","Distance m",color="#ffa502")}
  {kpi(stats['missions_completed'],"Missions",color="#00ff88")}
</div>
""", unsafe_allow_html=True)

    log = st.session_state["defect_log"]
    if log:
        summary: dict = {}
        for d in log:
            t = d["type"]
            if t not in summary:
                summary[t] = {"count":0,"severity":d["severity"],"max_conf":0.0}
            summary[t]["count"] += 1
            summary[t]["max_conf"] = max(summary[t]["max_conf"], d.get("conf",0))
        st.markdown("### Defect Summary")
        for dtype, info in sorted(summary.items(), key=lambda x: SEVERITY_RANK.get(x[1]["severity"],0), reverse=True):
            st.markdown(f"{severity_badge(info['severity'])} **{dtype.replace('_',' ').title()}** — "
                        f"{info['count']} occurrences, max conf: {info['max_conf']:.0%}", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 10 — GALLERY
# ════════════════════════════════════════════════════════════════════════════
with tab_gallery:
    shots = st.session_state.get("screenshots", [])
    disk_captures = sorted([f for f in os.listdir("captures") if f.endswith(".jpg")], reverse=True) if os.path.exists("captures") else []
    if not shots:
        st.info("No screenshots yet. Press 📸 Screenshot during a live camera session.")
    else:
        st.caption(f"{len(shots)} screenshots this session | {len(disk_captures)} saved to disk")
        cols = st.columns(3)
        for i, shot in enumerate(shots):
            with cols[i % 3]:
                st.markdown(f"""
<div class="shot-card">
  <img src="data:image/jpeg;base64,{shot['b64']}" style="width:100%;border-radius:4px">
  <div class="shot-ts">{shot['ts']} · ID:{shot['id']}</div>
  {''.join(severity_badge(d['severity']) + ' ' + d['type'] for d in shot.get('defects',[]))}
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 11 — WEATHER
# ════════════════════════════════════════════════════════════════════════════
with tab_weather:
    wx_l, wx_r = st.columns([2, 1], gap="medium")
    with wx_l:
        st.markdown('<div class="sec-hdr">🌤️ WEATHER & FLIGHT SAFETY</div>', unsafe_allow_html=True)
        if st.button("🔄 Refresh Weather", key="btn_wx", type="primary"):
            st.session_state["weather_ts"] = 0; st.rerun()
        wx = fetch_weather(st.session_state["site_lat"], st.session_state["site_lon"])
        if wx:
            fly_color = "#00ff88" if wx["fly_ok"] else "#ff4757"
            fly_label = "✅ SAFE TO FLY" if wx["fly_ok"] else "⛔ DO NOT FLY"
            st.markdown(f"""
<div class="wx-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <div class="wx-temp">{wx['temp']}°C</div>
      <div style="font-size:0.88rem;color:#c8d8e4">{_wx_icon(wx['code'])} {wx['desc']}</div>
      <div style="font-size:0.75rem;color:#4a8fa8;margin-top:4px">
        Feels {wx['feels_like']}°C · Humidity {wx['humidity']}%<br>
        Wind {wx['wind_kmh']} km/h {_wind_arrow(wx['wind_dir'])} · Cloud {wx['cloud']}%<br>
        Pressure {wx['pressure']} hPa · Visibility {wx['visibility']} km
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.85rem;font-weight:700;
                  color:{fly_color};border:1px solid {fly_color};padding:6px 12px;border-radius:5px">
        {fly_label}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
            advisories = []
            if wx["wind_kmh"] >= 25: advisories.append(("crit", f"🌬️ High wind {wx['wind_kmh']} km/h — grounded."))
            elif wx["wind_kmh"] >= 15: advisories.append(("warn", f"💨 Moderate wind {wx['wind_kmh']} km/h — caution."))
            else: advisories.append(("ok", f"✅ Wind {wx['wind_kmh']} km/h — OK."))
            if wx["precip"] > 0: advisories.append(("crit", "🌧️ Precipitation — DO NOT fly."))
            else: advisories.append(("ok", "✅ No precipitation."))
            if wx["visibility"] < 1.0: advisories.append(("warn", f"🌫️ Low visibility {wx['visibility']} km."))
            else: advisories.append(("ok", f"✅ Visibility {wx['visibility']} km."))
            for level, msg in advisories:
                css = {"crit":"alert-crit","warn":"alert-warn","ok":"alert-ok"}.get(level,"alert-info")
                st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Cannot fetch weather. Check internet connection.")
    with wx_r:
        st.markdown('<div class="sec-hdr">📍 SITE LOCATION</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="loc-card">
  <div style="color:#c8d8e4;font-weight:600;margin-bottom:8px">{st.session_state['site_name']}</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.82rem;color:#00d4ff">
    Lat: {st.session_state['site_lat']:.6f}°<br>
    Lon: {st.session_state['site_lon']:.6f}°
  </div>
  <div style="margin-top:8px">
    <a href="https://www.google.com/maps?q={st.session_state['site_lat']},{st.session_state['site_lon']}"
       target="_blank" style="color:#00d4ff;text-decoration:none;font-size:0.78rem">
      🗺️ Open in Google Maps →
    </a>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 12 — ANA AI ASSISTANT
# ════════════════════════════════════════════════════════════════════════════
with tab_ana:
    st.markdown('<div class="sec-hdr">🤖 ANA — AUTONOMOUS NAVIGATION ASSISTANT</div>', unsafe_allow_html=True)
    ana_l, ana_r = st.columns([3, 1], gap="medium")

    with ana_r:
        st.markdown("**Quick Actions**")
        quick_prompts = [
            ("🔍 Defect summary",    "Summarise all defects found and give a risk assessment."),
            ("🌤️ Flight safety",     "Based on current weather and battery, is it safe to fly?"),
            ("🤖 Best path mode",    f"What is the best AI path mode for a 5-storey facade? Currently using {st.session_state['ai_path_mode']}."),
            ("🛡️ Safety check",      "Check all safety parameters and tell me if anything needs attention."),
            ("🗄️ Database advice",   f"I'm using {st.session_state.get('db_type','MongoDB')}. How should I structure crack detection storage?"),
            ("🤖 YOLO vs CV2",       "When should I use YOLO detection vs CV2 for crack detection?"),
            ("📋 Write report",      "Draft a brief inspection report summary based on current session data."),
            ("⚡ Battery tips",      "How to maximise flight time during a full building inspection?"),
        ]
        for label, prompt in quick_prompts:
            if st.button(label, key=f"ana_q_{label}", use_container_width=True):
                st.session_state["ana_history"].append({"role":"user","content":prompt})
                with st.spinner("ANA thinking…"):
                    reply = ana_chat(prompt)
                st.session_state["ana_history"].append({"role":"assistant","content":reply})
                st.rerun()
        if st.button("🗑️ Clear Chat", key="ana_clear", use_container_width=True, type="secondary"):
            st.session_state["ana_history"] = []; st.rerun()
        st.markdown("---")
        st.caption("ANA is your AI co-pilot powered by Claude. She has real-time access to mission state, telemetry, database, YOLO model, and weather.")

    with ana_l:
        chat_history = st.session_state.get("ana_history", [])
        if not chat_history:
            st.markdown("""
<div style="text-align:center;padding:40px 20px;color:#4a8fa8">
  <div style="font-size:2.5rem;margin-bottom:10px">🤖</div>
  <div style="font-family:'Rajdhani',sans-serif;font-size:1.2rem;color:#c8d8e4;font-weight:700;letter-spacing:2px">ANA</div>
  <div style="font-size:0.8rem;margin-top:8px;line-height:1.8">
    Autonomous Navigation Assistant<br>
    Connected to: Drone · YOLO Model · MongoDB/MySQL · Weather
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            for msg in chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="ana-label">YOU</div><div class="ana-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ana-label">🤖 ANA</div><div class="ana-bubble-ai">{msg["content"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        ic, bc = st.columns([5, 1])
        with ic:
            user_input = st.text_input("Ask ANA…", placeholder="e.g. Is it safe to fly now? Best detection settings?",
                                       key="ana_input", label_visibility="collapsed")
        with bc:
            send_clicked = st.button("▶ SEND", key="ana_send", use_container_width=True, type="primary")

        if send_clicked and user_input.strip():
            user_msg = user_input.strip()
            st.session_state["ana_history"].append({"role":"user","content":user_msg})
            with st.spinner("ANA is thinking…"):
                reply = ana_chat(user_msg)
            st.session_state["ana_history"].append({"role":"assistant","content":reply})
            st.rerun()

        ctx_parts = []
        if conn: ctx_parts.append(f"🟢 {bat:.0f}% bat")
        if flying: ctx_parts.append(f"✈️ {tel.get('height',0)}cm")
        if stats["defects_found"] > 0: ctx_parts.append(f"🔍 {stats['defects_found']} defects")
        ctx_parts.append(f"🛡️ {evaluate_safety()}")
        ctx_parts.append(f"🗄️ {st.session_state.get('db_type','MongoDB')}")
        wx_c = st.session_state.get("weather_cache")
        if wx_c: ctx_parts.append(f"🌤️ {wx_c['temp']}°C")
        if ctx_parts:
            st.caption("ANA context: " + " · ".join(ctx_parts))


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<hr style="border-color:#0d4f6e;margin-top:16px">'
    '<div style="text-align:center;color:#1a4a5a;font-family:\'Share Tech Mono\',monospace;'
    'font-size:0.62rem;padding:8px;letter-spacing:1px">'
    'Building Contruction Inspection · Developed by Sandun Wijesinghe · Innovating the future of aerial technology ·Smarter skies, safer decisions.'
    '</div>',
    unsafe_allow_html=True,
)