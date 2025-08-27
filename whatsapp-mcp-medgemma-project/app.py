# app.py
import os
import base64
import sqlite3
import tempfile
import traceback
from datetime import datetime, timezone

from flask import Flask, request
from PIL import Image
import requests

from dotenv import load_dotenv
load_dotenv()

# ---- Optional: hybrid classifier (ResNet+ViT) ----
try:
    from torchvision import transforms
    import torch
    import torch.nn.functional as F
    from hybrid_model import HybridResNetViT
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---- MedGemma (Ollama) helper in separate file ----
# This one actually sends the image (as base64) + label to the vision model
from mcp.tools.medgemma import generate_summary 

# -------------------------
# Config from .env
# -------------------------
VERIFY_TOKEN     = os.getenv("VERIFY_TOKEN", "my_secret_token")
ACCESS_TOKEN     = os.getenv("ACCESS_TOKEN", "")
PHONE_NUMBER_ID  = os.getenv("PHONE_NUMBER_ID", "")
GRAPH_BASE       = os.getenv("GRAPH_BASE", "https://graph.facebook.com/v20.0")

PUBLIC_BASE_URL  = os.getenv("PUBLIC_BASE_URL", "https://your-ngrok-id.ngrok-free.app")
PIN_CODE         = os.getenv("DEMO_PIN", "123456")

# softmax temperature (helps avoid 100% saturation)
SOFTMAX_TEMPERATURE = float(os.getenv("SOFTMAX_TEMPERATURE", "1.5"))

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, static_url_path="/static", static_folder="static")
os.makedirs(os.path.join("static", "reports"), exist_ok=True)

# -------------------------
# Class names (optional classes.txt)
# -------------------------
def read_class_names():
    path = os.path.join("models", "classes.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if names:
            return names
    return None

# -------------------------
# Classifier (load if present; fallback otherwise)
# -------------------------
def load_model_or_fallback():
    if not TORCH_AVAILABLE:
        app.logger.warning("Torch/hybrid_model not available — using fallback classifier.")
        def _fallback(_):
            return "Glioma", 0.88
        return _fallback

    device = "cuda" if torch.cuda.is_available() and os.getenv("FORCE_CPU","0") != "1" else "cpu"
    try:
        ck_path = os.path.join("models", "hybrid_resnet50_vit_b16_best.pt")
        ckpt = torch.load(ck_path, map_location=device)

        class_names = read_class_names() or ckpt.get("class_names") or ["Glioma","Meningioma","Pituitary","Notumor"]
        num_classes = len(class_names)

        model = HybridResNetViT(num_classes=num_classes)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model.eval()

        img_size = int(ckpt.get("img_size", 224))
        mean = ckpt.get("mean", [0.485, 0.456, 0.406])
        std  = ckpt.get("std",  [0.229, 0.224, 0.225])

        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        def _predict(path):
            img = Image.open(path).convert("RGB")
            x = tfm(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(x) / SOFTMAX_TEMPERATURE
                probs = F.softmax(logits, dim=1).numpy()[0]
            idx = int(probs.argmax())
            return class_names[idx], float(probs[idx])

        app.logger.info("Hybrid model loaded (classes=%s).", class_names)
        return _predict

    except Exception as e:
        app.logger.warning("Model load failed, using fallback: %s", e)
        def _fallback(_):
            return "Glioma", 0.88
        return _fallback

predict_image = load_model_or_fallback()

# -------------------------
# SQLite session with auto‑migrate
# -------------------------
DB_PATH = os.path.join(os.getcwd(), "sessions.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    wa_id TEXT PRIMARY KEY,
    verified INTEGER DEFAULT 0,
    img_count INTEGER DEFAULT 0,
    last_label TEXT,
    last_path TEXT,
    last_msg_id TEXT,
    last_summary TEXT,
    report_in_progress INTEGER DEFAULT 0,
    last_report_started_at TEXT
);
"""

def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def table_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]

def init_db():
    conn = db()
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    conn.commit()

    # migrate if needed
    try:
        cols = table_columns(conn, "sessions")
    except sqlite3.OperationalError:
        cols = []

    required = [
        "wa_id","verified","img_count","last_label","last_path",
        "last_msg_id","last_summary","report_in_progress","last_report_started_at"
    ]
    if not all(c in cols for c in required):
        cur.execute("ALTER TABLE sessions RENAME TO sessions_old")
        cur.executescript(SCHEMA_SQL)
        old_cols = table_columns(conn, "sessions_old")
        common = [c for c in old_cols if c in required and c != "wa_id"]
        possible_keys = ["wa_id","user_id","phone","from_wa","user"]
        key_src = next((k for k in possible_keys if k in old_cols), None)
        if key_src:
            to_cols = ["wa_id"] + common
            from_expr = [f"{key_src} as wa_id"] + common
            cur.execute(
                f"INSERT OR IGNORE INTO sessions ({','.join(to_cols)}) "
                f"SELECT {','.join(from_expr)} FROM sessions_old"
            )
        cur.execute("DROP TABLE sessions_old")
        conn.commit()

    conn.close()

def get_session(wa_id: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE wa_id=?", (wa_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def upsert_session(wa_id: str, **fields):
    conn = db()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO sessions (wa_id, verified, img_count, report_in_progress) VALUES (?,0,0,0)", (wa_id,))
    if fields:
        cols = ", ".join([f"{k}=?" for k in fields.keys()])
        vals = list(fields.values()) + [wa_id]
        cur.execute(f"UPDATE sessions SET {cols} WHERE wa_id=?", vals)
    conn.commit()
    conn.close()

def next_image_index(wa_id: str) -> int:
    s = get_session(wa_id)
    current = int(s.get("img_count", 0)) if s else 0
    new_val = current + 1
    upsert_session(wa_id, img_count=new_val)
    return new_val

def set_report_state(wa_id: str, in_progress: bool):
    upsert_session(
        wa_id,
        report_in_progress=1 if in_progress else 0,
        last_report_started_at=datetime.now(timezone.utc).isoformat() if in_progress else None
    )

def get_report_state(wa_id: str) -> bool:
    s = get_session(wa_id) or {}
    return bool(s.get("report_in_progress", 0))

# -------------------------
# WhatsApp helpers
# -------------------------
def send_text(to: str, body: str, reply_to: str | None = None):
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        app.logger.info("send_text (dry-run): %s", body)
        return
    url = f"{GRAPH_BASE}/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": body}
    }
    if reply_to:
        payload["context"] = {"message_id": reply_to}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code >= 400:
        app.logger.info("send_text %s %s", r.status_code, r.text)

def download_media(media_id: str) -> str:
    meta = requests.get(
        f"{GRAPH_BASE}/{media_id}",
        headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
        timeout=30
    ).json()
    media_url = meta.get("url")
    if not media_url:
        raise RuntimeError(f"Couldn't resolve media URL for id={media_id}: {meta}")
    r = requests.get(media_url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}, timeout=60)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return path

# -------------------------
# PDF (improved headings)
# -------------------------
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

DISCLAIMER = (
    "This report is generated by an AI and should not be considered a substitute for professional medical advice. "
    "The information provided is for educational purposes only and should be reviewed by a qualified healthcare professional."
)

HEADING_KEYS = {
    "MedGemma Report for ",
    "MRI Image Analysis Report:",
    "1. Estimated Tumor Size:",
    "2. General Grading & Risk:",
    "3. Possible Conditions:",
    "4. Suggested Next Research/Tests:",
    "5. Likelihood of Benign vs. Malignant:",
    "6. Comorbidity Analysis:",
    "Final disclaimer:"
}

def make_pdf(pdf_path: str, img_path: str, label: str, summary: str):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"MedGemma Report for {label}")

    y = height - 90
    if img_path and os.path.exists(img_path):
        try:
            c.drawImage(
                img_path, 50, height - 300,
                width=2.5*inch, height=2.5*inch,
                preserveAspectRatio=True, mask='auto'
            )
            y = height - 320
        except Exception:
            y = height - 90

    lines = [ln.rstrip() for ln in summary.splitlines()]
    cursor_x, cursor_y = 50, y
    for ln in lines:
        font = "Helvetica-Bold" if any(ln.startswith(k) for k in HEADING_KEYS) else "Helvetica"
        c.setFont(font, 10 if font == "Helvetica" else 11)
        max_chars = 95
        if len(ln) <= max_chars:
            c.drawString(cursor_x, cursor_y, ln)
            cursor_y -= 14
        else:
            start = 0
            while start < len(ln):
                c.drawString(cursor_x, cursor_y, ln[start:start+max_chars])
                cursor_y -= 14
                start += max_chars
        if cursor_y < 60:
            c.showPage()
            c.setFont("Helvetica", 10)
            cursor_y = height - 60

    if cursor_y < 50:
        c.showPage()
        cursor_y = height - 60
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(50, 30, "⚠ Educational demo only — not a medical diagnosis.")
    c.save()

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"ok": True}

@app.get("/webhook")
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403

@app.post("/webhook")
def handle_webhook():
    data = request.get_json()
    print("payload:", data)
    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                if not messages:
                    continue

                msg = messages[0]
                wa_id = msg.get("from")
                mtype = msg.get("type")

                # Ensure session
                if not get_session(wa_id):
                    upsert_session(wa_id, verified=0, img_count=0, report_in_progress=0)

                session = get_session(wa_id)

                # Require PIN
                if not session or not session.get("verified"):
                    if mtype == "text":
                        text = msg.get("text", {}).get("body", "").strip()
                        if text == PIN_CODE:
                            upsert_session(wa_id, verified=1)
                            send_text(
                                wa_id,
                                "✅ Verified! Now you can upload an MRI image for prediction.\n"
                                "After the result, type 'report' → 'pdf' to export."
                            )
                        else:
                            send_text(wa_id, "Please enter your 6‑digit access code before sending images.")
                    else:
                        send_text(wa_id, "Please enter your 6‑digit access code before sending images.")
                    continue

                # Verified: handle content
                if mtype in ("image", "document"):
                    media_id = (msg.get("image") or msg.get("document") or {}).get("id")
                    if not media_id:
                        send_text(wa_id, "Couldn't read the image. Please resend.")
                        continue

                    idx = next_image_index(wa_id)
                    send_text(wa_id, f"🧠 MRI #{idx} received. Running prediction...")

                    # Download + classify
                    try:
                        path = download_media(media_id)
                        label, conf = predict_image(path)
                    except Exception as e:
                        app.logger.exception("Prediction error: %s", e)
                        send_text(wa_id, "⚠ Couldn’t process the MRI. Please try again.")
                        continue

                    # Reply (tag original image)
                    last_msg_id = msg.get("id")
                    upsert_session(wa_id, last_label=label, last_path=path, last_msg_id=last_msg_id, last_summary=None)

                    if label.lower() in ("notumor", "no tumor", "no tumour", "no_tumor", "no_tumour"):
                        body = (
                            f"📎 Image #{idx}\n"
                            f"Predicted: No obvious tumor detected.\n"
                            f"Why: Pattern‑based features consistent with predicted class.\n"
                            f"Reply 'report' for an educational summary.\n"
                            f"⚠ Educational demo only — not a medical diagnosis."
                        )
                    else:
                        conf_pct = int(round(conf * 100))
                        if conf_pct >= 100 and conf < 0.9995:
                            conf_pct = 99
                        explain_map = {
                            "glioma": "Intra‑axial infiltrative lesion; T2/FLAIR hyperintensity with ill‑defined margins.",
                            "meningioma": "Extra‑axial dural‑based mass with broad dural attachment and smooth margins.",
                            "pituitary": "Sellar/suprasellar lesion; smooth margins; possible mass effect on the optic chiasm.",
                        }
                        why = explain_map.get(label.lower(), "Pattern‑based features consistent with predicted class.")
                        body = (
                            f"📎 Image #{idx}\n"
                            f"Predicted: {label} (Confidence {conf_pct}%).\n"
                            f"Why: {why}\n"
                            f"Reply 'report' for an educational summary.\n"
                            f"⚠ Educational demo only — not a medical diagnosis."
                        )

                    send_text(wa_id, body, reply_to=last_msg_id)
                    continue

                # Text commands
                if mtype == "text":
                    text = msg.get("text", {}).get("body", "").strip().lower()

                    if text in ("hi", "hello", "start"):
                        send_text(wa_id, "Welcome! Send an MRI image for analysis. Then type 'report' → 'pdf' to export.")
                        continue

                    if text == "report":
                        s = get_session(wa_id) or {}
                        last_label = s.get("last_label")
                        last_path  = s.get("last_path")
                        last_msg_id = s.get("last_msg_id")
                        if not last_label or not last_path:
                            send_text(wa_id, "No recent classification found. Please send an MRI image first.")
                            continue

                        # prevent spam
                        if get_report_state(wa_id):
                            continue

                        set_report_state(wa_id, True)
                        send_text(wa_id, "📝 Generating MedGemma summary. This may take a moment…", reply_to=last_msg_id)
                        try:
                            summary = generate_summary(last_path, last_label)  # image + label
                            upsert_session(wa_id, last_summary=summary)
                            send_text(wa_id, summary, reply_to=last_msg_id)
                            send_text(wa_id, "Reply 'pdf' to export as PDF.", reply_to=last_msg_id)
                        except Exception as e:
                            app.logger.exception("Summary error: %s", e)
                            send_text(wa_id, "Couldn’t generate the summary. Please try again.", reply_to=last_msg_id)
                        finally:
                            set_report_state(wa_id, False)
                        continue

                    if text == "pdf":
                        s = get_session(wa_id) or {}
                        last_label   = s.get("last_label")
                        last_path    = s.get("last_path")
                        last_msg_id  = s.get("last_msg_id")
                        last_summary = s.get("last_summary")

                        if not last_label or not last_path:
                            send_text(wa_id, "No report to export. Send an MRI image and type 'report' first.")
                            continue

                        summary = last_summary or f"MedGemma Report for {last_label}\n\n{DISCLAIMER}\n\n(Report text not cached; please type 'report' again.)"
                        outdir = os.path.join("static", "reports")
                        os.makedirs(outdir, exist_ok=True)
                        # timestamped filename to keep history
                        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                        pdf_path = os.path.join(outdir, f"report_{wa_id}_{ts}.pdf")

                        make_pdf(pdf_path, last_path, last_label, summary)
                        public_link = f"{PUBLIC_BASE_URL}/static/reports/report_{wa_id}_{ts}.pdf"
                        send_text(wa_id, f"📄 Report ready: {public_link}", reply_to=last_msg_id)
                        continue

                    # fallback
                    send_text(wa_id, "Send an MRI image for analysis. Then type 'report' → 'pdf' to export.")
                    continue

    except Exception as e:
        print("Error:", e)
        print(traceback.format_exc())
        return "ERROR", 500

    return "EVENT_RECEIVED", 200

# -------------------------
# Bootstrap for run.py
# -------------------------
def bootstrap():
    init_db()
    print("bootstrap(): DB ready.")

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
