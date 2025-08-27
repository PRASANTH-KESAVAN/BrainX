# app/mcp/tools/reports.py
from typing import Dict
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pathlib import Path

DISCLAIMER = "⚠ Educational demo only — not a medical diagnosis."

BASE_DIR = Path(__file__).resolve().parents[2]   # → app/
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _is_no_tumor(label: str) -> bool:
    norm = (label or "").lower().replace(" ","").replace("-","").replace("_","")
    return norm in {"notumor","no","normal","none","negativemri","no-tumor","no_tumor"}

def create_pdf(session_id: str, image_path: str, result: Dict, base_url: str) -> Dict:
    out_path = UPLOAD_DIR / f"report_{session_id}.pdf"
    c = canvas.Canvas(str(out_path), pagesize=A4)
    W, H = A4
    y = H - 60

    c.setFont("Helvetica-Bold", 16); c.drawString(40, y, "Brain MRI — AI Summary"); y -= 25
    c.setFont("Helvetica", 11); c.drawString(40, y, f"Session: {session_id}"); y -= 18

    if _is_no_tumor(result.get("label","")):
        head = "Prediction: No obvious tumor detected"
    else:
        head = f"Prediction: {result['label'].title()}  (Confidence {int(result['confidence']*100)}%)"
    c.drawString(40, y, head); y -= 18

    c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Why:"); y -= 16
    c.setFont("Helvetica", 11)
    for line in (result.get("explanation","")).split("\n"):
        c.drawString(40, y, line); y -= 14

    y -= 8
    c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Summary:"); y -= 16
    c.setFont("Helvetica", 11)
    for line in (result.get("summary","")).split("\n"):
        c.drawString(40, y, line); y -= 14

    y_img = 120
    try:
        img = ImageReader(image_path)
        iw, ih = img.getSize()
        max_w, max_h = W - 80, 300
        scale = min(max_w/iw, max_h/ih)
        c.drawImage(img, 40, y_img, width=iw*scale, height=ih*scale, preserveAspectRatio=True, mask='auto')
    except Exception:
        c.drawString(40, y_img, "[Image embed failed]")

    c.setFont("Helvetica-Oblique", 9); c.drawString(40, 80, DISCLAIMER)
    c.showPage(); c.save()

    pdf_url = f"{base_url}/static/uploads/{out_path.name}" if base_url else ""
    return {"pdf_path": str(out_path), "pdf_url": pdf_url}
