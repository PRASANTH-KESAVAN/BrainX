# app/mcp/orchestrator.py
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict
import time, uuid

# package‑relative imports (this fixes "No module named 'tools'")
from .tools.classify import classify_image
from .tools.medgemma import generate_summary
from .tools.reports import create_pdf

class Event(BaseModel):
    user_id: str
    session_id: str
    kind: Literal["text","image"]
    text: Optional[str] = None
    image_path: Optional[str] = None
    reply_to: Optional[str] = None
    meta: Dict = Field(default_factory=dict)

class Action(BaseModel):
    type: Literal["send_text"]
    body: str
    reply_to: Optional[str] = None

class ActResponse(BaseModel):
    actions: List[Action]
    session: Dict
    trace_id: str

app = FastAPI(title="MCP Orchestrator")

SESS: Dict[str, Dict] = {}
DISCLAIMER = "⚠ Educational demo only — not a medical diagnosis."

def ensure_session(user_id: str, session_id: str):
    if user_id not in SESS:
        SESS[user_id] = {"session_id": session_id, "verified": False, "last_result": None}

def is_no_tumor(label: str) -> bool:
    if not label: return False
    norm = label.lower().replace(" ", "").replace("-", "").replace("_", "")
    return norm in {"notumor","no","normal","none","negativemri","no-tumor","no_tumor"}

@app.post("/act", response_model=ActResponse)
def act(event: Event):
    ensure_session(event.user_id, event.session_id)
    st = SESS[event.user_id]
    actions: List[Action] = []
    trace_id = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"

    if event.kind == "text":
        t = (event.text or "").strip().lower()
        if t.isdigit() and len(t) == 6:
            st["verified"] = True
            actions.append(Action(type="send_text", body="✅ Verified! You can now send an MRI image.", reply_to=event.reply_to))
        elif t == "report":
            if not st.get("last_result"):
                actions.append(Action(type="send_text", body="No recent classification found. Send an MRI first.", reply_to=event.reply_to))
            else:
                summary = generate_summary(st["last_result"], event.meta)
                st["last_result"]["summary"] = summary
                actions.append(Action(type="send_text",
                    body=f"📝 Report:\n\n{summary}\n\nReply 'pdf' to download as PDF.\n{DISCLAIMER}",
                    reply_to=event.reply_to))
        elif t == "pdf":
            if not st.get("last_result") or "summary" not in st["last_result"]:
                actions.append(Action(type="send_text", body="No summary found. Type 'report' first.", reply_to=event.reply_to))
            else:
                pdf = create_pdf(st["session_id"], st["last_result"]["image_path"], st["last_result"], event.meta.get("base_url",""))
                if pdf.get("pdf_url"):
                    actions.append(Action(type="send_text", body=f"📄 PDF ready: {pdf['pdf_url']}", reply_to=event.reply_to))
                else:
                    actions.append(Action(type="send_text", body="📄 PDF generated (set BASE_PUBLIC_URL to share).", reply_to=event.reply_to))
        else:
            actions.append(Action(type="send_text",
                body="Send an MRI image for analysis. Then type 'report' → 'pdf' to export.",
                reply_to=event.reply_to))

    elif event.kind == "image":
        if not st.get("verified"):
            actions.append(Action(type="send_text", body="🔒 Please enter your 6‑digit access code before sending images.", reply_to=event.reply_to))
        else:
            actions.append(Action(type="send_text", body="🧠 MRI received. Running prediction…", reply_to=event.reply_to))
            res = classify_image(event.image_path)
            st["last_result"] = {**res, "image_path": event.image_path}

            if is_no_tumor(res["label"]):
                pred_line = "Predicted: No obvious tumor detected."
            else:
                conf_pct = int(round(res.get("confidence",0)*100))
                pred_line = f"Predicted: {res['label'].title()} (Confidence {conf_pct}%)."
            explanation = res.get("explanation","Pattern-based features consistent with predicted class.")
            body = f"{pred_line}\nWhy: {explanation}\nReply 'report' for a PDF-ready summary.\n{DISCLAIMER}"
            actions.append(Action(type="send_text", body=body, reply_to=event.reply_to))

    return ActResponse(actions=actions, session=st, trace_id=trace_id)
