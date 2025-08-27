# run_mcp.py at repo root
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import uvicorn

load_dotenv()

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from mcp.orchestrator import app

if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "7001"))
    reload_flag = os.getenv("MCP_RELOAD", "false").lower() == "true"
    uvicorn.run(app, host=host, port=port, reload=reload_flag)
