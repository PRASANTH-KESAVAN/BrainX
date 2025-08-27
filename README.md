# WhatsApp–MCP–MedGemma

> Final project repo (code only). **Model files are stored separately on Google Drive** to keep the repo lightweight.

## Quickstart

1. **Clone** the repo and **create a virtual environment**.
2. **Install** Python deps from `requirements.txt`.
3. **Download model files** from the Drive link and **place them in `./models/`** exactly as shown below.
4. **Run** the app / scripts.

---

## 1) Requirements

* **Python**: 3.10+ recommended (3.11 OK)
* **pip**: 23+
* (Optional) **Conda** 23+ if you prefer conda envs
* OS: Windows / macOS / Linux

> If you use GPUs, install the **CUDA-enabled** build of PyTorch that matches your driver.

---

## 2) Setup

### Option A — `venv` (built‑in)

```bash
# clone your fork
git clone https://github.com/<your-org-or-user>/whatsapp-mcp-medgemma.git
cd whatsapp-mcp-medgemma

# create and activate venv
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — Conda

```bash
git clone https://github.com/<your-org-or-user>/whatsapp-mcp-medgemma.git
cd whatsapp-mcp-medgemma

conda create -n wmm python=3.10 -y
conda activate wmm
pip install -r requirements.txt
```
### Option C — Ollama (required for MedGemma summaries)

We use [**Ollama**](https://ollama.ai/) to serve the MedGemma model locally.

1. **Install Ollama**
   - **macOS / Windows**: download from <https://ollama.ai/download>  
   - **Linux**:  
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```

2. **Pull the MedGemma model**
   ```bash
   ollama pull amsaravi/medgemma-4b-it:q6

3. **Run the Ollama server**
   ```bash
   ollama serve

4. **Verify installation**
   ```bash
   ollama list
You should see amsaravi/medgemma-4b-it:q6 listed.

✅ Note: Keep Ollama running in a separate terminal session while using the chat/summary features of this project.

---

## 3) Bring the models (from Google Drive)

**Drive link:** [https://drive.google.com/file/d/1ZfwngBL7RO7HAm1WiXmeTYoHzwyQ2OiX/view?usp=sharing](https://drive.google.com/file/d/1ZfwngBL7RO7HAm1WiXmeTYoHzwyQ2OiX/view?usp=sharing)

1. Download the archive from the link above.
2. Extract it locally.
3. Move **all model files/folders** into the repo at: `./models/`

Your folder should end up like:

```
whatsapp-mcp-medgemma/
├─ models/
│  ├─ <model file (you can only  use one model at a time. Suggection use Resnet x ViT Hybrid model)>/*
│  └─ CLasses.txt
├─ src/
│  ├─ <your two project files from the ZIP>
│  └─ __init__.py
├─ requirements.txt
├─ .env                # (optional) API keys, config
├─ run.py              # (if applicable)
└─ README.md
```

> **Important:** Keep the **same filenames** and **directory layout** that your code expects. If your code looks for `models/medgemma/weights.safetensors`, make sure that exact path exists.

---

## 4) Environment variables / config

Create a `.env` file at the repo root if needed (examples):

```
WHATSAPP_TOKEN=...
WHATSAPP_PHONE_ID=...
WHATSAPP_VERIFY_TOKEN=...
OPENAI_API_KEY=...
MCP_SERVER_URL=...
```

> The project should read from `.env` via `python-dotenv` or your preferred config loader.

---

## 5) How to run

### Example: run a script

```bash
python run.py
```

### Example: launch a web service (FastAPI/Flask)

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### Example: CLI usage

```bash
python -m src.<module> --help
```

> Replace with the real entry points used by your two project files. If they are standalone scripts, document each one briefly below.

---

## 6) About the two project files (from your ZIP)

Describe briefly what each file does and how to run it.

* **`src/<FILE_ONE>.py`** — what it does, expected inputs/outputs, example command.
* **`src/<FILE_TWO>.py`** — what it does, expected inputs/outputs, example command.

---

## 7) Troubleshooting

* **`ModuleNotFoundError`** → make sure your virtual env is **activated** and `pip install -r requirements.txt` completed without errors.
* **Model not found / path errors** → verify the **`./models/`** directory exists and matches expected subpaths.
* **CUDA errors** → install matching CUDA/PyTorch builds or run CPU‑only.

---

## 8) Contributing

PRs and issues welcome. Please run formatters/linters before committing.

---

## 9) License

Choose a license (e.g., MIT) and add `LICENSE` at the repo root.
