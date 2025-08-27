# start_all.ps1 — launch Flask webhook, MCP orchestrator, and ngrok

# 1. Activate your virtual environment (edit path if different)
Write-Host "Activating virtual environment..."
. .\venv\Scripts\Activate.ps1

# 2. Start Flask webhook (WhatsApp) in new window
Write-Host "Starting Flask webhook..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python run.py"

# 3. Start ngrok on port 5000 in new window
Write-Host "Starting ngrok tunnel (http://localhost:5000)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "ngrok http --url=enhanced-suited-mink.ngrok-free.app 5000"

# 4. Start Ollama local server in new window
Write-Host "Starting Ollama server..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "ollama serve"

Write-Host "`nAll services started."
Write-Host "⚠ Reminder: Copy the new ngrok HTTPS URL and paste it into Meta Developer Console webhook."


