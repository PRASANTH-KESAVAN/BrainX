import os
from dotenv import load_dotenv

# Load .env BEFORE importing app
load_dotenv()

from app import app, bootstrap

if __name__ == "__main__":
    bootstrap()
    app.run(port=5000, debug=True)
