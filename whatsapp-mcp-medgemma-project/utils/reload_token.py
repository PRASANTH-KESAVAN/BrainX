import os
from dotenv import load_dotenv

def reload_token() -> str:
    """
    Reloads ACCESS_TOKEN from .env without restarting the app.
    """
    load_dotenv(override=True)  # re-read .env, override old vars
    token = os.getenv("ACCESS_TOKEN")
    if token:
        print("🔑 ACCESS_TOKEN reloaded.")
    else:
        print("⚠ ACCESS_TOKEN missing in .env")
    return token
