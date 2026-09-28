import os
from pathlib import Path
from google import genai

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass

LLM_MODEL = "models/gemini-2.5-flash-lite"

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        # try Streamlit secrets first (for Streamlit Cloud deployment)
        api_key = None
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

        # fall back to env var (local development)
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found")

        _client = genai.Client(api_key=api_key)
    return _client