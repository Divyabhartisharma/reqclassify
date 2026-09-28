import os
import time
import random
from google import genai

# --- config ---
#API_KEY = os.getenv("GEMINI_API_KEY", " ")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

# gemini-2.5-flash-lite = most stable, highest RPD (1000/day), good for classification
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# --- client ---
client = genai.Client(api_key=API_KEY)


def call_gemini(prompt: str, temperature: float = 0.0, max_retries: int = 6) -> str:
    """
    Call Gemini with TRUE exponential backoff + jitter:
      - 503 UNAVAILABLE       → server overload  : 10s → 20s → 40s → 80s ...
      - 429 RESOURCE_EXHAUSTED → quota hit        : 15s → 30s → 60s → 120s ...
    Jitter (+0-5s random) added to avoid thundering herd.
    All other exceptions are re-raised immediately.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={"temperature": temperature},
            )
            return (resp.text or "").strip()

        except Exception as e:
            err = str(e)

            if "503" in err or "UNAVAILABLE" in err:
                # Exponential backoff: 10, 20, 40, 80, 160, 320 seconds
                wait = (10 * (2 ** (attempt - 1))) + random.uniform(0, 5)
                print(f"[Gemini] 503 overload  | attempt {attempt}/{max_retries} | waiting {wait:.0f}s ...")
                time.sleep(wait)

            elif "429" in err or "RESOURCE_EXHAUSTED" in err:
                # Exponential backoff: 15, 30, 60, 120, 240, 480 seconds
                wait = (15 * (2 ** (attempt - 1))) + random.uniform(0, 5)
                print(f"[Gemini] 429 quota hit | attempt {attempt}/{max_retries} | waiting {wait:.0f}s ...")
                time.sleep(wait)

            else:
                print(f"[Gemini] unexpected error: {e}")
                raise

    print(f"[Gemini] all {max_retries} retries exhausted — returning ERROR")
    return "ERROR"