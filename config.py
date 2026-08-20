import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in local runs
    def load_dotenv():
        return False


load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
LANGUAGE_ENDPOINT = os.getenv("LANGUAGE_ENDPOINT", "")
LANGUAGE_KEY = os.getenv("LANGUAGE_KEY", "")
