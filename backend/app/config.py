import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_ROOT / ".env")
load_dotenv()


def _split_origins(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    APP_NAME: str = "DisasterPulse TN"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Evidence-backed flood intelligence and risk-aware evacuation decision "
        "support for Tamil Nadu. Decision-support only — not an authoritative "
        "emergency dispatch system."
    )

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

    CORS_ORIGINS: list[str] = _split_origins(
        os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5173,"
            "http://127.0.0.1:3000,http://127.0.0.1:5173",
        )
    )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DATA_DIR: Path = BACKEND_ROOT / "data"


settings = Settings()
