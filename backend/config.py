"""
config.py — All settings loaded from .env
Import `settings` anywhere in the app.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    # LLM — Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Voice
    deepgram_api_key: str = ""
    azure_tts_key: str = ""
    azure_tts_region: str = ""
    # Backward-compatible aliases if you used Azure naming in portal notes/scripts.
    azure_speech_key: str = ""
    azure_speech_region: str = ""
    azure_tts_voice_en: str = "en-IN-NeerjaNeural"
    azure_tts_voice_hi: str = "hi-IN-SwaraNeural"
    azure_tts_voice_ta: str = "ta-IN-PallaviNeural"

    # Database
    database_url: str = "postgresql+asyncpg://voiceagent:voiceagent@localhost:5432/voiceagent"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    public_base_url: str = "https://example.ngrok.io"

    # App
    session_ttl_seconds: int = 1800
    log_level: str = "INFO"


settings = Settings()


def get_azure_tts_key() -> str:
    return settings.azure_tts_key or settings.azure_speech_key


def get_azure_tts_region() -> str:
    return settings.azure_tts_region or settings.azure_speech_region
