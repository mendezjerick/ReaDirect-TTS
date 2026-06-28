from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from agent_voice_profiles import env_bool


SERVICE_ROOT = Path(__file__).resolve().parent


def _resolve_path(value: str | None, default: str) -> Path:
    raw = (value or default).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = SERVICE_ROOT / path
    return path


def default_engine() -> str:
    return (os.getenv("TTS_ENGINE") or "kokoro").strip().lower() or "kokoro"


def expressive_engine_name() -> str:
    return (os.getenv("TTS_EXPRESSIVE_ENGINE") or "index_tts2").strip().lower() or "index_tts2"


def expressive_feature_enabled() -> bool:
    return env_bool("TTS_EXPRESSIVE_ENGINE_ENABLED", False)


def expressive_engine_enabled(request_engine: Optional[str] = None, request_expressive: Optional[bool] = None) -> bool:
    if request_expressive is not None:
        return request_expressive and expressive_feature_enabled()

    engine = (request_engine or default_engine()).strip().lower()
    if engine in {"expressive", "index_tts2", expressive_engine_name()}:
        return expressive_feature_enabled()

    return expressive_feature_enabled() and engine != "kokoro"


def expressive_fallback_to_kokoro() -> bool:
    return env_bool("TTS_EXPRESSIVE_FALLBACK_TO_KOKORO", True)


def cache_enabled(request_cache: bool = True) -> bool:
    if env_bool("TTS_CACHE_BYPASS", False):
        return False
    return request_cache and env_bool("TTS_CACHE_ENABLED", True)


def cache_version() -> str:
    return (os.getenv("TTS_CACHE_VERSION") or "hybrid-v1").strip() or "hybrid-v1"


def auto_prompt_extension_enabled() -> bool:
    return env_bool("TTS_AUTO_PROMPT_EXTENSION_ENABLED", False)


def curated_prompts_enabled() -> bool:
    return env_bool("TTS_CURATED_PROMPTS_ENABLED", True)


def curated_prompt_target_seconds() -> float:
    value = (os.getenv("TTS_CURATED_PROMPT_TARGET_SECONDS") or "7.0").strip()
    try:
        return float(value)
    except ValueError:
        return 7.0


def curated_prompt_min_seconds() -> float:
    value = (os.getenv("TTS_CURATED_PROMPT_MIN_SECONDS") or "6.0").strip()
    try:
        return float(value)
    except ValueError:
        return 6.0


def curated_prompt_max_seconds() -> float:
    value = (os.getenv("TTS_CURATED_PROMPT_MAX_SECONDS") or "9.0").strip()
    try:
        return float(value)
    except ValueError:
        return 9.0


def reference_weighting_enabled() -> bool:
    return env_bool("TTS_REFERENCE_WEIGHTING_ENABLED", True)


def reference_high_priority_min_seconds() -> float:
    value = (os.getenv("TTS_REFERENCE_HIGH_PRIORITY_MIN_SECONDS") or "7.0").strip()
    try:
        return float(value)
    except ValueError:
        return 7.0


def reference_medium_priority_min_seconds() -> float:
    value = (os.getenv("TTS_REFERENCE_MEDIUM_PRIORITY_MIN_SECONDS") or "4.0").strip()
    try:
        return float(value)
    except ValueError:
        return 4.0


def reference_weighting_version() -> str:
    return (os.getenv("TTS_REFERENCE_WEIGHTING_VERSION") or "duration-weight-v1").strip() or "duration-weight-v1"


def cache_root() -> Path:
    return _resolve_path(os.getenv("TTS_CACHE_ROOT"), "storage/tts/cache")


def kokoro_cache_root() -> Path:
    return cache_root() / "kokoro"


def expressive_cache_root() -> Path:
    return cache_root() / "expressive"


def comparison_output_root() -> Path:
    return _resolve_path(os.getenv("TTS_COMPARISON_OUTPUT_ROOT"), "storage/tts/cache/comparisons")


def reference_audio_root() -> Path:
    return _resolve_path(os.getenv("TTS_REFERENCE_AUDIO_ROOT"), "storage/tts/references")


def reference_manifest_path() -> Path:
    return _resolve_path(os.getenv("TTS_REFERENCE_MANIFEST_PATH"), "storage/tts/references/manifest.json")


def index_tts2_adapter_module() -> str:
    return (
        os.getenv("INDEXTTS2_ADAPTER_MODULE")
        or os.getenv("TTS_INDEXTTS2_ADAPTER_MODULE")
        or "index_tts2_adapter"
    ).strip()


def index_tts2_enabled() -> bool:
    return env_bool("INDEXTTS2_ENABLED", expressive_feature_enabled())


def index_tts2_model_dir() -> Path:
    return _resolve_path(os.getenv("INDEXTTS2_MODEL_DIR"), "models/indextts2/checkpoints")


def index_tts2_config_path() -> Path:
    configured = os.getenv("INDEXTTS2_CONFIG_PATH")
    if configured and configured.strip():
        return _resolve_path(configured, "models/indextts2/checkpoints/config.yaml")

    return index_tts2_model_dir() / "config.yaml"


def index_tts2_repo_path() -> str:
    return (os.getenv("INDEXTTS2_REPO_PATH") or "").strip()


def index_tts2_device() -> str:
    return (os.getenv("INDEXTTS2_DEVICE") or "auto").strip().lower() or "auto"


def index_tts2_output_sample_rate() -> int:
    value = (os.getenv("INDEXTTS2_OUTPUT_SAMPLE_RATE") or "24000").strip()
    try:
        return int(value)
    except ValueError:
        return 24000


def index_tts2_emo_alpha() -> float:
    value = (os.getenv("INDEXTTS2_EMO_ALPHA") or "0.75").strip()
    try:
        return max(0.0, min(1.0, float(value)))
    except ValueError:
        return 0.75


def index_tts2_use_random() -> bool:
    return env_bool("INDEXTTS2_USE_RANDOM", False)


def index_tts2_verbose() -> bool:
    return env_bool("INDEXTTS2_VERBOSE", debug_logging_enabled())


def debug_logging_enabled() -> bool:
    return env_bool("TTS_DEBUG_LOGGING", False) or env_bool("TTS_TEXT_HUMANIZER_LOGGING", False)


def two_stage_generation_enabled() -> bool:
    return env_bool("READIRECT_TTS_TWO_STAGE_GENERATION", True)


def stage1_reference_style_enabled() -> bool:
    return env_bool("READIRECT_TTS_STAGE1_REFERENCE_STYLE_ENABLED", True)


def stage2_kokoro_identity_enabled() -> bool:
    return env_bool("READIRECT_TTS_STAGE2_KOKORO_IDENTITY_ENABLED", True)


def active_stage() -> str:
    stage = (os.getenv("READIRECT_TTS_ACTIVE_STAGE") or "reference_style").strip().lower()
    return stage if stage in {"reference_style", "kokoro_identity"} else "reference_style"
