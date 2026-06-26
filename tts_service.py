from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent_voice_profiles import (
    AGENT_PROFILES,
    AgentVoiceProfile,
    agent_aliases,
    agent_profiles_payload,
    env_bool,
    profiles_enabled,
)
from audio_humanizer import (
    AudioHumanizerResult,
    audio_humanizer_enabled,
    humanize_audio,
    join_audio_chunks_with_pauses,
    pause_control_enabled,
    split_text_for_sentence_pauses,
)
from tts_humanizer import (
    DeliveryResult,
    TextHumanizationResult,
    apply_delivery_direction,
    humanize_text,
)


SERVICE_NAME = "readirect-tts"
PROVIDER = "kokoro"
PIPELINE_VERSION = "humanized-v1"
SAMPLE_RATE = 24000
MAX_TEXT_LENGTH = 300
CACHE_DIR = Path(__file__).resolve().parent / "generated_audio"
BREATH_DIR = Path(__file__).resolve().parent / "breaths"

AGENTS = agent_profiles_payload()
ALIASES = agent_aliases()

logger = logging.getLogger(SERVICE_NAME)
logging.basicConfig(level=logging.INFO)

pipeline = None

app = FastAPI(title="ReaDirect TTS", version="1.1.0")


class SynthesizeRequest(BaseModel):
    agent: str
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None
    cache: bool = True
    context: Optional[str] = None
    text_kind: Optional[str] = None
    state: Optional[str] = None
    outcome: Optional[str] = None
    attempt: Optional[int] = None
    protected_terms: list[str] = Field(default_factory=list)
    humanize: Optional[bool] = None
    delivery_control: Optional[bool] = None
    audio_humanizer: Optional[bool] = None
    pause_control: Optional[bool] = None


@dataclass(frozen=True)
class PreparedSynthesis:
    cache_key: str
    output_path: Path
    agent: str
    profile: AgentVoiceProfile
    voice: str
    speed: float
    original_text: str
    humanized_text: str
    synthesis_text: str
    text_result: TextHumanizationResult
    delivery_result: DeliveryResult


def normalize_agent(agent: str) -> str:
    key = ALIASES.get((agent or "").strip(), (agent or "").strip())
    if key not in AGENT_PROFILES:
        raise HTTPException(status_code=422, detail="Unknown agent.")
    return key


def sanitize_text(text: str) -> str:
    cleaned = re.sub(r"<[^>]*>", " ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace("`", "").replace("{", "").replace("}", "")

    if not cleaned:
        raise HTTPException(status_code=422, detail="Text is required.")

    if len(cleaned) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=422, detail=f"Text must be {MAX_TEXT_LENGTH} characters or fewer.")

    return cleaned


def cache_key(agent: str, voice: str, speed: float, text: str, options_key: str = "default") -> str:
    source = f"{PROVIDER}|{PIPELINE_VERSION}|{options_key}|{agent}|{voice}|{speed:.2f}|{text}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def get_pipeline():
    global pipeline
    if pipeline is None:
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code="a")
    return pipeline


def _request_context(request: SynthesizeRequest) -> dict:
    return {
        "context": request.context,
        "text_kind": request.text_kind,
        "state": request.state,
        "outcome": request.outcome,
        "attempt": request.attempt,
    }


def prepare_tts_text(agent: str, text: str, profile: AgentVoiceProfile, request: SynthesizeRequest) -> tuple[str, str, TextHumanizationResult, DeliveryResult]:
    context = _request_context(request)
    text_result = humanize_text(
        agent=agent,
        text=text,
        profile=profile,
        context=context,
        protected_terms=request.protected_terms,
        request_enabled=request.humanize,
    )
    delivery_result = apply_delivery_direction(
        agent=agent,
        text=text_result.text,
        profile=profile,
        context=context,
        request_enabled=request.delivery_control,
    )
    return text_result.text, delivery_result.text, text_result, delivery_result


def _generate_audio_chunk(text: str, voice: str, speed: float) -> list[np.ndarray]:
    chunks = []
    generator = get_pipeline()(text, voice=voice, speed=speed)

    for _, _, audio in generator:
        chunks.append(np.asarray(audio, dtype=np.float32))

    return chunks


def generate_audio(
    text: str,
    voice: str,
    speed: float,
    output_path: Path,
    audio_humanizer_request: Optional[bool] = None,
    pause_control_request: Optional[bool] = None,
) -> AudioHumanizerResult:
    sentence_texts = split_text_for_sentence_pauses(text)
    sentence_audio: list[np.ndarray] = []

    for sentence in sentence_texts:
        sentence_chunks = _generate_audio_chunk(sentence, voice, speed)
        if sentence_chunks:
            sentence_audio.append(np.concatenate(sentence_chunks) if len(sentence_chunks) > 1 else sentence_chunks[0])

    if not sentence_audio:
        raise RuntimeError("No audio was generated.")

    audio, pause_applied = join_audio_chunks_with_pauses(
        sentence_audio,
        sentence_texts,
        SAMPLE_RATE,
        request_enabled=pause_control_request,
    )
    audio, result = humanize_audio(
        audio,
        text,
        SAMPLE_RATE,
        BREATH_DIR,
        request_enabled=audio_humanizer_request,
        pause_already_applied=pause_applied,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, SAMPLE_RATE)
    return result


def audio_path_for(request: SynthesizeRequest) -> PreparedSynthesis:
    agent = normalize_agent(request.agent)
    original_text = sanitize_text(request.text)
    profile = AGENT_PROFILES[agent]
    voice = profile.voice
    speed = profile.speed_for_request(request.speed)
    humanized_text, synthesis_text, text_result, delivery_result = prepare_tts_text(agent, original_text, profile, request)
    options_key = "|".join(
        [
            f"humanize={request.humanize}",
            f"delivery={request.delivery_control}",
            f"audio={request.audio_humanizer}",
            f"pause={request.pause_control}",
        ]
    )
    key = cache_key(agent, voice, speed, synthesis_text, options_key)

    return PreparedSynthesis(
        cache_key=key,
        output_path=CACHE_DIR / f"{key}.wav",
        agent=agent,
        profile=profile,
        voice=voice,
        speed=speed,
        original_text=original_text,
        humanized_text=humanized_text,
        synthesis_text=synthesis_text,
        text_result=text_result,
        delivery_result=delivery_result,
    )


def _debug_logging_enabled() -> bool:
    return env_bool("TTS_DEBUG_LOGGING", False) or env_bool("TTS_TEXT_HUMANIZER_LOGGING", False)


def _empty_audio_result() -> AudioHumanizerResult:
    return AudioHumanizerResult(
        enabled=audio_humanizer_enabled(),
        normalized=False,
        faded=False,
        pause_control_applied=False,
        breath_applied=False,
        duration_seconds=0.0,
    )


def log_tts_request(prepared: PreparedSynthesis, audio_result: AudioHumanizerResult, cache_hit: bool) -> None:
    if not _debug_logging_enabled():
        return

    payload = {
        "agent": prepared.agent,
        "voice": prepared.voice,
        "speed": prepared.speed,
        "original_text": prepared.original_text,
        "humanized_text": prepared.humanized_text,
        "synthesis_text": prepared.synthesis_text,
        "text_humanization_applied": prepared.text_result.applied,
        "text_humanization_reason": prepared.text_result.reason,
        "delivery_control_applied": prepared.delivery_result.applied,
        "safe_chunking_applied": prepared.delivery_result.safe_chunking_applied,
        "audio_post_processing_applied": audio_result.enabled,
        "audio_normalized": audio_result.normalized,
        "audio_faded": audio_result.faded,
        "pause_control_applied": audio_result.pause_control_applied,
        "breath_insertion_applied": audio_result.breath_applied,
        "output_path": str(prepared.output_path),
        "audio_duration_seconds": round(audio_result.duration_seconds, 3),
        "cache_hit": cache_hit,
    }
    logger.info("TTS request prepared: %s", json.dumps(payload, ensure_ascii=True))


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": SERVICE_NAME,
        "provider": PROVIDER,
        "pipeline_version": PIPELINE_VERSION,
        "agent_profiles_enabled": profiles_enabled(),
        "voices": {
            "miss_vivian": AGENT_PROFILES["miss_vivian"].voice,
            "miss_ciel": AGENT_PROFILES["miss_ciel"].voice,
            "miss_estelle": AGENT_PROFILES["miss_estelle"].voice,
        },
        "humanization": {
            "text_humanizer_enabled": env_bool("TTS_TEXT_HUMANIZER_ENABLED", True),
            "delivery_control_enabled": env_bool("TTS_DELIVERY_CONTROL_ENABLED", True),
            "safe_chunking_enabled": env_bool("TTS_SAFE_CHUNKING_ENABLED", True),
            "audio_humanizer_enabled": audio_humanizer_enabled(),
            "pause_control_enabled": pause_control_enabled(),
            "breaths_enabled": env_bool("TTS_BREATHS_ENABLED", False),
        },
    }


@app.get("/voices")
def voices():
    return {"agents": agent_profiles_payload()}


@app.post("/synthesize")
def synthesize(request: SynthesizeRequest):
    prepared = audio_path_for(request)
    audio_result = _empty_audio_result()
    cache_hit = request.cache and prepared.output_path.exists()

    if not cache_hit:
        try:
            generated_result = generate_audio(
                prepared.synthesis_text,
                prepared.voice,
                prepared.speed,
                prepared.output_path,
                request.audio_humanizer,
                request.pause_control,
            )
            audio_result = generated_result or _empty_audio_result()
        except Exception as exc:
            raise HTTPException(status_code=500, detail="TTS generation failed.") from exc
    else:
        if prepared.output_path.exists():
            try:
                info = sf.info(prepared.output_path)
                audio_result = AudioHumanizerResult(
                    enabled=audio_humanizer_enabled(),
                    normalized=False,
                    faded=False,
                    pause_control_applied=False,
                    breath_applied=False,
                    duration_seconds=float(info.duration),
                )
            except Exception:
                audio_result = _empty_audio_result()

    log_tts_request(prepared, audio_result, cache_hit)

    return FileResponse(
        prepared.output_path,
        media_type="audio/wav",
        filename=f"{prepared.cache_key}.wav",
        headers={
            "X-ReaDirect-TTS-Provider": PROVIDER,
            "X-ReaDirect-TTS-Agent": prepared.agent,
            "X-ReaDirect-TTS-Voice": prepared.voice,
            "X-ReaDirect-TTS-Speed": f"{prepared.speed:.2f}",
            "X-ReaDirect-TTS-Humanized": "1" if prepared.text_result.applied else "0",
            "X-ReaDirect-TTS-Cache-Key": prepared.cache_key,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)
