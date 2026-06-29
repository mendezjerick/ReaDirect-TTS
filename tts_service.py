from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

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
from curated_agent_lines import CuratedPromptResult, resolve_curated_prompt
from tts_humanizer import (
    DeliveryResult,
    TextHumanizationResult,
    apply_delivery_direction,
    humanize_text,
    looks_protected,
    text_humanizer_enabled,
)
import tts_config
from expressive_references import ReferenceManifest, ReferenceSelection, reference_file_digest
from kokoro_reference import ensure_kokoro_timbre_reference, kokoro_reference_path
from prosody_intents import ProsodyIntentResult, classify_prosody_intent
from tts_engines import EngineRequest, EngineResult, IndexTTS2ExpressiveEngine, KokoroEngine


SERVICE_NAME = "readirect-tts"
PROVIDER = "kokoro"
PIPELINE_VERSION = tts_config.cache_version()
SAMPLE_RATE = 24000
MAX_TEXT_LENGTH = 300
STORAGE_DIR = Path(__file__).resolve().parent / "storage" / "tts"
CACHE_DIR = tts_config.kokoro_cache_root()
EXPRESSIVE_CACHE_DIR = tts_config.expressive_cache_root()
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
    intent: Optional[str] = None
    line_key: Optional[str] = None
    engine: Optional[str] = None
    expressive: Optional[bool] = None
    voice: Optional[str] = None
    speed: Optional[float] = None
    cache: bool = True
    context: Optional[str] = None
    text_kind: Optional[str] = None
    state: Optional[str] = None
    outcome: Optional[str] = None
    attempt: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    protected_terms: list[str] = Field(default_factory=list)
    humanize: Optional[bool] = None
    delivery_control: Optional[bool] = None
    audio_humanizer: Optional[bool] = None
    pause_control: Optional[bool] = None


class VoiceLineBatchItem(BaseModel):
    id: Optional[int] = None
    line_key: str
    agent: str
    intent: str
    text: str
    synthesis_text: Optional[str] = None
    voice_id: Optional[str] = None
    reference_audio_path: Optional[str] = None
    is_static: bool = True
    is_defense_demo: bool = False


class VoiceLineBatchRequest(BaseModel):
    items: list[VoiceLineBatchItem]
    mode: str = "pregenerate_two_stage"
    engine: str = "index_tts2"
    fallback: bool = True
    force: bool = False
    generate_stage2: bool = True
    active_stage: str = "reference_style"
    output_root: str
    public_relative_root: str = "tts/generated_voice_lines"


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
    curated_result: CuratedPromptResult
    requested_engine: str
    engine: str
    intent_result: ProsodyIntentResult
    reference_selection: ReferenceSelection
    reference_digest: str
    emotion_prompt: str
    speaker_reference_path: Optional[Path]
    fallback_reason: Optional[str] = None


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


def cache_key(
    agent: str,
    voice: str,
    speed: float,
    text: str,
    options_key: str = "default",
    engine: str = PROVIDER,
    original_text: Optional[str] = None,
    humanized_text: Optional[str] = None,
    intent: str = "none",
    reference_digest: str = "none",
    speaker_reference_digest: str = "none",
    emotion_prompt: str = "none",
    reference_duration: str = "none",
    reference_weighting_version: str = "none",
) -> str:
    source = "|".join(
        [
            engine,
            PIPELINE_VERSION,
            options_key,
            agent,
            voice,
            f"{speed:.2f}",
            original_text or text,
            humanized_text or text,
            text,
            intent,
            reference_digest,
            speaker_reference_digest,
            emotion_prompt,
            reference_duration,
            reference_weighting_version,
        ]
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def get_pipeline():
    global pipeline
    if pipeline is None:
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code="a")
    return pipeline


def warmup_on_startup_enabled() -> bool:
    return env_bool("TTS_WARMUP_ON_STARTUP", False)


@app.on_event("startup")
def warmup_pipeline_on_startup() -> None:
    if not warmup_on_startup_enabled():
        return

    logger.info("TTS startup warmup enabled; loading Kokoro pipeline.")
    get_pipeline()


def _request_context(request: SynthesizeRequest) -> dict:
    return {
        "context": request.context,
        "text_kind": request.text_kind,
        "state": request.state,
        "outcome": request.outcome,
        "attempt": request.attempt,
        "intent": request.intent,
        "line_key": request.line_key,
        "metadata": request.metadata,
    }


def _load_reference_manifest() -> ReferenceManifest:
    return ReferenceManifest.load(
        tts_config.reference_manifest_path(),
        tts_config.reference_audio_root(),
        logger if _debug_logging_enabled() else None,
    )


def _kokoro_engine() -> KokoroEngine:
    return KokoroEngine(generate_audio)


def _index_tts2_engine() -> IndexTTS2ExpressiveEngine:
    return IndexTTS2ExpressiveEngine(
        kokoro_fallback=_kokoro_engine(),
        fallback_to_kokoro=tts_config.expressive_fallback_to_kokoro(),
        adapter_module_name=tts_config.index_tts2_adapter_module(),
        normalized_reference_root=tts_config.cache_root(),
    )


def _requested_engine(request: SynthesizeRequest) -> str:
    if tts_config.expressive_engine_enabled(request.engine, request.expressive):
        return tts_config.expressive_engine_name()

    return PROVIDER


def _resolve_engine(
    requested_engine: str,
    reference_selection: ReferenceSelection,
) -> tuple[str, Optional[str]]:
    if requested_engine != tts_config.expressive_engine_name():
        return PROVIDER, None

    available, reason = _index_tts2_engine().availability()
    if not available:
        return PROVIDER, reason

    if reference_selection.path is None:
        return PROVIDER, reference_selection.fallback_reason or "style_reference_missing"

    return requested_engine, reference_selection.fallback_reason


def prepare_tts_text(
    agent: str,
    text: str,
    profile: AgentVoiceProfile,
    request: SynthesizeRequest,
) -> tuple[str, str, TextHumanizationResult, DeliveryResult, CuratedPromptResult]:
    context = _request_context(request)
    protected = looks_protected(text, context, request.protected_terms)
    explicit_intent = request.intent or str(request.metadata.get("prosody_intent") or request.metadata.get("intent") or "").strip() or None
    explicit_line_key = request.line_key or str(request.metadata.get("line_key") or "").strip() or None
    if request.humanize is False:
        curated_result = CuratedPromptResult(text, text, False, "request_disabled")
    else:
        curated_result = resolve_curated_prompt(
            agent=agent,
            text=text,
            intent=explicit_intent,
            line_key=explicit_line_key,
            protected=protected,
        )
    if curated_result.applied or curated_result.line is not None:
        text_result = TextHumanizationResult(
            original_text=text,
            text=curated_result.text,
            applied=curated_result.applied,
            protected=False,
            reason=f"curated:{curated_result.reason}",
        )
    else:
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
    return text_result.text, delivery_result.text, text_result, delivery_result, curated_result


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
    humanized_text, synthesis_text, text_result, delivery_result, curated_result = prepare_tts_text(agent, original_text, profile, request)
    context = _request_context(request)
    intent_result = classify_prosody_intent(agent, synthesis_text, context)
    reference_selection = _load_reference_manifest().select(
        intent_result.agent_key,
        intent_result.intent,
        f"{agent}|{original_text}|{humanized_text}|{synthesis_text}",
    )
    reference_digest = reference_file_digest(reference_selection.path)
    requested_engine = _requested_engine(request)
    engine, fallback_reason = _resolve_engine(requested_engine, reference_selection)
    speaker_reference_path = kokoro_reference_path(agent, profile, CACHE_DIR) if requested_engine == tts_config.expressive_engine_name() else None
    speaker_reference_digest = reference_file_digest(speaker_reference_path) if speaker_reference_path and speaker_reference_path.exists() else str(speaker_reference_path or "none")
    options_key = "|".join(
        [
            f"humanize={request.humanize}",
            f"delivery={request.delivery_control}",
            f"audio={request.audio_humanizer}",
            f"pause={request.pause_control}",
            f"requested_engine={requested_engine}",
            f"engine={engine}",
            f"fallback={fallback_reason or 'none'}",
            f"curated={curated_result.line.line_key if curated_result.line else 'none'}",
        ]
    )
    key = cache_key(
        agent,
        voice,
        speed,
        synthesis_text,
        options_key,
        engine=engine,
        original_text=original_text,
        humanized_text=humanized_text,
        intent=intent_result.intent,
        reference_digest=reference_digest,
        speaker_reference_digest=speaker_reference_digest,
        emotion_prompt=intent_result.emotion_prompt,
        reference_duration=f"{reference_selection.duration_seconds:.3f}" if reference_selection.duration_seconds is not None else "none",
        reference_weighting_version=reference_selection.weighting_version,
    )
    cache_dir = EXPRESSIVE_CACHE_DIR if engine == tts_config.expressive_engine_name() else CACHE_DIR

    return PreparedSynthesis(
        cache_key=key,
        output_path=cache_dir / f"{key}.wav",
        agent=agent,
        profile=profile,
        voice=voice,
        speed=speed,
        original_text=original_text,
        humanized_text=humanized_text,
        synthesis_text=synthesis_text,
        text_result=text_result,
        delivery_result=delivery_result,
        curated_result=curated_result,
        requested_engine=requested_engine,
        engine=engine,
        intent_result=intent_result,
        reference_selection=reference_selection,
        reference_digest=reference_digest,
        emotion_prompt=intent_result.emotion_prompt,
        speaker_reference_path=speaker_reference_path,
        fallback_reason=fallback_reason,
    )


def _debug_logging_enabled() -> bool:
    return tts_config.debug_logging_enabled()


def _empty_audio_result() -> AudioHumanizerResult:
    return AudioHumanizerResult(
        enabled=audio_humanizer_enabled(),
        normalized=False,
        faded=False,
        pause_control_applied=False,
        breath_applied=False,
        duration_seconds=0.0,
    )


def _engine_request(prepared: PreparedSynthesis, request: SynthesizeRequest) -> EngineRequest:
    return EngineRequest(
        text=prepared.synthesis_text,
        agent=prepared.agent,
        profile=prepared.profile,
        voice=prepared.voice,
        speed=prepared.speed,
        output_path=prepared.output_path,
        intent=prepared.intent_result.intent,
        emotion_prompt=prepared.emotion_prompt,
        style_reference_path=prepared.reference_selection.path,
        speaker_reference_path=prepared.speaker_reference_path,
        audio_humanizer_request=request.audio_humanizer,
        pause_control_request=request.pause_control,
    )


def generate_prepared_audio(prepared: PreparedSynthesis, request: SynthesizeRequest) -> EngineResult:
    engine_request = _engine_request(prepared, request)

    if prepared.requested_engine == tts_config.expressive_engine_name() and prepared.speaker_reference_path is not None:
        ensure_kokoro_timbre_reference(
            prepared.agent,
            prepared.profile,
            CACHE_DIR,
            generate_audio,
        )

    if prepared.engine == tts_config.expressive_engine_name():
        return _index_tts2_engine().generate(engine_request)

    result = _kokoro_engine().generate(engine_request)
    if prepared.requested_engine != PROVIDER or prepared.fallback_reason:
        result = replace(
            result,
            engine_requested=prepared.requested_engine,
            fallback_reason=prepared.fallback_reason,
            style_reference_path=prepared.reference_selection.path,
        )
    return result


def _cached_engine_result(prepared: PreparedSynthesis, audio_result: AudioHumanizerResult) -> EngineResult:
    return EngineResult(
        output_path=prepared.output_path,
        engine_requested=prepared.requested_engine,
        engine_used=prepared.engine,
        audio_result=audio_result,
        expressive_used=prepared.engine == tts_config.expressive_engine_name(),
        speaker_reference_used=prepared.speaker_reference_path is not None,
        style_reference_path=prepared.reference_selection.path,
        fallback_reason=prepared.fallback_reason,
    )


def log_tts_request(
    prepared: PreparedSynthesis,
    engine_result: EngineResult,
    cache_hit: bool,
    duration_ms: Optional[int] = None,
) -> None:
    if not _debug_logging_enabled():
        return

    payload = {
        "engine_requested": engine_result.engine_requested,
        "engine_used": engine_result.engine_used,
        "agent": prepared.agent,
        "voice": prepared.voice,
        "speed": prepared.speed,
        "original_text": prepared.original_text,
        "humanized_text": prepared.humanized_text,
        "synthesis_text": prepared.synthesis_text,
        "detected_intent": prepared.intent_result.intent,
        "intent_reason": prepared.intent_result.reason,
        "emotion_prompt": prepared.emotion_prompt,
        "reference_file": str(prepared.reference_selection.path) if prepared.reference_selection.path else None,
        "reference_intent": prepared.reference_selection.selected_intent,
        "reference_duration_seconds": prepared.reference_selection.duration_seconds,
        "reference_priority": prepared.reference_selection.priority,
        "reference_weight": prepared.reference_selection.weight,
        "reference_weighting_version": prepared.reference_selection.weighting_version,
        "reference_fallback_reason": prepared.reference_selection.fallback_reason,
        "kokoro_timbre_reference": str(prepared.speaker_reference_path) if prepared.speaker_reference_path else None,
        "kokoro_timbre_reference_used": engine_result.speaker_reference_used,
        "expressive_engine_used": engine_result.expressive_used,
        "fallback_reason": engine_result.fallback_reason,
        "final_mode": engine_result.metadata.get("mode") if engine_result.metadata else ("fallback_kokoro" if engine_result.fallback_reason else engine_result.engine_used),
        "text_humanization_applied": prepared.text_result.applied,
        "text_humanization_reason": prepared.text_result.reason,
        "curated_prompt_applied": prepared.curated_result.applied,
        "curated_line_key": prepared.curated_result.line.line_key if prepared.curated_result.line else None,
        "delivery_control_applied": prepared.delivery_result.applied,
        "safe_chunking_applied": prepared.delivery_result.safe_chunking_applied,
        "audio_post_processing_applied": engine_result.audio_result.enabled,
        "audio_normalized": engine_result.audio_result.normalized,
        "audio_faded": engine_result.audio_result.faded,
        "pause_control_applied": engine_result.audio_result.pause_control_applied,
        "breath_insertion_applied": engine_result.audio_result.breath_applied,
        "output_path": str(prepared.output_path),
        "audio_duration_seconds": round(engine_result.audio_result.duration_seconds, 3),
        "cache_hit": cache_hit,
        "generation_duration_ms": duration_ms,
    }
    logger.info("TTS request prepared: %s", json.dumps(payload, ensure_ascii=True))


def _public_relative_path(public_root: str, *parts: str) -> str:
    clean_root = (public_root or "tts/generated_voice_lines").replace("\\", "/").strip("/")
    clean_parts = [re.sub(r"[^A-Za-z0-9_.-]+", "_", part).strip("_").lower() for part in parts if part]
    return "/".join([clean_root, *clean_parts])


def _voice_line_module_part(line_key: str) -> Optional[str]:
    match = re.search(r"\.module_([123])\.", line_key or "")
    if match:
        return f"module_{match.group(1)}"

    return None


def _voice_line_intent_parts(intent: str, line_key: str = "") -> tuple[str, ...]:
    if intent == "module_echo_correct":
        module_part = _voice_line_module_part(line_key)
        if module_part:
            return ("module_echo", "correct", module_part)

        return ("module_echo", "correct")

    return (intent,)


def _voice_line_filename(line_key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", line_key).strip("_").lower() or "voice_line"
    digest = hashlib.sha256(line_key.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}.wav"


def _reference_priority_and_weight(duration_seconds: Optional[float]) -> tuple[str, int]:
    if duration_seconds is None:
        return "unknown", 1
    if duration_seconds >= tts_config.reference_high_priority_min_seconds():
        return "high", 5
    if duration_seconds >= tts_config.reference_medium_priority_min_seconds():
        return "medium", 2
    return "low", 1


def _reference_duration(path: Path) -> Optional[float]:
    try:
        return float(sf.info(path).duration)
    except Exception:
        return None


def _override_reference_selection(
    item: VoiceLineBatchItem,
    prepared: PreparedSynthesis,
) -> Optional[ReferenceSelection]:
    configured = (item.reference_audio_path or "").strip()
    if not configured:
        return None

    relative_path = configured.replace("\\", "/").strip("/")
    path = Path(configured)
    if not path.is_absolute():
        path = tts_config.reference_audio_root() / relative_path

    if not path.exists() or not path.is_file():
        logger.warning("Voice line reference override missing for %s: %s", item.line_key, path)
        return None

    duration = _reference_duration(path)
    priority, weight = _reference_priority_and_weight(duration)
    return ReferenceSelection(
        prepared.intent_result.intent,
        prepared.intent_result.intent,
        prepared.profile.agent_key,
        path,
        relative_path,
        "reference_override",
        True,
        duration_seconds=duration,
        priority=priority,
        weight=weight,
        weighting_version="override-v1",
    )


def _file_checksum(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_duration(path: Optional[Path]) -> Optional[float]:
    if path is None or not path.exists():
        return None
    try:
        return float(sf.info(path).duration)
    except Exception:
        return None


def _stage_status(result: EngineResult) -> str:
    return "generated" if result.expressive_used else "fallback_generated"


def _stage_payload(
    stage: str,
    output_path: Path,
    public_path: str,
    result: Optional[EngineResult] = None,
    status: Optional[str] = None,
    error: Optional[str] = None,
    style_source_path: Optional[Path] = None,
    speaker_reference_path: Optional[Path] = None,
) -> dict[str, Any]:
    duration = result.audio_result.duration_seconds if result and result.audio_result else _audio_duration(output_path)
    return {
        "stage": stage,
        "status": status or (_stage_status(result) if result else ("generated" if output_path.exists() else "failed")),
        "engine_used": result.engine_used if result else "cached",
        "engine_requested": result.engine_requested if result else tts_config.expressive_engine_name(),
        "audio_path": str(output_path),
        "public_audio_path": public_path if output_path.exists() else None,
        "duration_seconds": duration,
        "fallback_reason": result.fallback_reason if result else None,
        "mode": (result.metadata or {}).get("mode") if result else "cached",
        "error": error,
        "style_source_path": str(style_source_path) if style_source_path else None,
        "speaker_reference_path": str(speaker_reference_path) if speaker_reference_path else None,
    }


def _generate_voice_line_stage(
    engine_request: EngineRequest,
    public_path: str,
    stage: str,
    force: bool,
    style_source_path: Optional[Path],
    speaker_reference_path: Optional[Path],
) -> dict[str, Any]:
    if engine_request.output_path.exists() and not force:
        return _stage_payload(
            stage,
            engine_request.output_path,
            public_path,
            status="generated",
            style_source_path=style_source_path,
            speaker_reference_path=speaker_reference_path,
        )

    try:
        result = _index_tts2_engine().generate(engine_request)
        return _stage_payload(
            stage,
            engine_request.output_path,
            public_path,
            result=result,
            style_source_path=style_source_path,
            speaker_reference_path=speaker_reference_path,
        )
    except Exception as exc:
        return _stage_payload(
            stage,
            engine_request.output_path,
            public_path,
            status="failed",
            error=f"{exc.__class__.__name__}:{exc}",
            style_source_path=style_source_path,
            speaker_reference_path=speaker_reference_path,
        )


def _active_batch_path(active_stage: str, stage1: dict[str, Any], stage2: dict[str, Any]) -> tuple[str, Optional[str]]:
    active = active_stage if active_stage in {"reference_style", "kokoro_identity"} else "reference_style"
    preferred = stage2 if active == "kokoro_identity" else stage1
    fallback = stage1 if active == "kokoro_identity" else stage2

    if preferred.get("public_audio_path"):
        return active, preferred.get("public_audio_path")
    if fallback.get("public_audio_path"):
        return "reference_style" if active == "kokoro_identity" else "kokoro_identity", fallback.get("public_audio_path")
    return active, None


def _voice_line_batch_item(request: VoiceLineBatchRequest, item: VoiceLineBatchItem) -> dict[str, Any]:
    is_echo = item.intent == "module_echo_correct"
    is_correct_echo = item.intent == "module_echo_correct"
    generate_stage2 = request.generate_stage2 and request.mode != "pregenerate_stage1"
    synthesis_source_text = (item.synthesis_text or "").strip() or item.text
    synth_request = SynthesizeRequest(
        agent=item.agent,
        text=synthesis_source_text,
        intent=item.intent,
        line_key=item.line_key,
        engine=tts_config.expressive_engine_name(),
        expressive=True,
        cache=False,
        context="voice_line_pregeneration",
        metadata={
            "intent": item.intent,
            "prosody_intent": item.intent,
            "line_key": item.line_key,
        },
        humanize=not is_echo,
        delivery_control=not is_correct_echo,
        audio_humanizer=False,
        pause_control=not is_correct_echo,
    )
    prepared = audio_path_for(synth_request)
    reference_override = _override_reference_selection(item, prepared)
    agent_key = prepared.profile.agent_key
    filename = _voice_line_filename(item.line_key)
    output_root = Path(request.output_root)
    folder_intent = item.intent if is_echo else prepared.intent_result.intent
    intent_parts = _voice_line_intent_parts(folder_intent, item.line_key)
    reference_style_public = _public_relative_path(request.public_relative_root, "reference_style", agent_key, *intent_parts, filename)
    kokoro_identity_public = _public_relative_path(request.public_relative_root, "kokoro_identity", agent_key, *intent_parts, filename)
    reference_style_output = output_root / "reference_style" / agent_key / Path(*intent_parts) / filename
    kokoro_identity_output = output_root / "kokoro_identity" / agent_key / Path(*intent_parts) / filename

    reference = reference_override or prepared.reference_selection
    stage1_speaker = reference.path
    stage1_style = reference.path
    stage1_request = EngineRequest(
        text=prepared.synthesis_text,
        agent=prepared.agent,
        profile=prepared.profile,
        voice=prepared.voice,
        speed=prepared.speed,
        output_path=reference_style_output,
        intent=prepared.intent_result.intent,
        emotion_prompt=prepared.emotion_prompt,
        style_reference_path=stage1_style,
        speaker_reference_path=stage1_speaker,
        audio_humanizer_request=False,
        pause_control_request=True,
    )
    stage1 = _generate_voice_line_stage(
        stage1_request,
        reference_style_public,
        "stage1_reference_style",
        request.force,
        stage1_style,
        stage1_speaker,
    )

    if generate_stage2:
        kokoro_speaker = kokoro_reference_path(prepared.agent, prepared.profile, CACHE_DIR)
        try:
            ensure_kokoro_timbre_reference(prepared.agent, prepared.profile, CACHE_DIR, generate_audio)
        except Exception as exc:
            logger.warning("Could not prepare Kokoro speaker reference for %s: %s", prepared.agent, exc)

        stage2_style = reference_style_output if reference_style_output.exists() else reference.path
        stage2 = _generate_voice_line_stage(
            EngineRequest(
                text=prepared.synthesis_text,
                agent=prepared.agent,
                profile=prepared.profile,
                voice=prepared.voice,
                speed=prepared.speed,
                output_path=kokoro_identity_output,
                intent=prepared.intent_result.intent,
                emotion_prompt=prepared.emotion_prompt,
                style_reference_path=stage2_style,
                speaker_reference_path=kokoro_speaker,
                audio_humanizer_request=False,
                pause_control_request=True,
            ),
            kokoro_identity_public,
            "stage2_kokoro_identity",
            request.force,
            stage2_style,
            kokoro_speaker,
        )
    else:
        stage2 = {
            "status": "skipped",
            "error": "stage2_disabled_for_request",
            "public_audio_path": None,
            "duration_seconds": None,
            "engine_used": None,
            "speaker_reference_path": None,
            "style_source_path": None,
        }

    active_type, active_path = _active_batch_path(request.active_stage, stage1, stage2)
    stage1_ready = bool(stage1.get("public_audio_path"))
    stage2_ready = bool(stage2.get("public_audio_path"))
    if generate_stage2:
        status = "generated" if stage1_ready and stage2_ready else "fallback_generated" if stage1_ready or stage2_ready else "failed"
        generation_error = None if status != "failed" else "; ".join(filter(None, [stage1.get("error"), stage2.get("error")])) or "generation_failed"
    else:
        status = "generated" if stage1_ready else "failed"
        generation_error = None if stage1_ready else stage1.get("error") or "stage1_generation_failed"
    active_file = reference_style_output if active_path == reference_style_public else kokoro_identity_output if active_path == kokoro_identity_public else None

    return {
        "id": item.id,
        "line_key": item.line_key,
        "status": status,
        "active_audio_type": active_type,
        "active_audio_path": active_path,
        "defense_audio_path": stage1.get("public_audio_path"),
        "stage2_demo_audio_path": stage2.get("public_audio_path"),
        "stage1": stage1,
        "stage2": {
            **stage2,
            "kokoro_voice_id": prepared.voice,
        },
        "reference": {
            "path": reference.relative_path,
            "absolute_path": str(reference.path) if reference.path else None,
            "duration_seconds": reference.duration_seconds,
            "priority": reference.priority,
            "weight": reference.weight,
            "weighting_version": reference.weighting_version,
            "fallback_reason": reference.fallback_reason,
        },
        "emotion_prompt": prepared.emotion_prompt,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "format": "wav",
        "cache_key": prepared.cache_key,
        "checksum": _file_checksum(active_file),
        "generation_error": generation_error,
    }


@app.get("/health")
def health():
    index_engine = _index_tts2_engine()
    index_available, index_reason = index_engine.availability()
    manifest = _load_reference_manifest()

    return {
        "ok": True,
        "service": SERVICE_NAME,
        "provider": tts_config.default_engine(),
        "pipeline_version": PIPELINE_VERSION,
        "agent_profiles_enabled": profiles_enabled(),
        "pipeline": {
            "loaded": pipeline is not None,
            "warmup_on_startup": warmup_on_startup_enabled(),
        },
        "engines": {
            "default": tts_config.default_engine(),
            "kokoro": {
                "available": True,
                "cache_root": str(CACHE_DIR),
            },
            "index_tts2": {
                "enabled": tts_config.expressive_feature_enabled(),
                "selected_by_default": tts_config.expressive_engine_enabled(),
                "available": index_available,
                "availability_reason": index_reason,
                "adapter_module": tts_config.index_tts2_adapter_module() or None,
                "fallback_to_kokoro": tts_config.expressive_fallback_to_kokoro(),
                "cache_root": str(EXPRESSIVE_CACHE_DIR),
            },
        },
        "references": {
            "manifest_path": str(tts_config.reference_manifest_path()),
            "audio_root": str(tts_config.reference_audio_root()),
            "manifest_loaded": manifest.loaded,
            "load_reason": manifest.load_reason,
            "weighting_enabled": tts_config.reference_weighting_enabled(),
            "weighting_version": tts_config.reference_weighting_version(),
        },
        "cache": {
            "enabled": tts_config.cache_enabled(True),
            "root": str(tts_config.cache_root()),
            "bypass": env_bool("TTS_CACHE_BYPASS", False),
            "comparison_output_root": str(tts_config.comparison_output_root()),
        },
        "voices": {
            "miss_vivian": AGENT_PROFILES["miss_vivian"].voice,
            "miss_ciel": AGENT_PROFILES["miss_ciel"].voice,
            "miss_estelle": AGENT_PROFILES["miss_estelle"].voice,
        },
        "humanization": {
            "auto_prompt_extension_enabled": tts_config.auto_prompt_extension_enabled(),
            "curated_prompts_enabled": tts_config.curated_prompts_enabled(),
            "curated_prompt_target_seconds": tts_config.curated_prompt_target_seconds(),
            "curated_prompt_min_seconds": tts_config.curated_prompt_min_seconds(),
            "curated_prompt_max_seconds": tts_config.curated_prompt_max_seconds(),
            "text_humanizer_enabled": text_humanizer_enabled(),
            "delivery_control_enabled": env_bool("TTS_DELIVERY_CONTROL_ENABLED", True),
            "safe_chunking_enabled": env_bool("TTS_SAFE_CHUNKING_ENABLED", True),
            "audio_humanizer_enabled": audio_humanizer_enabled(),
            "pause_control_enabled": pause_control_enabled(),
            "breaths_enabled": env_bool("TTS_BREATHS_ENABLED", False),
        },
        "two_stage_generation": {
            "enabled": tts_config.two_stage_generation_enabled(),
            "stage1_reference_style_enabled": tts_config.stage1_reference_style_enabled(),
            "stage2_kokoro_identity_enabled": tts_config.stage2_kokoro_identity_enabled(),
            "active_stage": tts_config.active_stage(),
        },
    }


@app.get("/voices")
def voices():
    return {"agents": agent_profiles_payload()}


@app.post("/synthesize")
def synthesize(request: SynthesizeRequest):
    prepared = audio_path_for(request)
    audio_result = _empty_audio_result()
    engine_result = _cached_engine_result(prepared, audio_result)
    cache_hit = tts_config.cache_enabled(request.cache) and prepared.output_path.exists()
    started = time.perf_counter()
    generation_duration_ms = 0

    if not cache_hit:
        try:
            engine_result = generate_prepared_audio(prepared, request)
            audio_result = engine_result.audio_result or _empty_audio_result()
            generation_duration_ms = int(round((time.perf_counter() - started) * 1000))
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
        engine_result = _cached_engine_result(prepared, audio_result)

    log_tts_request(prepared, engine_result, cache_hit, generation_duration_ms)

    return FileResponse(
        prepared.output_path,
        media_type="audio/wav",
        filename=f"{prepared.cache_key}.wav",
        headers={
            "X-ReaDirect-TTS-Provider": engine_result.engine_used,
            "X-ReaDirect-TTS-Requested-Engine": engine_result.engine_requested,
            "X-ReaDirect-TTS-Agent": prepared.agent,
            "X-ReaDirect-TTS-Voice": prepared.voice,
            "X-ReaDirect-TTS-Speed": f"{prepared.speed:.2f}",
            "X-ReaDirect-TTS-Humanized": "1" if prepared.text_result.applied else "0",
            "X-ReaDirect-TTS-Intent": prepared.intent_result.intent,
            "X-ReaDirect-TTS-Reference": prepared.reference_selection.relative_path or "",
            "X-ReaDirect-TTS-Expressive": "1" if engine_result.expressive_used else "0",
            "X-ReaDirect-TTS-Fallback-Reason": engine_result.fallback_reason or "",
            "X-ReaDirect-TTS-Mode": str(engine_result.metadata.get("mode", "fallback_kokoro" if engine_result.fallback_reason else engine_result.engine_used)) if engine_result.metadata or engine_result.fallback_reason else engine_result.engine_used,
            "X-ReaDirect-TTS-Cache-Key": prepared.cache_key,
        },
    )


@app.post("/voice-lines/generate-batch")
def generate_voice_line_batch(request: VoiceLineBatchRequest):
    if request.mode not in {"pregenerate_two_stage", "pregenerate_stage1"}:
        raise HTTPException(status_code=422, detail="Unsupported batch generation mode.")
    if not tts_config.two_stage_generation_enabled():
        raise HTTPException(status_code=409, detail="Two-stage generation is disabled.")
    if len(request.items) > 50:
        raise HTTPException(status_code=422, detail="Batch is limited to 50 voice lines.")

    started = time.perf_counter()
    items: list[dict[str, Any]] = []
    for item in request.items:
        try:
            items.append(_voice_line_batch_item(request, item))
        except Exception as exc:
            items.append(
                {
                    "id": item.id,
                    "line_key": item.line_key,
                    "status": "failed",
                    "active_audio_type": request.active_stage,
                    "active_audio_path": None,
                    "defense_audio_path": None,
                    "stage2_demo_audio_path": None,
                    "stage1": {"status": "failed", "error": f"{exc.__class__.__name__}:{exc}"},
                    "stage2": {"status": "failed", "error": "skipped_after_stage1_setup_failure"},
                    "reference": {},
                    "generation_error": f"{exc.__class__.__name__}:{exc}",
                }
            )

    return {
        "service": SERVICE_NAME,
        "mode": request.mode,
        "active_stage": request.active_stage,
        "items": items,
        "duration_ms": int(round((time.perf_counter() - started) * 1000)),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)
