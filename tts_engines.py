from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import soundfile as sf

from agent_voice_profiles import AgentVoiceProfile
from audio_humanizer import AudioHumanizerResult
from expressive_references import normalize_reference_audio


class KokoroGenerateCallable(Protocol):
    def __call__(
        self,
        text: str,
        voice: str,
        speed: float,
        output_path: Path,
        audio_humanizer_request: Optional[bool] = None,
        pause_control_request: Optional[bool] = None,
    ) -> AudioHumanizerResult:
        ...


@dataclass(frozen=True)
class EngineRequest:
    text: str
    agent: str
    profile: AgentVoiceProfile
    voice: str
    speed: float
    output_path: Path
    intent: str
    emotion_prompt: str
    style_reference_path: Optional[Path] = None
    speaker_reference_path: Optional[Path] = None
    audio_humanizer_request: Optional[bool] = None
    pause_control_request: Optional[bool] = None


@dataclass(frozen=True)
class EngineResult:
    output_path: Path
    engine_requested: str
    engine_used: str
    audio_result: AudioHumanizerResult
    expressive_used: bool = False
    speaker_reference_used: bool = False
    style_reference_path: Optional[Path] = None
    fallback_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TtsEngine(Protocol):
    name: str

    def generate(self, request: EngineRequest) -> EngineResult:
        ...


class KokoroEngine:
    name = "kokoro"

    def __init__(self, generate_audio: KokoroGenerateCallable) -> None:
        self.generate_audio = generate_audio

    def _empty_audio_result(self) -> AudioHumanizerResult:
        return AudioHumanizerResult(
            enabled=False,
            normalized=False,
            faded=False,
            pause_control_applied=False,
            breath_applied=False,
            duration_seconds=0.0,
        )

    def generate(self, request: EngineRequest) -> EngineResult:
        audio_result = self.generate_audio(
            request.text,
            request.voice,
            request.speed,
            request.output_path,
            request.audio_humanizer_request,
            request.pause_control_request,
        )
        if audio_result is None:
            audio_result = self._empty_audio_result()

        return EngineResult(
            output_path=request.output_path,
            engine_requested=self.name,
            engine_used=self.name,
            audio_result=audio_result,
        )


class IndexTTS2ExpressiveEngine:
    name = "index_tts2"

    def __init__(
        self,
        kokoro_fallback: KokoroEngine,
        fallback_to_kokoro: bool,
        adapter_module_name: str = "",
        normalized_reference_root: Optional[Path] = None,
    ) -> None:
        self.kokoro_fallback = kokoro_fallback
        self.fallback_to_kokoro = fallback_to_kokoro
        self.adapter_module_name = adapter_module_name
        self.normalized_reference_root = normalized_reference_root

    def _adapter_module(self):
        try:
            return importlib.import_module(self.adapter_module_name)
        except Exception as exc:
            raise RuntimeError(f"adapter_import_failed:{exc.__class__.__name__}") from exc

    def availability(self) -> tuple[bool, str]:
        try:
            module = self._adapter_module()
        except RuntimeError as exc:
            return False, str(exc)

        if not callable(getattr(module, "generate", None)):
            return False, "adapter_generate_callable_missing"

        status = getattr(module, "status", None)
        if callable(status):
            try:
                payload = status()
            except Exception as exc:
                return False, f"adapter_status_failed:{exc.__class__.__name__}"
            if isinstance(payload, dict):
                available = bool(payload.get("available"))
                reason = str(payload.get("reason") or ("available" if available else "adapter_unavailable"))
                detail = payload.get("details")
                if detail:
                    reason = f"{reason}:{detail}"
                return available, reason

        return True, "available"

    def _adapter_generate(self, request: EngineRequest, style_reference_path: Optional[Path]) -> dict[str, Any]:
        module = self._adapter_module()
        generate = getattr(module, "generate")
        result = generate(
            text=request.text,
            output_path=str(request.output_path),
            agent=request.agent,
            kokoro_voice=request.voice,
            speed=request.speed,
            speaker_reference_path=str(request.speaker_reference_path) if request.speaker_reference_path else None,
            style_reference_path=str(style_reference_path) if style_reference_path else None,
            emotion_prompt=request.emotion_prompt,
            intent=request.intent,
        )

        if isinstance(result, bytes):
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            request.output_path.write_bytes(result)
            return {"mode": "bytes"}
        elif isinstance(result, (str, Path)):
            generated_path = Path(result)
            if generated_path != request.output_path and generated_path.exists():
                request.output_path.parent.mkdir(parents=True, exist_ok=True)
                request.output_path.write_bytes(generated_path.read_bytes())
            return {"mode": "path", "output_path": str(generated_path)}
        elif isinstance(result, dict):
            return result

        return {"mode": "unknown"}

    def _fallback(self, request: EngineRequest, reason: str) -> EngineResult:
        if not self.fallback_to_kokoro:
            raise RuntimeError(f"IndexTTS2 expressive TTS unavailable: {reason}")

        fallback_result = self.kokoro_fallback.generate(request)
        return EngineResult(
            output_path=fallback_result.output_path,
            engine_requested=self.name,
            engine_used=fallback_result.engine_used,
            audio_result=fallback_result.audio_result,
            expressive_used=False,
            speaker_reference_used=False,
            style_reference_path=request.style_reference_path,
            fallback_reason=reason,
        )

    def _audio_result_for_output(self, output_path: Path) -> AudioHumanizerResult:
        duration = 0.0
        try:
            duration = float(sf.info(output_path).duration)
        except Exception:
            pass

        return AudioHumanizerResult(
            enabled=False,
            normalized=False,
            faded=False,
            pause_control_applied=False,
            breath_applied=False,
            duration_seconds=duration,
        )

    def generate(self, request: EngineRequest) -> EngineResult:
        available, reason = self.availability()
        if not available:
            return self._fallback(request, reason)

        if request.style_reference_path is None:
            return self._fallback(request, "style_reference_missing")

        style_reference_path = request.style_reference_path
        if self.normalized_reference_root is not None:
            normalized = normalize_reference_audio(request.style_reference_path, self.normalized_reference_root)
            if normalized is not None:
                style_reference_path = normalized

        try:
            metadata = self._adapter_generate(request, style_reference_path)
        except Exception as exc:
            reason = getattr(exc, "reason", None)
            details = getattr(exc, "details", None)
            fallback_reason = str(reason or f"expressive_generation_failed:{exc.__class__.__name__}")
            if details:
                fallback_reason = f"{fallback_reason}:{details}"
            return self._fallback(request, fallback_reason)

        if not request.output_path.exists():
            return self._fallback(request, "expressive_output_missing")

        return EngineResult(
            output_path=request.output_path,
            engine_requested=self.name,
            engine_used=self.name,
            audio_result=self._audio_result_for_output(request.output_path),
            expressive_used=True,
            speaker_reference_used=request.speaker_reference_path is not None,
            style_reference_path=style_reference_path,
            metadata=metadata,
        )
