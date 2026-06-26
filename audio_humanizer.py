from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from agent_voice_profiles import env_bool, env_float


SENTENCE_SPLIT_PATTERN = re.compile(r"[^.!?]+[.!?]")


@dataclass(frozen=True)
class AudioHumanizerResult:
    enabled: bool
    normalized: bool
    faded: bool
    pause_control_applied: bool
    breath_applied: bool
    duration_seconds: float


def audio_humanizer_enabled(request_value: bool | None = None) -> bool:
    if request_value is not None:
        return request_value and env_bool("TTS_HUMANIZER_ENABLED", True)

    return env_bool("TTS_HUMANIZER_ENABLED", True)


def audio_normalize_enabled() -> bool:
    return env_bool("TTS_AUDIO_NORMALIZE_ENABLED", True)


def audio_fade_enabled() -> bool:
    return env_bool("TTS_AUDIO_FADE_ENABLED", True)


def pause_control_enabled(request_value: bool | None = None) -> bool:
    if request_value is not None:
        return request_value and env_bool("TTS_PAUSE_CONTROL_ENABLED", True)

    return env_bool("TTS_PAUSE_CONTROL_ENABLED", True)


def breaths_enabled() -> bool:
    return env_bool("TTS_BREATHS_ENABLED", False)


def breaths_volume() -> float:
    return max(0.0, min(0.2, env_float("TTS_BREATHS_VOLUME", 0.08)))


def breaths_min_text_length() -> int:
    return int(env_float("TTS_BREATHS_MIN_TEXT_LENGTH", 80))


def split_text_for_sentence_pauses(text: str) -> list[str]:
    clean = (text or "").strip()
    if not clean:
        return []

    matches = [part.strip() for part in SENTENCE_SPLIT_PATTERN.findall(clean) if part.strip()]
    remainder = SENTENCE_SPLIT_PATTERN.sub("", clean).strip()
    if remainder:
        matches.append(remainder)

    if len(matches) <= 1 or len(matches) > 4:
        return [clean]

    return matches


def sentence_pause(sample_rate: int, sentence_text: str) -> np.ndarray:
    words = len(re.findall(r"[A-Za-z0-9']+", sentence_text or ""))
    milliseconds = 90 if words <= 8 else 145
    return np.zeros(int(sample_rate * (milliseconds / 1000.0)), dtype=np.float32)


def _normalize(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    if not audio.size:
        return audio, False

    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio, False

    factor = 1.0
    if peak > 0.95:
        factor = 0.92 / peak
    elif peak < 0.35:
        factor = min(1.35, 0.45 / peak)

    if abs(factor - 1.0) < 0.01:
        return audio, False

    return np.asarray(audio * factor, dtype=np.float32), True


def _fade(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, bool]:
    fade_samples = int(sample_rate * 0.008)
    if fade_samples <= 0 or audio.size <= fade_samples * 2:
        return audio, False

    faded = np.array(audio, dtype=np.float32, copy=True)
    faded[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    faded[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    return faded, True


def _load_breath_sample(sample_rate: int, breath_dir: Path) -> np.ndarray | None:
    if not breath_dir.exists():
        return None

    for path in sorted(breath_dir.glob("*.wav")):
        try:
            sample, rate = sf.read(path, dtype="float32")
        except Exception:
            continue

        if rate != sample_rate:
            continue

        if sample.ndim > 1:
            sample = np.mean(sample, axis=1)

        if sample.size:
            return np.asarray(sample, dtype=np.float32)

    return None


def should_insert_breath(text: str) -> bool:
    if not breaths_enabled():
        return False

    clean = (text or "").strip()
    if len(clean) < breaths_min_text_length():
        return False

    words = re.findall(r"[A-Za-z0-9']+", clean)
    if len(words) <= 4:
        return False

    return True


def _insert_breath(audio: np.ndarray, text: str, sample_rate: int, breath_dir: Path) -> tuple[np.ndarray, bool]:
    if not should_insert_breath(text):
        return audio, False

    breath = _load_breath_sample(sample_rate, breath_dir)
    if breath is None:
        return audio, False

    breath = np.asarray(breath * breaths_volume(), dtype=np.float32)
    spacer = np.zeros(int(sample_rate * 0.06), dtype=np.float32)
    return np.concatenate([breath, spacer, audio]), True


def humanize_audio(
    audio: np.ndarray,
    text: str,
    sample_rate: int,
    breath_dir: Path,
    request_enabled: bool | None = None,
    pause_already_applied: bool = False,
) -> tuple[np.ndarray, AudioHumanizerResult]:
    enabled = audio_humanizer_enabled(request_enabled)
    if not enabled:
        return audio, AudioHumanizerResult(False, False, False, False, False, len(audio) / sample_rate)

    processed = np.asarray(audio, dtype=np.float32)
    normalized = False
    faded = False
    breath_applied = False

    if audio_normalize_enabled():
        processed, normalized = _normalize(processed)

    if audio_fade_enabled():
        processed, faded = _fade(processed, sample_rate)

    processed, breath_applied = _insert_breath(processed, text, sample_rate, breath_dir)

    return processed, AudioHumanizerResult(
        enabled=True,
        normalized=normalized,
        faded=faded,
        pause_control_applied=pause_already_applied,
        breath_applied=breath_applied,
        duration_seconds=len(processed) / sample_rate,
    )


def join_audio_chunks_with_pauses(
    chunks: Sequence[np.ndarray],
    sentence_texts: Sequence[str],
    sample_rate: int,
    request_enabled: bool | None = None,
) -> tuple[np.ndarray, bool]:
    if not chunks:
        return np.array([], dtype=np.float32), False

    if len(chunks) == 1 or not pause_control_enabled(request_enabled):
        return np.concatenate(chunks) if len(chunks) > 1 else chunks[0], False

    joined: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        joined.append(chunk)
        if index < len(chunks) - 1:
            joined.append(sentence_pause(sample_rate, sentence_texts[index]))

    return np.concatenate(joined), True
