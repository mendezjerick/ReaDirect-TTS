from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import soundfile as sf

from prosody_intents import default_intent_for_agent
import tts_config


ALLOWED_REFERENCE_EXTENSIONS = {".wav", ".mp3"}


@dataclass(frozen=True)
class ReferenceSelection:
    requested_intent: str
    selected_intent: str
    agent_key: str
    path: Optional[Path]
    relative_path: Optional[str]
    fallback_reason: Optional[str]
    manifest_loaded: bool
    duration_seconds: Optional[float] = None
    priority: str = "none"
    weight: int = 0
    weighting_version: str = "none"

    @property
    def selected(self) -> bool:
        return self.path is not None


class ReferenceManifest:
    def __init__(
        self,
        manifest_path: Path,
        audio_root: Path,
        data: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
        loaded: bool = False,
        load_reason: Optional[str] = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.audio_root = audio_root
        self.data = data or {}
        self.loaded = loaded
        self.load_reason = load_reason

    @classmethod
    def load(cls, manifest_path: Path, audio_root: Path, logger: Optional[logging.Logger] = None) -> "ReferenceManifest":
        if not manifest_path.exists():
            if logger:
                logger.info("No expressive reference manifest configured at %s; expressive references are disabled.", manifest_path)
            return cls(manifest_path, audio_root, loaded=False, load_reason="manifest_missing")

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            if logger:
                logger.warning("Expressive reference manifest could not be loaded: %s", exc)
            return cls(manifest_path, audio_root, loaded=False, load_reason="manifest_invalid")

        if not isinstance(payload, Mapping):
            return cls(manifest_path, audio_root, loaded=False, load_reason="manifest_not_object")

        clean: dict[str, dict[str, list[str]]] = {}
        for agent_key, intents in payload.items():
            if not isinstance(intents, Mapping):
                continue
            clean[str(agent_key)] = {}
            for intent, paths in intents.items():
                if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
                    clean[str(agent_key)][str(intent)] = [str(path) for path in paths]

        return cls(manifest_path, audio_root, clean, loaded=True)

    def _reference_duration(self, path: Path) -> Optional[float]:
        try:
            return float(sf.info(path).duration)
        except Exception:
            return None

    def _priority_and_weight(self, duration_seconds: Optional[float]) -> tuple[str, int]:
        if duration_seconds is None:
            return "unknown", 1
        if duration_seconds >= tts_config.reference_high_priority_min_seconds():
            return "high", 5
        if duration_seconds >= tts_config.reference_medium_priority_min_seconds():
            return "medium", 2
        return "low", 1

    def _valid_candidates(self, agent_key: str, intent: str) -> list[tuple[Path, str, Optional[float], str, int]]:
        candidates: list[tuple[Path, str, Optional[float], str, int]] = []
        for relative in self.data.get(agent_key, {}).get(intent, []):
            if Path(relative).is_absolute():
                path = Path(relative)
            else:
                path = self.audio_root / relative

            if path.suffix.lower() not in ALLOWED_REFERENCE_EXTENSIONS:
                continue
            if path.exists() and path.is_file():
                duration = self._reference_duration(path)
                priority, weight = self._priority_and_weight(duration)
                candidates.append((path, relative, duration, priority, weight))

        return candidates

    def _select_candidate(
        self,
        candidates: list[tuple[Path, str, Optional[float], str, int]],
        digest: str,
    ) -> tuple[Path, str, Optional[float], str, int]:
        if not tts_config.reference_weighting_enabled():
            path, relative, duration, priority, _weight = candidates[int(digest[:8], 16) % len(candidates)]
            return path, relative, duration, priority, 1

        total_weight = sum(max(1, candidate[4]) for candidate in candidates)
        bucket = int(digest[:12], 16) % total_weight
        running = 0
        for candidate in candidates:
            running += max(1, candidate[4])
            if bucket < running:
                return candidate

        return candidates[-1]

    def select(self, agent_key: str, intent: str, cache_seed: str) -> ReferenceSelection:
        fallback_reason = self.load_reason
        if not self.loaded:
            return ReferenceSelection(intent, intent, agent_key, None, None, fallback_reason, False)

        intent_order = [intent]
        default_intent = default_intent_for_agent(agent_key)
        if default_intent not in intent_order:
            intent_order.append(default_intent)

        for candidate_intent in intent_order:
            candidates = self._valid_candidates(agent_key, candidate_intent)
            if not candidates:
                continue

            digest = hashlib.sha256(f"{agent_key}|{intent}|{cache_seed}".encode("utf-8")).hexdigest()
            path, relative, duration, priority, weight = self._select_candidate(candidates, digest)
            reason = None if candidate_intent == intent else f"reference_intent_fallback:{candidate_intent}"
            return ReferenceSelection(
                intent,
                candidate_intent,
                agent_key,
                path,
                relative,
                reason,
                True,
                duration_seconds=duration,
                priority=priority,
                weight=weight,
                weighting_version=tts_config.reference_weighting_version() if tts_config.reference_weighting_enabled() else "equal-v1",
            )

        return ReferenceSelection(
            intent,
            intent,
            agent_key,
            None,
            None,
            "no_reference_files_configured",
            True,
        )


def reference_file_digest(path: Optional[Path]) -> str:
    if path is None:
        return "none"

    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8"))
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return "unreadable"

    return digest.hexdigest()


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio

    source_positions = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    target_length = max(1, int(round(audio.size * (target_rate / source_rate))))
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def normalize_reference_audio(source_path: Path, output_root: Path, sample_rate: int = 24000) -> Optional[Path]:
    try:
        audio, rate = sf.read(source_path, dtype="float32")
    except Exception:
        return None

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)
    if rate != sample_rate:
        audio = _resample_linear(audio, rate, sample_rate)

    if audio.size:
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            factor = 0.88 / peak if peak > 0.88 else min(1.25, 0.45 / peak) if peak < 0.35 else 1.0
            if factor != 1.0:
                audio = np.asarray(audio * factor, dtype=np.float32)
        audio = np.clip(audio, -0.98, 0.98)

    digest = reference_file_digest(source_path)[:16]
    output_path = output_root / "normalized_references" / f"{source_path.stem}_{digest}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    return output_path
