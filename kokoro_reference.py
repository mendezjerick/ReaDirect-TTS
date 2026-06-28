from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from agent_voice_profiles import AgentVoiceProfile
from audio_humanizer import AudioHumanizerResult
from prosody_intents import reference_agent_key


REFERENCE_LINES = {
    "miss_ciel": "Hi, I'm Miss Ciel. I'll read with you today.",
    "miss_vivian": "Hello, I'm Miss Vivian. Listen carefully and take your time.",
    "miss_estelle": "Hello, I'm Miss Estelle. Let's look at your result together.",
}


def kokoro_reference_path(agent: str, profile: AgentVoiceProfile, cache_root: Path) -> Path:
    agent_key = reference_agent_key(agent)
    return cache_root / "reference_voice" / f"{agent_key}_{profile.voice}.wav"


def ensure_kokoro_timbre_reference(
    agent: str,
    profile: AgentVoiceProfile,
    cache_root: Path,
    generate_audio: Callable[..., AudioHumanizerResult],
    force: bool = False,
) -> Optional[Path]:
    output_path = kokoro_reference_path(agent, profile, cache_root)
    if output_path.exists() and not force:
        return output_path

    line = REFERENCE_LINES.get(agent)
    if not line:
        return None

    generate_audio(
        line,
        profile.voice,
        profile.default_speed,
        output_path,
        audio_humanizer_request=False,
        pause_control_request=False,
    )
    return output_path
