from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError:
        return default


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def is_isabella_voice_id(voice_id: str) -> bool:
    return "isabella" in (voice_id or "").strip().lower()


def resolve_estelle_voice_id() -> str:
    """Allow only an Isabella-named local override for Estelle."""
    for env_name in ("TTS_AGENT_VOICE_ESTELLE", "TTS_VOICE_ESTELLE"):
        candidate = (os.getenv(env_name) or "").strip()
        if candidate and is_isabella_voice_id(candidate):
            return candidate

    return "bf_isabella"


@dataclass(frozen=True)
class AgentVoiceProfile:
    agent: str
    agent_key: str
    display_name: str
    voice: str
    default_speed: float
    min_speed: float
    max_speed: float
    role: str
    personality: tuple[str, ...]
    delivery: tuple[str, ...]
    emotional_direction: str
    delivery_style: str
    aliases: tuple[str, ...] = ()

    def speed_for_request(self, requested_speed: Optional[float]) -> float:
        speed = self.default_speed if requested_speed is None else float(requested_speed)
        return round(clamp(speed, self.min_speed, self.max_speed), 2)

    def to_public_dict(self) -> dict:
        payload = asdict(self)
        payload["speed"] = self.default_speed
        payload["speed_range"] = {
            "min": self.min_speed,
            "max": self.max_speed,
        }
        return payload


def profiles_enabled() -> bool:
    return env_bool("TTS_AGENT_PROFILES_ENABLED", True)


def _profile_speed(env_name: str, default: float, minimum: float, maximum: float) -> float:
    if not profiles_enabled():
        return default

    return round(clamp(env_float(env_name, default), minimum, maximum), 2)


def load_agent_profiles() -> Dict[str, AgentVoiceProfile]:
    return {
        "miss_ciel": AgentVoiceProfile(
            agent="miss_ciel",
            agent_key="ciel",
            display_name="Miss Ciel",
            voice="af_heart",
            default_speed=_profile_speed("TTS_AGENT_SPEED_CIEL", 0.94, 0.92, 0.96),
            min_speed=0.92,
            max_speed=0.96,
            role="reading coach",
            personality=("warm", "gentle", "patient", "friendly"),
            delivery=("warm", "gentle", "patient", "friendly"),
            emotional_direction="encouraging, soft, calm, never harsh",
            delivery_style=(
                "sounds like a supportive friend helping the learner read, "
                "not a strict teacher"
            ),
            aliases=("coach_feedback", "ciel"),
        ),
        "miss_vivian": AgentVoiceProfile(
            agent="miss_vivian",
            agent_key="vivian",
            display_name="Miss Vivian",
            voice="af_bella",
            default_speed=_profile_speed("TTS_AGENT_SPEED_VIVIAN", 0.97, 0.95, 1.00),
            min_speed=0.95,
            max_speed=1.00,
            role="assessment guide",
            personality=("clear", "friendly", "encouraging"),
            delivery=("clear", "friendly", "encouraging"),
            emotional_direction="encouraging but controlled",
            delivery_style=(
                "friendly guide who gives clear instructions without sounding robotic"
            ),
            aliases=("assessment", "vivian"),
        ),
        "miss_estelle": AgentVoiceProfile(
            agent="miss_estelle",
            agent_key="estelle",
            display_name="Miss Estelle",
            voice=resolve_estelle_voice_id(),
            default_speed=_profile_speed("TTS_AGENT_SPEED_ESTELLE", 0.93, 0.90, 0.95),
            min_speed=0.90,
            max_speed=0.95,
            role="evaluator",
            personality=("calm", "reassuring", "kind"),
            delivery=("calm", "reassuring", "kind"),
            emotional_direction="gentle, formal, not too playful",
            delivery_style="explains results in a comforting and human way",
            aliases=("evaluator", "evaluator_recommendation", "estelle"),
        ),
    }


AGENT_PROFILES = load_agent_profiles()


def agent_aliases(profiles: Mapping[str, AgentVoiceProfile] = AGENT_PROFILES) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for agent, profile in profiles.items():
        aliases[agent] = agent
        for alias in profile.aliases:
            aliases[alias] = agent

    return aliases


def agent_profiles_payload(profiles: Mapping[str, AgentVoiceProfile] = AGENT_PROFILES) -> dict:
    return {agent: profile.to_public_dict() for agent, profile in profiles.items()}
