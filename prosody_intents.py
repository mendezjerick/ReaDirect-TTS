from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional


SUPPORTED_INTENTS = (
    "intro",
    "gentle_reassurance",
    "friendly_encouragement",
    "happy_praise",
    "focused_instruction",
    "calm_evaluation",
    "playful_friend",
    "module_echo_correct",
)

AGENT_REFERENCE_KEYS = {
    "miss_ciel": "ciel",
    "ciel": "ciel",
    "coach_feedback": "ciel",
    "miss_vivian": "vivian",
    "vivian": "vivian",
    "assessment": "vivian",
    "miss_estelle": "estelle",
    "estelle": "estelle",
    "evaluator": "estelle",
    "evaluator_recommendation": "estelle",
}

AGENT_DEFAULT_INTENTS = {
    "ciel": "friendly_encouragement",
    "vivian": "focused_instruction",
    "estelle": "calm_evaluation",
}

AGENT_ALLOWED_INTENTS = {
    "ciel": {
        "intro",
        "gentle_reassurance",
        "friendly_encouragement",
        "happy_praise",
        "focused_instruction",
        "playful_friend",
        "module_echo_correct",
    },
    "vivian": {
        "intro",
        "gentle_reassurance",
        "friendly_encouragement",
        "happy_praise",
        "focused_instruction",
    },
    "estelle": {
        "intro",
        "gentle_reassurance",
        "calm_evaluation",
        "happy_praise",
        "focused_instruction",
    },
}

INTENT_PROMPTS = {
    "intro": "warm, friendly, welcoming, gentle",
    "gentle_reassurance": "gentle, warm, reassuring",
    "friendly_encouragement": "cheerful and encouraging",
    "happy_praise": "bright, pleased, supportive praise",
    "focused_instruction": "focused, clear, patient instruction",
    "calm_evaluation": "calm and supportive",
    "playful_friend": "light, friendly, playful reading buddy",
    "module_echo_correct": "clear, exact target pronunciation",
}


def emotion_prompt_for_intent(intent: str) -> str:
    env_name = f"TTS_EMOTION_PROMPT_{intent.upper()}"
    configured = (os.getenv(env_name) or "").strip().strip("\"'")
    if configured:
        return configured

    return INTENT_PROMPTS.get(intent, INTENT_PROMPTS["friendly_encouragement"])


@dataclass(frozen=True)
class ProsodyIntentResult:
    intent: str
    agent_key: str
    reason: str
    matched_text: Optional[str] = None

    @property
    def emotion_prompt(self) -> str:
        return emotion_prompt_for_intent(self.intent)


def reference_agent_key(agent: str) -> str:
    return AGENT_REFERENCE_KEYS.get((agent or "").strip().lower(), "vivian")


def default_intent_for_agent(agent: str) -> str:
    return AGENT_DEFAULT_INTENTS[reference_agent_key(agent)]


def allowed_intent(agent_key: str, intent: str) -> bool:
    return intent in AGENT_ALLOWED_INTENTS.get(agent_key, set())


def normalize_intent(agent_key: str, intent: str) -> str:
    clean = (intent or "").strip().lower()
    if clean in SUPPORTED_INTENTS and allowed_intent(agent_key, clean):
        return clean
    return AGENT_DEFAULT_INTENTS.get(agent_key, "friendly_encouragement")


def _flatten_context(context: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in (context or {}).items():
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                data[str(nested_key)] = nested_value
        else:
            data[str(key)] = value
    return data


def _contains(text: str, *phrases: str) -> Optional[str]:
    for phrase in phrases:
        pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return phrase
    return None


def classify_prosody_intent(agent: str, text: str, context: Optional[Mapping[str, Any]] = None) -> ProsodyIntentResult:
    agent_key = reference_agent_key(agent)
    context_data = _flatten_context(context)

    explicit = str(
        context_data.get("prosody_intent")
        or context_data.get("intent")
        or context_data.get("delivery_intent")
        or ""
    ).strip().lower()
    if explicit:
        intent = normalize_intent(agent_key, explicit)
        return ProsodyIntentResult(intent=intent, agent_key=agent_key, reason="metadata", matched_text=explicit)

    outcome = str(context_data.get("outcome") or context_data.get("state") or "").strip().lower()
    if outcome in {"correct", "success", "module_success", "module_complete"}:
        return ProsodyIntentResult("happy_praise", agent_key, "outcome", outcome)
    if outcome in {"retry", "incorrect", "wrong", "not_quite"}:
        return ProsodyIntentResult("gentle_reassurance", agent_key, "outcome", outcome)

    text_kind = str(context_data.get("context") or context_data.get("text_kind") or "").strip().lower()
    if agent_key == "estelle" and text_kind in {"result", "results", "evaluation", "assessment_result"}:
        return ProsodyIntentResult("calm_evaluation", agent_key, "context", text_kind)

    clean = re.sub(r"\s+", " ", text or "").strip().lower()

    intro_match = _contains(
        clean,
        "hi i'm miss ciel",
        "hi i am miss ciel",
        "hello, i'm miss ciel",
        "hello i am miss ciel",
        "hi i'm miss vivian",
        "hi i am miss vivian",
        "hello, i'm miss vivian",
        "hello i am miss vivian",
        "hi i'm miss estelle",
        "hi i am miss estelle",
        "hello, i'm miss estelle",
        "hello i am miss estelle",
    )
    if intro_match:
        return ProsodyIntentResult("intro", agent_key, "keyword", intro_match)

    matched = _contains(clean, "try again", "not quite", "that's okay", "that is okay", "that was close")
    if matched:
        return ProsodyIntentResult("gentle_reassurance", agent_key, "keyword", matched)

    matched = _contains(clean, "good job", "nice work", "great job", "well done", "that's correct", "that is correct", "correct")
    if matched:
        return ProsodyIntentResult("happy_praise", agent_key, "keyword", matched)

    matched = _contains(clean, "listen carefully", "say the word", "say the sound", "say the letter", "read this", "submit your answer")
    if matched:
        return ProsodyIntentResult("focused_instruction", agent_key, "keyword", matched)

    matched = _contains(clean, "you can do it", "take your time", "keep going", "when you're ready", "when you are ready")
    if matched:
        return ProsodyIntentResult("friendly_encouragement", agent_key, "keyword", matched)

    matched = _contains(clean, "ready", "let's try", "read together", "buddy")
    if matched and agent_key == "ciel":
        return ProsodyIntentResult("playful_friend", agent_key, "keyword", matched)

    if agent_key == "estelle" and _contains(clean, "result", "score", "practice next", "tricky ones"):
        return ProsodyIntentResult("calm_evaluation", agent_key, "keyword", "evaluation")

    default_intent = AGENT_DEFAULT_INTENTS[agent_key]
    return ProsodyIntentResult(default_intent, agent_key, "agent_default", None)
