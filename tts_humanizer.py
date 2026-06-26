from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from agent_voice_profiles import AgentVoiceProfile, env_bool, env_float


PROTECTED_CONTEXTS = {
    "answer_choice",
    "answer_choices",
    "asr_result",
    "asr_transcript",
    "assessment_passage",
    "comprehension_question",
    "content_bank",
    "debug",
    "expected_answer",
    "expected_spoken_answer",
    "learner_answer",
    "learner_transcript",
    "letter",
    "phrase",
    "score",
    "scoring_label",
    "system",
    "target",
    "target_letter",
    "target_phrase",
    "target_word",
    "transcript",
    "word",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]?")
ANSWER_CHOICE_PATTERN = re.compile(r"(?:^|\s)[A-D][\).]\s+\S", re.IGNORECASE)
SCORE_PATTERN = re.compile(r"\b(?:score|points?|percent|accuracy|correct|incorrect)\s*[:=]?\s*\d", re.IGNORECASE)


@dataclass(frozen=True)
class TextHumanizationResult:
    original_text: str
    text: str
    applied: bool
    protected: bool
    reason: str


@dataclass(frozen=True)
class DeliveryResult:
    original_text: str
    text: str
    applied: bool
    safe_chunking_applied: bool


def text_humanizer_enabled(request_value: Optional[bool] = None) -> bool:
    if request_value is not None:
        return request_value and env_bool("TTS_TEXT_HUMANIZER_ENABLED", True)

    return env_bool("TTS_TEXT_HUMANIZER_ENABLED", True)


def variation_enabled() -> bool:
    return env_bool("TTS_TEXT_HUMANIZER_VARIATION_ENABLED", True)


def delivery_control_enabled(request_value: Optional[bool] = None) -> bool:
    if request_value is not None:
        return request_value and env_bool("TTS_DELIVERY_CONTROL_ENABLED", True)

    return env_bool("TTS_DELIVERY_CONTROL_ENABLED", True)


def safe_chunking_enabled() -> bool:
    return env_bool("TTS_SAFE_CHUNKING_ENABLED", True)


def min_friendly_tokens() -> int:
    return int(env_float("TTS_MIN_FRIENDLY_TOKENS", 12))


def max_coaching_sentences() -> int:
    return int(env_float("TTS_MAX_COACHING_SENTENCES", 3))


def token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text or ""))


def canonical_key(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"<[^>]*>", " ", lowered)
    lowered = re.sub(r"[^a-z0-9']+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def sentence_count(text: str) -> int:
    return len([part for part in SENTENCE_PATTERN.findall(text or "") if part.strip()])


def _stable_variant(agent: str, text: str, variants: Sequence[str]) -> str:
    if not variants:
        return text
    if not variation_enabled() or len(variants) == 1:
        return variants[0]

    digest = hashlib.sha256(f"{agent}|{canonical_key(text)}".encode("utf-8")).hexdigest()
    return variants[int(digest[:8], 16) % len(variants)]


def _agent_variants(agent: str, variants: Mapping[str, Sequence[str]], fallback: str) -> Sequence[str]:
    return variants.get(agent) or variants.get("default") or (fallback,)


SHORT_LINE_VARIANTS: dict[str, dict[str, Sequence[str]]] = {
    "good job": {
        "miss_ciel": (
            "Good job. You said that clearly.",
            "Good job. That sounded clear and careful.",
        ),
        "miss_vivian": (
            "Good job. You answered that clearly.",
            "Good job. Nice clear answer.",
        ),
        "miss_estelle": (
            "Good job. That was clear work.",
            "Good job. You handled that well.",
        ),
    },
    "try again": {
        "miss_ciel": (
            "That's okay. Let's try that one more time.",
            "That is okay. Take your time, and let's try it again.",
        ),
        "miss_vivian": (
            "That's okay. Try that one more time when you're ready.",
            "Not quite yet. Try that one more time.",
        ),
        "miss_estelle": (
            "That's okay. Let's try that again carefully.",
            "Not quite yet. Take a moment, then try again.",
        ),
    },
    "correct": {
        "default": (
            "That's correct. Nice work.",
            "That is correct. Nice work.",
        ),
    },
    "incorrect": {
        "miss_ciel": (
            "That's not quite right. Let's try it again together.",
            "Not quite yet. Let's slow it down and try again.",
        ),
        "miss_vivian": (
            "That's not quite right. Try it again when you're ready.",
            "Not quite yet. Listen once more, then try again.",
        ),
        "miss_estelle": (
            "That's not quite right. Let's look at it gently and try again.",
            "Not quite yet. We can try that again carefully.",
        ),
    },
    "listen carefully": {
        "miss_ciel": (
            "Listen carefully first, then say the word after me.",
            "Listen carefully first. Then give it a gentle try.",
        ),
        "miss_vivian": (
            "Listen carefully first, then say the sound out loud.",
            "Listen carefully first. When you're ready, say it out loud.",
        ),
        "miss_estelle": (
            "Listen carefully first, then answer when you're ready.",
            "Listen carefully first. Take your time with the answer.",
        ),
    },
    "read this": {
        "default": (
            "Take your time, then read this one out loud.",
            "Take your time. Read this one out loud when you're ready.",
        ),
    },
    "next": {
        "miss_ciel": (
            "Nice. Let's move to the next one.",
            "Nice work. Let's go to the next one.",
        ),
        "miss_vivian": (
            "Nice. Let's move to the next one.",
            "Good. Let's go to the next one.",
        ),
        "miss_estelle": (
            "Good. Let's continue to the next one.",
            "Nice. We can move to the next one.",
        ),
    },
    "submit": {
        "default": (
            "When you're ready, you can submit your answer.",
            "Submit your answer when you're ready.",
        ),
    },
    "submit your answer": {
        "default": (
            "When you're ready, you can submit your answer.",
            "Submit your answer when you're ready.",
        ),
    },
    "that was close": {
        "miss_ciel": (
            "That was close. Let's listen carefully and try again.",
            "That was close. Let's slow it down together.",
        ),
        "miss_vivian": (
            "That was close. Listen once more, then try again.",
            "That was close. Try it again when you're ready.",
        ),
        "miss_estelle": (
            "That was close. Let's review it carefully.",
            "That was close. Take a moment, then try again.",
        ),
    },
    "good try": {
        "miss_ciel": (
            "Good try. Let's keep practicing together.",
            "Good try. You're working through it carefully.",
        ),
        "miss_vivian": (
            "Good try. Keep going when you're ready.",
            "Good try. Let's keep moving carefully.",
        ),
        "miss_estelle": (
            "Good try. Let's look at the next step calmly.",
            "Good try. We can keep working carefully.",
        ),
    },
    "keep going": {
        "miss_ciel": (
            "Keep going. You're doing careful reading work.",
            "Keep going. Take your time with each sound.",
        ),
        "miss_vivian": (
            "Keep going. Stay focused on one item at a time.",
            "Keep going. You're ready for the next step.",
        ),
        "miss_estelle": (
            "Keep going. You're making steady progress.",
            "Keep going. We will take it one step at a time.",
        ),
    },
    "great effort try again": {
        "miss_ciel": (
            "Great effort. That one is worth another try.",
            "Great effort. Let's try that one more time.",
        ),
        "miss_vivian": (
            "Great effort. Try that one more time.",
            "Great effort. Listen again, then try once more.",
        ),
        "miss_estelle": (
            "Great effort. Let's try that again carefully.",
            "Great effort. Take a moment, then try again.",
        ),
    },
    "please try again": {
        "default": (
            "That's okay. Please try that one more time.",
            "That's okay. Try that one more time when you're ready.",
        ),
    },
    "say the letter out loud": {
        "miss_vivian": (
            "Listen first. When you're ready, say the letter out loud.",
            "Take your time. Say the letter out loud when you're ready.",
        ),
        "default": (
            "When you're ready, say the letter out loud.",
            "Take your time, then say the letter out loud.",
        ),
    },
    "say the sound out loud": {
        "miss_vivian": (
            "Listen carefully first. When you're ready, say the sound out loud.",
            "Take your time. Say the sound out loud when you're ready.",
        ),
        "default": (
            "When you're ready, say the sound out loud.",
            "Take your time, then say the sound out loud.",
        ),
    },
    "say the word out loud": {
        "default": (
            "Take your time, then say the word out loud.",
            "When you're ready, say the word out loud.",
        ),
    },
}


def _context_value(context: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in context and context[name] is not None:
            return context[name]
    return None


def is_known_humanizable(text: str) -> bool:
    key = canonical_key(text)
    if key in SHORT_LINE_VARIANTS:
        return True

    return any(pattern.match(text.strip()) for pattern, _ in PATTERN_REWRITES)


def looks_protected(
    text: str,
    context: Optional[Mapping[str, Any]] = None,
    protected_terms: Optional[Sequence[str]] = None,
) -> bool:
    clean = (text or "").strip()
    if not clean:
        return True

    context = context or {}
    text_kind = str(_context_value(context, "context", "text_kind", "kind") or "").strip().lower()
    if text_kind in PROTECTED_CONTEXTS:
        return True

    if ANSWER_CHOICE_PATTERN.search(clean) or SCORE_PATTERN.search(clean):
        return True

    if re.search(r"\b(?:transcript|raw transcript|asr|debug|system)\b", clean, re.IGNORECASE):
        return True

    stripped = clean.strip(" .!?\"'")
    tokens = TOKEN_PATTERN.findall(stripped)
    if len(tokens) == 1 and len(tokens[0]) == 1:
        return True

    if len(tokens) == 1 and not is_known_humanizable(clean):
        return True

    if 2 <= len(tokens) <= 7 and not is_known_humanizable(clean):
        return True

    for term in protected_terms or ():
        term_clean = str(term or "").strip()
        if not term_clean:
            continue
        if clean.casefold() == term_clean.casefold():
            return True

    return False


def _rewrite_good_job_target(match: re.Match[str], agent: str, original: str) -> str:
    target = match.group("target").strip()
    variants = {
        "miss_ciel": (
            f"Good job. You read {target} clearly.",
            f"Good job. You said {target} clearly.",
        ),
        "miss_vivian": (
            f"Good job. You read {target} clearly.",
            "Good job. That answer was clear.",
        ),
        "miss_estelle": (
            f"Good job. You read {target} clearly.",
            "Good job. That was clear work.",
        ),
    }
    return _stable_variant(agent, original, _agent_variants(agent, variants, original))


def _rewrite_listen_target(match: re.Match[str], agent: str, original: str) -> str:
    target = match.group("target").strip()
    variants = {
        "miss_ciel": (
            f"Listen carefully first. This is how we say it: {target}.",
            f"Listen carefully first, then say it with me: {target}.",
        ),
        "miss_vivian": (
            f"Listen carefully first. This is how we say it: {target}.",
            f"Listen first, then say it out loud: {target}.",
        ),
        "miss_estelle": (
            f"Listen carefully first. This is how we say it: {target}.",
        ),
    }
    return _stable_variant(agent, original, _agent_variants(agent, variants, original))


def _rewrite_lets_practice(match: re.Match[str], agent: str, original: str) -> str:
    target = match.group("target").strip()
    variants = {
        "miss_ciel": (
            f"Let's practice {target} together. Listen carefully first, then try it after me. {target}.",
            f"Let's practice {target}. Listen first, then say it with me. {target}.",
        ),
        "miss_vivian": (
            f"Let's practice {target}. Listen carefully first, then repeat it. {target}.",
        ),
        "miss_estelle": (
            f"Let's review {target} carefully. Listen first, then try it. {target}.",
        ),
    }
    return _stable_variant(agent, original, _agent_variants(agent, variants, original))


PATTERN_REWRITES: tuple[tuple[re.Pattern[str], Any], ...] = (
    (
        re.compile(r"^Good job\. You read (?P<target>.+?) correctly\.$", re.IGNORECASE),
        _rewrite_good_job_target,
    ),
    (
        re.compile(r"^Listen carefully\. This is how we say it:\s*(?P<target>.+?)\.?$", re.IGNORECASE),
        _rewrite_listen_target,
    ),
    (
        re.compile(
            r"^Let'?s practice (?P<target>.+?)\. Listen carefully\. (?P=target)\.?$",
            re.IGNORECASE,
        ),
        _rewrite_lets_practice,
    ),
)

SAFE_EXACT_REWRITES: dict[str, dict[str, Sequence[str]]] = {
    "i could not hear that clearly please try again with your clear reading voice": {
        "miss_ciel": (
            "That's okay. I could not hear it clearly yet, so let's try that one more time with your clear reading voice.",
        ),
        "default": (
            "That's okay. I could not hear it clearly yet, so please try that one more time.",
        ),
    },
    "read the prompt then record your voice i will help you practice": {
        "miss_ciel": (
            "Take your time with the prompt, then record your voice. I'll be here to help you practice.",
        ),
        "default": (
            "Take your time with the prompt, then record your voice when you're ready.",
        ),
    },
    "take your time and read clearly": {
        "default": (
            "Take your time, and read it out loud clearly.",
        ),
    },
    "take your time let us keep the sentence moving": {
        "miss_ciel": (
            "Take your time. Let's keep the sentence moving smoothly.",
        ),
        "default": (
            "Take your time. Keep the sentence moving smoothly.",
        ),
    },
    "listen once more and read clearly": {
        "default": (
            "Listen once more, then read it clearly when you're ready.",
        ),
    },
    "try the word again": {
        "miss_ciel": (
            "That's okay. Try the word one more time.",
        ),
        "default": (
            "Try the word one more time when you're ready.",
        ),
    },
    "read the sentence again with a smooth voice": {
        "miss_ciel": (
            "Take a moment, then read the sentence again with a smooth voice.",
        ),
        "default": (
            "Read the sentence again with a smooth voice when you're ready.",
        ),
    },
}


def _state_based_rewrite(agent: str, text: str, context: Mapping[str, Any]) -> Optional[str]:
    outcome = str(_context_value(context, "outcome", "state") or "").strip().lower()
    attempt = _context_value(context, "attempt", "attempt_count")
    try:
        attempt_number = int(attempt) if attempt is not None else None
    except (TypeError, ValueError):
        attempt_number = None

    key = canonical_key(text)
    if outcome in {"correct", "success"} and key in {"correct", "good job"}:
        variants = {
            "miss_ciel": (
                "Nice work. You said that clearly.",
                "Good job. You read that clearly.",
            ),
            "miss_vivian": (
                "That's correct. Nice clear answer.",
            ),
            "miss_estelle": (
                "That's correct. You handled that well.",
            ),
        }
        return _stable_variant(agent, text, _agent_variants(agent, variants, text))

    if outcome in {"retry", "incorrect", "wrong"} and key in {"try again", "incorrect"}:
        if attempt_number is not None and attempt_number >= 2:
            variants = {
                "miss_ciel": (
                    "That's okay. This one is a little tricky, so let's slow it down together.",
                ),
                "miss_vivian": (
                    "That's okay. Listen once more, then try it again carefully.",
                ),
                "miss_estelle": (
                    "That's okay. Take a moment, then try it again carefully.",
                ),
            }
            return _stable_variant(agent, text, _agent_variants(agent, variants, text))

        variants = {
            "miss_ciel": (
                "That was close. Let's listen carefully and try again.",
            ),
            "miss_vivian": (
                "That was close. Listen again, then try once more.",
            ),
            "miss_estelle": (
                "That was close. Let's try that again carefully.",
            ),
        }
        return _stable_variant(agent, text, _agent_variants(agent, variants, text))

    if outcome in {"module_success", "module_complete"}:
        variants = {
            "miss_ciel": (
                "Great job. You're getting more confident with this.",
            ),
            "miss_vivian": (
                "Great job. You finished that part carefully.",
            ),
            "miss_estelle": (
                "Great job. You completed this part well.",
            ),
        }
        return _stable_variant(agent, text, _agent_variants(agent, variants, text))

    return None


def humanize_text(
    agent: str,
    text: str,
    profile: AgentVoiceProfile,
    context: Optional[Mapping[str, Any]] = None,
    protected_terms: Optional[Sequence[str]] = None,
    request_enabled: Optional[bool] = None,
) -> TextHumanizationResult:
    original = text
    context = context or {}

    if not text_humanizer_enabled(request_enabled):
        return TextHumanizationResult(original, text, False, False, "disabled")

    if looks_protected(text, context, protected_terms):
        return TextHumanizationResult(original, text, False, True, "protected")

    state_rewrite = _state_based_rewrite(agent, text, context)
    if state_rewrite and state_rewrite != text:
        return TextHumanizationResult(original, state_rewrite, True, False, "state")

    key = canonical_key(text)
    if key in SHORT_LINE_VARIANTS:
        variants = _agent_variants(agent, SHORT_LINE_VARIANTS[key], text)
        return TextHumanizationResult(original, _stable_variant(agent, text, variants), True, False, "short_line")

    if key in SAFE_EXACT_REWRITES:
        variants = _agent_variants(agent, SAFE_EXACT_REWRITES[key], text)
        return TextHumanizationResult(original, _stable_variant(agent, text, variants), True, False, "safe_exact")

    for pattern, rewrite in PATTERN_REWRITES:
        match = pattern.match(text.strip())
        if match:
            rewritten = rewrite(match, agent, text)
            return TextHumanizationResult(original, rewritten, rewritten != text, False, "pattern")

    if token_count(text) < min_friendly_tokens() and is_known_humanizable(text):
        return TextHumanizationResult(original, text, False, False, "already_safe")

    return TextHumanizationResult(original, text, False, False, "no_match")


def _ensure_sentence_punctuation(text: str) -> str:
    clean = text.strip()
    if clean and clean[-1] not in ".!?":
        return clean + "."

    return clean


def _normalize_delivery_punctuation(text: str) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    clean = clean.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    clean = re.sub(r"\bWhen you're ready(?!,)\b", "When you're ready,", clean)
    clean = re.sub(r"\bListen carefully first then\b", "Listen carefully first, then", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bTake your time then\b", "Take your time, then", clean, flags=re.IGNORECASE)
    clean = clean.replace("Let us ", "Let's ")
    return _ensure_sentence_punctuation(clean)


def _limit_coaching_sentences(text: str) -> str:
    max_sentences = max(1, max_coaching_sentences())
    parts = [part.strip() for part in SENTENCE_PATTERN.findall(text) if part.strip()]
    if len(parts) <= max_sentences:
        return text

    return " ".join(parts[:max_sentences])


def apply_delivery_direction(
    agent: str,
    text: str,
    profile: AgentVoiceProfile,
    context: Optional[Mapping[str, Any]] = None,
    request_enabled: Optional[bool] = None,
) -> DeliveryResult:
    original = text
    if not delivery_control_enabled(request_enabled):
        return DeliveryResult(original, text, False, False)

    if looks_protected(text, context):
        return DeliveryResult(original, text, False, False)

    safe_chunking_applied = False
    prepared = text

    if safe_chunking_enabled():
        chunked = re.sub(r"\s*(?:\|\||\|)\s*", ". ", prepared)
        safe_chunking_applied = chunked != prepared
        prepared = chunked

    prepared = _normalize_delivery_punctuation(prepared)
    if profile.agent in {"miss_ciel", "miss_estelle"}:
        prepared = _limit_coaching_sentences(prepared)

    return DeliveryResult(original, prepared, prepared != original, safe_chunking_applied)
