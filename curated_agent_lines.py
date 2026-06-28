from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import tts_config


TOKEN_PATTERN = re.compile(r"[^a-z0-9']+")


@dataclass(frozen=True)
class CuratedLine:
    agent: str
    intent: str
    line_key: str
    text: str
    target_duration_seconds: float
    min_duration_seconds: float
    max_duration_seconds: float
    protected: bool
    context: str
    legacy_texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class CuratedPromptResult:
    original_text: str
    text: str
    applied: bool
    reason: str
    line: Optional[CuratedLine] = None


def _canonical(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"<[^>]*>", " ", lowered)
    lowered = TOKEN_PATTERN.sub(" ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _line(
    agent: str,
    intent: str,
    line_key: str,
    text: str,
    context: str,
    legacy_texts: Iterable[str] = (),
) -> CuratedLine:
    return CuratedLine(
        agent=agent,
        intent=intent,
        line_key=line_key,
        text=text,
        target_duration_seconds=tts_config.curated_prompt_target_seconds(),
        min_duration_seconds=tts_config.curated_prompt_min_seconds(),
        max_duration_seconds=tts_config.curated_prompt_max_seconds(),
        protected=False,
        context=context,
        legacy_texts=tuple(legacy_texts),
    )


CURATED_LINES: tuple[CuratedLine, ...] = (
    _line(
        "miss_ciel",
        "intro",
        "ciel_intro_read_together",
        "Hi, I'm Miss Ciel. I'll read with you today, and we'll take each word slowly together.",
        "Ciel introduction before reading practice",
        (
            "Hi! I am Miss Ciel. I will help you practice reading.",
            "Hi! I am Miss Ciel. I will help you practice reading. Mistakes are okay. I am here to guide you.",
            "Hi, I'm Miss Ciel. I'll read with you today.",
        ),
    ),
    _line(
        "miss_ciel",
        "friendly_encouragement",
        "read_slowly_together",
        "Take your time, then read this one out loud. I'll stay with you, and we can go slowly together.",
        "module coaching before learner reads a word or sentence",
        (
            "Read the prompt, then record your voice. I will help you practice.",
            "Take your time and read clearly.",
            "Take your time, then read this one out loud.",
        ),
    ),
    _line(
        "miss_ciel",
        "friendly_encouragement",
        "breathe_then_read",
        "You can do it. Look at the word first, breathe softly, and then say it when you're ready.",
        "module coaching before learner records",
        (
            "Try your best and speak clearly.",
            "You can do it. Let's try it slowly.",
        ),
    ),
    _line(
        "miss_ciel",
        "friendly_encouragement",
        "ready_read_together",
        "Ready? Let's read this one together. Go slowly, and just try your best.",
        "light module guidance",
        (
            "Your practice path is ready. We will work one step at a time.",
            "Ready? Let's read this one together.",
        ),
    ),
    _line(
        "miss_ciel",
        "gentle_reassurance",
        "try_one_more_time",
        "That's okay, let's try that one more time. This one can be tricky, but we can slow it down together.",
        "retry after an incorrect or unclear attempt",
        (
            "Try again.",
            "Good try!",
            "Try this same item again.",
            "Good effort! Let us try again when you are ready.",
            "That's okay, let's try that one more time.",
        ),
    ),
    _line(
        "miss_ciel",
        "gentle_reassurance",
        "slow_down_together",
        "No worries, you were close. Let's listen carefully, then say it again a little slower.",
        "near-miss feedback",
        (
            "Almost there. Try again slowly and clearly.",
            "Good effort. Let us try one more time.",
            "No worries, we can slow it down together.",
        ),
    ),
    _line(
        "miss_ciel",
        "gentle_reassurance",
        "step_by_step_retry",
        "That was close. We'll try it again together, and this time we'll take it step by step.",
        "near-miss retry",
        (
            "Good try! That was very close. Let us fix one small sound.",
            "That was close. Let's say it again.",
        ),
    ),
    _line(
        "miss_ciel",
        "happy_praise",
        "clear_confident_praise",
        "Nice work! You said that clearly, and I can hear that you're getting more confident.",
        "positive feedback after a correct answer",
        (
            "Good job.",
            "Good job. You read it correctly.",
            "Nice reading. You got it.",
            "Nice work! You said that clearly.",
        ),
    ),
    _line(
        "miss_ciel",
        "happy_praise",
        "got_that_one",
        "Great job! You got that one, and you read it with a nice clear voice.",
        "positive feedback after a correct answer",
        (
            "Great job! You got that one.",
            "Great job. That is correct.",
            "Great reading. You are getting stronger.",
            "Great streak. Keep going.",
        ),
    ),
    _line(
        "miss_ciel",
        "happy_praise",
        "keep_going_clear",
        "Good job, that was clear. Let's keep going while you're doing so well.",
        "positive feedback before moving forward",
        (
            "That is correct. Go to the next one.",
            "Good job, that was clear.",
        ),
    ),
    _line(
        "miss_ciel",
        "focused_instruction",
        "listen_then_say_word",
        "Listen carefully first, then say the word after me. Take your time and speak clearly when you're ready.",
        "instruction before hear-and-repeat practice",
        (
            "Listen carefully. This is how we say it.",
            "Listen first, then say it after me.",
            "Listen carefully first, then say the word after me.",
        ),
    ),
    _line(
        "miss_ciel",
        "focused_instruction",
        "look_listen_read",
        "Look at the word, listen to the sound, and then read it out loud in your own voice.",
        "instruction before a displayed word",
        (
            "Read this.",
            "Look at the word, then read it out loud.",
        ),
    ),
    _line(
        "miss_ciel",
        "focused_instruction",
        "say_sound_clearly",
        "When you're ready, say the sound clearly. We'll go slowly, so you don't need to rush.",
        "instruction before sound practice",
        (
            "When you're ready, say the sound clearly.",
            "Listen carefully, then say the sound in your own voice.",
        ),
    ),
    _line(
        "miss_ciel",
        "playful_friend",
        "go_slowly_try",
        "Ready? Let's go slowly and give this one a try. I'll be right here with you.",
        "friendly module transition",
        (
            "Ready? Let's go slowly.",
            "Let us keep practicing.",
            "I am here to help you read.",
        ),
    ),
    _line(
        "miss_ciel",
        "playful_friend",
        "try_together_smile",
        "Let's try this one together. Look closely, smile a little, and say it when you're ready.",
        "light module guidance",
        (
            "Let's give this one a try.",
            "Let us answer this first.",
        ),
    ),
    _line(
        "miss_ciel",
        "playful_friend",
        "next_one_clear",
        "Nice, let's move to the next one. Keep your voice clear and take your time.",
        "transition after a completed item",
        (
            "Nice, let's move to the next one.",
            "Good try. Go to the next one.",
        ),
    ),
    _line(
        "miss_vivian",
        "intro",
        "vivian_intro_assessment",
        "Hi, I'm Miss Vivian. I'll guide you through this activity, so listen carefully and take your time.",
        "assessment introduction",
        (
            "Hello! I am Miss Vivian. I will guide you through your reading assessment. Try your best and answer one step at a time.",
            "We will do a short reading check together. I will guide each step - just try your best!",
            "This is your final reading check. Do your best, one step at a time.",
        ),
    ),
    _line(
        "miss_vivian",
        "focused_instruction",
        "listen_then_say_sound",
        "Listen carefully first, then say the sound out loud. When you're ready, use a clear voice.",
        "assessment sound instruction",
        (
            "Listen carefully first, then say the sound out loud.",
            "Say this letter clearly for your final check.",
        ),
    ),
    _line(
        "miss_vivian",
        "focused_instruction",
        "look_item_answer",
        "Look at the item on the screen, listen to the instruction, and answer when you are ready.",
        "assessment item instruction",
        (
            "Let us answer this first.",
            "Hold the orange button to record your answer first.",
            "Hold the orange button to record the highlighted word first.",
        ),
    ),
    _line(
        "miss_vivian",
        "focused_instruction",
        "listen_choose_or_say",
        "Take your time before you answer. Listen first, then choose or say the response clearly.",
        "assessment choice or spoken response",
        (
            "Choose the best answer based on the story you read.",
            "Choose one story for your final reading passage.",
            "Which story do you want to read? Pick the one that sounds most interesting to you!",
        ),
    ),
    _line(
        "miss_vivian",
        "friendly_encouragement",
        "stay_focused_ready",
        "You're doing fine. Just stay focused, listen carefully, and answer when you feel ready.",
        "assessment encouragement",
        (
            "Thank you. Let us continue.",
            "I heard your answer. Let us keep going.",
            "Listen to your answer. If you are happy with your answer, click Submit.",
        ),
    ),
    _line(
        "miss_vivian",
        "friendly_encouragement",
        "keep_going_each_item",
        "Keep going. Take your time with each item, and remember to listen before you answer.",
        "assessment transition",
        (
            "Good effort. Let us go to the next one.",
            "Thank you. Let us continue.",
        ),
    ),
    _line(
        "miss_vivian",
        "friendly_encouragement",
        "one_item_at_time",
        "You can do this. Stay calm, look carefully, and answer one item at a time.",
        "assessment start or transition",
        (
            "Try your best and answer one step at a time.",
            "Do your best, one step at a time.",
        ),
    ),
    _line(
        "miss_vivian",
        "gentle_reassurance",
        "next_item_clear_voice",
        "That's okay. Just listen carefully and try the next item with a calm and clear voice.",
        "assessment retry or neutral continuation",
        (
            "We could not check these letters yet. Please review them and try again.",
            "We could not check these words yet. Please review them and try again.",
            "We could not check these sentences yet. Please review them and try again.",
        ),
    ),
    _line(
        "miss_vivian",
        "gentle_reassurance",
        "continue_calmly",
        "No worries. Stay focused, take your time, and let's continue with the next one.",
        "assessment continuation after a miss",
        (
            "Almost there. Finish each letter before checking your answer.",
            "Almost there. Finish each sentence before checking your words.",
        ),
    ),
    _line(
        "miss_vivian",
        "gentle_reassurance",
        "tricky_keep_going",
        "That one was a bit tricky. Keep going, and remember to listen carefully first.",
        "assessment reassurance",
    ),
    _line(
        "miss_vivian",
        "happy_praise",
        "answered_clearly",
        "Nice work. You answered that clearly, so let's keep moving through the activity.",
        "assessment positive feedback",
        (
            "Nice, let's move to the next one.",
            "Nice work. You answered that clearly.",
        ),
    ),
    _line(
        "miss_vivian",
        "happy_praise",
        "clear_voice_next",
        "Good job. Stay focused and keep using your clear voice for the next item.",
        "assessment positive transition",
        (
            "Good job. Stay focused and keep using your clear voice.",
        ),
    ),
    _line(
        "miss_vivian",
        "happy_praise",
        "one_step_time",
        "Great work. You're doing well, and we'll continue one step at a time.",
        "assessment praise",
        (
            "Great work. You're doing well.",
        ),
    ),
    _line(
        "miss_estelle",
        "intro",
        "estelle_intro_results",
        "Hi, I'm Miss Estelle. I'll help you look at your results in a calm and simple way.",
        "results introduction",
        (
            "Hello! I am Miss Estelle. I will help explain your results so you know what to do next.",
            "Let's look at your result together.",
        ),
    ),
    _line(
        "miss_estelle",
        "calm_evaluation",
        "look_result_together",
        "Let's look at your result together. This will help us understand what you already do well and what we can practice next.",
        "result explanation",
        (
            "Your next step is ready.",
            "Here is how your reading changed.",
        ),
    ),
    _line(
        "miss_estelle",
        "calm_evaluation",
        "did_well_tricky_items",
        "You did well in this part, and there are still a few tricky items that we can keep practicing.",
        "result explanation",
        (
            "Good effort. We will practice this module again.",
            "You did well in this part, and we'll keep practicing the tricky ones.",
        ),
    ),
    _line(
        "miss_estelle",
        "calm_evaluation",
        "result_guides_support",
        "This result is here to help us. It shows where you are improving and where you may need more support.",
        "result explanation",
    ),
    _line(
        "miss_estelle",
        "gentle_reassurance",
        "result_not_failed",
        "That's okay. This result does not mean you failed; it simply helps us know what to practice next.",
        "reassuring result explanation",
        (
            "That's okay, this helps us know what to practice next.",
        ),
    ),
    _line(
        "miss_estelle",
        "gentle_reassurance",
        "parts_are_tricky",
        "No worries. Some parts can be tricky, and we can use this result to help you improve.",
        "reassuring result explanation",
    ),
    _line(
        "miss_estelle",
        "gentle_reassurance",
        "guide_next_practice",
        "It's okay if some items were hard. We'll use this result to guide your next practice.",
        "reassuring result explanation",
    ),
    _line(
        "miss_estelle",
        "happy_praise",
        "effort_progress",
        "Great job. You showed good effort, and your result shows that you are making progress.",
        "results praise",
    ),
    _line(
        "miss_estelle",
        "happy_praise",
        "proud_effort",
        "Nice work. You did well in this part, and you should feel proud of your effort.",
        "results praise",
    ),
    _line(
        "miss_estelle",
        "happy_praise",
        "stayed_focused",
        "Good job. You stayed focused, and that helped you complete this activity.",
        "results praise",
        (
            "Great job finishing your final assessment. Here is how your reading changed.",
        ),
    ),
    _line(
        "miss_estelle",
        "focused_instruction",
        "look_result_carefully",
        "Please look at the result carefully. I'll explain it in a simple way so it is easy to understand.",
        "results instruction",
    ),
    _line(
        "miss_estelle",
        "focused_instruction",
        "go_through_slowly",
        "Let's go through this part slowly. I'll help you understand what the result means.",
        "results instruction",
    ),
    _line(
        "miss_estelle",
        "focused_instruction",
        "score_then_practice",
        "Take a moment to look at your score, then we'll talk about what you can practice next.",
        "results instruction",
    ),
)


def lines_for(agent: Optional[str] = None, intent: Optional[str] = None) -> list[CuratedLine]:
    return [
        line
        for line in CURATED_LINES
        if (agent is None or line.agent == agent)
        and (intent is None or line.intent == intent)
    ]


def resolve_curated_prompt(
    agent: str,
    text: str,
    intent: Optional[str] = None,
    line_key: Optional[str] = None,
    protected: bool = False,
) -> CuratedPromptResult:
    if not tts_config.curated_prompts_enabled():
        return CuratedPromptResult(text, text, False, "curated_prompts_disabled")
    if protected:
        return CuratedPromptResult(text, text, False, "protected")

    agent_lines = [line for line in CURATED_LINES if line.agent == agent]
    if intent:
        agent_lines = [line for line in agent_lines if line.intent == intent]

    clean_line_key = (line_key or "").strip()
    if clean_line_key:
        for line in agent_lines:
            if line.line_key == clean_line_key:
                return CuratedPromptResult(text, line.text, line.text != text, f"line_key:{line.line_key}", line)
        return CuratedPromptResult(text, text, False, "line_key_not_found")

    clean_text = _canonical(text)
    for line in agent_lines:
        if clean_text == _canonical(line.text):
            return CuratedPromptResult(text, line.text, False, f"already_curated:{line.line_key}", line)
        for legacy in line.legacy_texts:
            if clean_text == _canonical(legacy):
                return CuratedPromptResult(text, line.text, line.text != text, f"legacy_exact:{line.line_key}", line)

    return CuratedPromptResult(text, text, False, "no_curated_match")
