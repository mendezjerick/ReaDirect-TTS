# ReaDirect TTS Humanization

This revamp exists because the original agent speech sounded too clean, short, and command-like. Kokoro voice choice matters, but the bigger fix is the full generation pipeline: profile selection, warmer writing, delivery preparation, and light audio post-processing.

## Fixed Voice Mapping

- Miss Ciel: `af_heart`, default speed `0.94`, allowed range `0.92` to `0.96`
- Miss Vivian: `af_bella`, default speed `0.97`, allowed range `0.95` to `1.00`
- Miss Estelle: `bf_isabella`, default speed `0.93`, allowed range `0.90` to `0.95`

Miss Ciel must always remain `af_heart`. Runtime request payloads are not allowed to change her voice. Vivian is fixed to `af_bella`. Estelle must use an Isabella Kokoro voice; the expected ID is `bf_isabella`, with only an Isabella-named local override accepted.

## Line Ownership

The main ReaDirect Laravel/Vue repo owns learner-facing line context and most runtime base text. ReaDirect-TTS owns synthesis, fixed voice profiles, curated fallback selection, reference selection, IndexTTS2, and caching.

See `../READIRECT_AGENT_LINE_SOURCE_AUDIT.md` for the cross-repo source audit.

## Why Line Revamping Matters

Very short lines such as "Good job.", "Try again.", or "Correct." often sound flat because Kokoro receives too little context for natural rhythm. ReaDirect now treats curated base lines as the main fix. The app-owned line should already be natural enough for voice synthesis before it reaches the TTS engine.

The Kokoro-friendly goldilocks zone in this repo means:

- Avoid ultra-short robotic feedback when safe.
- Prefer 1 to 3 natural spoken sentences for coaching and instructions.
- Keep assessment instructions clear and brief.
- Do not over-explain simple tasks.
- Never rewrite protected learner, target, scoring, or content-bank text.

## Four-Layer Pipeline

1. Agent Voice Profile Layer
   - Centralized in `agent_voice_profiles.py`.
   - Stores fixed voice IDs, speed defaults, speed ranges, role, personality, emotion, and delivery style.
   - Clamps runtime speed requests to each agent range.

2. Curated Prompt Layer
   - Implemented in `curated_agent_lines.py`.
   - Selects a curated prompt only from an explicit `line_key` or an exact known legacy narration line.
   - Targets natural 6 to 9 second spoken prompts for regular agent narration.
   - Does not pad arbitrary short text.

3. Legacy Text Humanizer Layer
   - Implemented in `tts_humanizer.py`.
   - Disabled by default with `TTS_AUTO_PROMPT_EXTENSION_ENABLED=false`.
   - Kept only for debugging and comparison against the previous generic extension behavior.

4. Delivery Direction Layer
   - Implemented in `tts_humanizer.py`.
   - Adds safe punctuation, sentence endings, comma placement, and simple chunk joining.
   - Avoids dramatic ellipses and skips protected content.

5. Audio Post-Processing / Humanizer Layer
   - Implemented in `audio_humanizer.py`.
   - Applies optional peak normalization, tiny fade-in/fade-out, and controlled sentence pauses.
   - Breath insertion is disabled by default. If breath WAV files are added later under `breaths/`, they are only considered for longer coaching/explanation lines.

## Protected Text Rules

The humanizer must not alter:

- target letters, words, or phrases
- expected spoken answers
- learner transcripts or ASR result text
- scoring labels and score values
- answer choices
- CSV content-bank items
- assessment passages unless explicitly agent narration
- reading comprehension question content
- debug or system messages

If the service cannot tell whether short content is narration or target content, it does not rewrite it. Single letters, single words, answer-choice patterns, score-like text, transcript/debug text, and unknown short phrases are treated as protected.

## Config Flags

```env
TTS_AGENT_PROFILES_ENABLED=true
TTS_AGENT_SPEED_CIEL=0.94
TTS_AGENT_SPEED_VIVIAN=0.97
TTS_AGENT_SPEED_ESTELLE=0.93

TTS_AUTO_PROMPT_EXTENSION_ENABLED=false
TTS_CURATED_PROMPTS_ENABLED=true
TTS_CURATED_PROMPT_TARGET_SECONDS=7.0
TTS_CURATED_PROMPT_MIN_SECONDS=6.0
TTS_CURATED_PROMPT_MAX_SECONDS=9.0

TTS_TEXT_HUMANIZER_ENABLED=true
TTS_TEXT_HUMANIZER_MODE=friendly
TTS_TEXT_HUMANIZER_VARIATION_ENABLED=true
TTS_TEXT_HUMANIZER_LOGGING=true

TTS_DELIVERY_CONTROL_ENABLED=true
TTS_SAFE_CHUNKING_ENABLED=true
TTS_MIN_FRIENDLY_TOKENS=12
TTS_MAX_COACHING_SENTENCES=3

TTS_HUMANIZER_ENABLED=true
TTS_AUDIO_NORMALIZE_ENABLED=true
TTS_AUDIO_FADE_ENABLED=true
TTS_PAUSE_CONTROL_ENABLED=true
TTS_BREATHS_ENABLED=false
TTS_BREATHS_VOLUME=0.08
TTS_BREATHS_MIN_TEXT_LENGTH=80

TTS_DEBUG_LOGGING=true
```

To disable all spoken humanization while keeping the service available, set:

```env
TTS_CURATED_PROMPTS_ENABLED=false
TTS_TEXT_HUMANIZER_ENABLED=false
TTS_DELIVERY_CONTROL_ENABLED=false
TTS_HUMANIZER_ENABLED=false
TTS_PAUSE_CONTROL_ENABLED=false
```

## Curated 7s Comparison Test

Start the service:

```powershell
cd "C:\Users\Lost\Documents\holder-ReaDirect\ReaDirect-TTS"
.\.venv\Scripts\Activate.ps1
python tts_service.py
```

Generate curated comparison files:

```powershell
python scripts\generate_curated_7s_comparisons.py --force
```

The script creates raw Kokoro, old humanizer comparison, curated Kokoro, and IndexTTS2 expressive attempts for the required agent/intent coverage. Outputs and `GENERATION_REPORT.md` are saved under:

```text
storage/tts/cache/comparisons/curated_7s/
```

Suggested listening lines:

- "Good job."
- "Try again."
- "Listen carefully."
- "That was close."
- "Take your time and read the word out loud."
- "You did well today. Let's look at your result together."

## Tuning Speed And Pacing

Tune speed with the `TTS_AGENT_SPEED_*` variables. Values outside the allowed profile range are clamped. For warmer coaching, adjust Ciel inside `0.92` to `0.96`. For assessment clarity, keep Vivian inside `0.95` to `1.00`. For result explanations, keep Estelle inside `0.90` to `0.95`.

Pacing is mainly controlled by curated line writing, sentence punctuation, safe sentence chunking, reference duration weighting, and post-generation sentence pauses. Avoid adding ellipses everywhere; that usually sounds artificial.

## Adding Breath Samples Later

Breathing is intentionally off by default. To test it later:

1. Add quiet breath WAV files to `ReaDirect-TTS/breaths/`.
2. Use the same sample rate as Kokoro output: `24000`.
3. Set `TTS_BREATHS_ENABLED=true`.
4. Keep `TTS_BREATHS_VOLUME` low, such as `0.08`.

Breaths are never the main fix. Better writing, pacing, punctuation, and profile speed are the primary humanization controls.
