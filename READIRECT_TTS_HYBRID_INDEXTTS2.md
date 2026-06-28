# ReaDirect Hybrid Kokoro + IndexTTS2 TTS

This phase prepares the architecture for expressive TTS while keeping Kokoro as the stable default.

## Why Hybrid TTS

Kokoro gives ReaDirect consistent agent identities, but short instructional lines can sound monotone when many sentence endings share the same period-ending cadence. Repeated falling intonation makes feedback sound AI-like even when the voice ID is good.

The hybrid model separates the three responsibilities:

- ReaDirect provides what line should be spoken.
- Kokoro provides who is speaking.
- IndexTTS2 or an IndexTTS2-compatible expressive adapter provides how the line is delivered.

## Fixed Voice Identities

These mappings are fixed and must not be changed by request payloads:

- Miss Ciel: Kokoro `af_heart`
- Miss Vivian: Kokoro `af_bella`
- Miss Estelle: Kokoro Isabella, expected `bf_isabella`

Ciel must always remain `af_heart`. Vivian must remain `af_bella`. Estelle must remain an Isabella Kokoro voice.

## Runtime Pipeline

1. Laravel sends the existing `/synthesize` request, with optional `intent` and `line_key`.
2. ReaDirect-TTS sanitizes text and protects assessment/ASR content.
3. A curated prompt is selected from an explicit line key or exact known legacy narration line when available.
4. A deterministic prosody intent is selected.
5. Kokoro is used directly, or expressive mode is attempted if enabled.
6. Expressive mode selects a style reference from `storage/tts/references/manifest.json`.
7. Kokoro neutral timbre references are generated only when expressive generation can use them.
8. Output is cached under `storage/tts/cache/`.
9. The service returns `audio/wav` bytes to Laravel as before.

## Expressive References

Reference files are style references only. They are for cadence, pacing, emotion, and delivery. They must not replace the fixed ReaDirect agent voices.

Place future files here:

```text
storage/tts/references/
  manifest.example.json
  manifest.json
  ciel/
    gentle_reassurance/
    friendly_encouragement/
    happy_praise/
    focused_instruction/
    playful_friend/
  vivian/
    gentle_reassurance/
    friendly_encouragement/
    happy_praise/
    focused_instruction/
  estelle/
    gentle_reassurance/
    calm_evaluation/
    happy_praise/
    focused_instruction/
```

Preferred source format is `.wav`. `.mp3` paths are accepted in the manifest when the local audio library can decode them, then normalized internally to mono 24 kHz WAV.

## Manifest Format

Copy `storage/tts/references/manifest.example.json` to `manifest.json` when references are ready. Paths are relative to `storage/tts/references`.

If `manifest.json` is missing, invalid, or empty, expressive mode logs the reason and falls back to Kokoro when fallback is enabled.

## Supported Intents

- `gentle_reassurance`
- `intro`
- `friendly_encouragement`
- `happy_praise`
- `focused_instruction`
- `calm_evaluation`
- `playful_friend`

Defaults:

- Ciel: `friendly_encouragement`
- Vivian: `focused_instruction`
- Estelle: `calm_evaluation`

The classifier is deterministic and does not use an LLM at runtime.

## Protected Text

The humanizer must not rewrite target letters, target words, expected spoken answers, answer choices, scoring labels, score values, ASR transcript text, learner transcript text, assessment passages, reading comprehension questions, debug messages, or system messages.

When the service is unsure, it leaves the text unchanged.

## Configuration

```env
TTS_ENGINE=kokoro
TTS_EXPRESSIVE_ENGINE_ENABLED=true
TTS_EXPRESSIVE_ENGINE=index_tts2
TTS_EXPRESSIVE_FALLBACK_TO_KOKORO=true
INDEXTTS2_ENABLED=true
INDEXTTS2_ADAPTER_MODULE=index_tts2_adapter
INDEXTTS2_REPO_PATH=
INDEXTTS2_MODEL_DIR=models/indextts2/checkpoints
INDEXTTS2_CONFIG_PATH=models/indextts2/checkpoints/config.yaml
INDEXTTS2_DEVICE=auto
INDEXTTS2_OUTPUT_SAMPLE_RATE=24000

TTS_AGENT_SPEED_CIEL=0.94
TTS_AGENT_SPEED_VIVIAN=0.97
TTS_AGENT_SPEED_ESTELLE=0.93

TTS_AUTO_PROMPT_EXTENSION_ENABLED=false
TTS_CURATED_PROMPTS_ENABLED=true
TTS_CURATED_PROMPT_TARGET_SECONDS=7.0
TTS_CURATED_PROMPT_MIN_SECONDS=6.0
TTS_CURATED_PROMPT_MAX_SECONDS=9.0

TTS_REFERENCE_MANIFEST_PATH=storage/tts/references/manifest.json
TTS_REFERENCE_AUDIO_ROOT=storage/tts/references
TTS_REFERENCE_WEIGHTING_ENABLED=true
TTS_REFERENCE_HIGH_PRIORITY_MIN_SECONDS=7.0
TTS_REFERENCE_MEDIUM_PRIORITY_MIN_SECONDS=4.0

TTS_EMOTION_PROMPT_INTRO="warm, friendly, welcoming, gentle"
TTS_EMOTION_PROMPT_GENTLE_REASSURANCE="gentle, warm, reassuring"
TTS_EMOTION_PROMPT_FRIENDLY_ENCOURAGEMENT="cheerful and encouraging"
TTS_EMOTION_PROMPT_HAPPY_PRAISE="bright, pleased, supportive praise"
TTS_EMOTION_PROMPT_FOCUSED_INSTRUCTION="focused, clear, patient instruction"
TTS_EMOTION_PROMPT_CALM_EVALUATION="calm and supportive"
TTS_EMOTION_PROMPT_PLAYFUL_FRIEND="light, friendly, playful reading buddy"

TTS_CACHE_ENABLED=true
TTS_CACHE_BYPASS=false
TTS_CACHE_ROOT=storage/tts/cache
TTS_COMPARISON_OUTPUT_ROOT=storage/tts/cache/comparisons
TTS_DEBUG_LOGGING=true
```

Expressive mode is optional. Kokoro remains available when IndexTTS2 is missing or disabled. The adapter module exposes:

```python
def generate(text, output_path, agent, kokoro_voice, speed, speaker_reference_path, style_reference_path, emotion_prompt, intent):
    ...
```

The adapter should write a WAV file to `output_path` or return WAV bytes.

## Kokoro Timbre References

When expressive generation is selected and available, the service can create neutral Kokoro identity samples:

```text
storage/tts/cache/kokoro/reference_voice/ciel_af_heart.wav
storage/tts/cache/kokoro/reference_voice/vivian_af_bella.wav
storage/tts/cache/kokoro/reference_voice/estelle_bf_isabella.wav
```

These are generated from short neutral lines and reused. They are not regenerated on every request unless deleted or explicitly regenerated in a later tool phase.

## Caching

Cache keys include the engine, agent, Kokoro voice ID, original text, curated/humanized text, synthesis text, intent, selected reference digest, selected reference duration, reference weighting version, speed, and cache version. Kokoro cache files are stored under:

```text
storage/tts/cache/kokoro/
```

Expressive cache files are stored under:

```text
storage/tts/cache/expressive/
```

Set `TTS_CACHE_BYPASS=true` for manual testing when you need regeneration.

## Comparison Script

Start the TTS service, then run:

```powershell
cd "C:\Users\Lost\Documents\holder-ReaDirect\ReaDirect-TTS"
python scripts\compare_hybrid_tts.py
```

Outputs are saved to:

```text
storage/tts/cache/comparisons/
```

The script creates raw Kokoro and humanized Kokoro files. Expressive files are created only when IndexTTS2 expressive mode is enabled and the adapter is available. Otherwise it logs why expressive output was skipped.

## Adding Ciel Expressive Reference Files

Ciel remains Kokoro `af_heart`. The reference files only guide cadence, emotion, pacing, intonation, sentence-ending variation, and tone. They must not replace Ciel's speaker identity.

Place loose Ciel MP3 files in:

```text
storage/tts/references/ciel/
```

Accepted loose filename patterns:

- `friendly*.mp3` or typo-tolerant `freindly*.mp3` -> `friendly_encouragement/`
- `gentle*.mp3` -> `gentle_reassurance/`
- `happy*.mp3` -> `happy_praise/`
- `instruct*.mp3` -> `focused_instruction/`
- `playful*.mp3` -> `playful_friend/`
- `intro.mp3` -> `intro/`

Run the organizer:

```powershell
cd "C:\Users\Lost\Documents\holder-ReaDirect\ReaDirect-TTS"
python scripts\organize_ciel_references.py
```

The organizer leaves original MP3 files in place and creates normalized WAV copies in the category folders, for example:

```text
storage/tts/references/ciel/friendly_encouragement/ciel_friendly_encouragement_01.wav
storage/tts/references/ciel/gentle_reassurance/ciel_gentle_reassurance_01.wav
storage/tts/references/ciel/intro/ciel_intro_01.wav
```

Conversion target:

- WAV
- Mono
- 24 kHz
- Normalized volume
- No clipping
- Obvious leading/trailing silence trimmed when safe

The script uses `ffmpeg` when available. If `ffmpeg` is missing and the local `soundfile` build cannot decode MP3, it exits with a clear conversion message. Normal Kokoro TTS does not depend on this script and continues to work.

After conversion, the organizer writes:

```text
storage/tts/references/manifest.json
```

The manifest includes only Ciel reference WAVs that actually exist. `intro` includes only `ciel_intro_01.wav`. Other Ciel intents include one to three available files and warn when fewer than three are present.

Reference selection is deterministic and duration-weighted. The service chooses among available files using a hash of agent, intent, text, and cache seed, then includes the selected reference digest, duration, and weighting version in the cache key. This gives variation without breaking cache stability.

Duration priority:

- `high`: 7.0 seconds and above, weight 5
- `medium`: 4.0 to 6.99 seconds, weight 2
- `low`: below 4.0 seconds, weight 1

Short references are still allowed. Longer references are selected more often because they usually provide stronger cadence and pacing guidance.

Generate Ciel comparison samples:

```powershell
python scripts\generate_ciel_reference_comparison.py --force
```

Outputs are saved under:

```text
storage/tts/cache/comparisons/ciel/
```

The script writes raw Kokoro and humanized Kokoro files for each safe narration line. It writes expressive files only when IndexTTS2 expressive mode is enabled and the adapter is available. When listening, check that the output still sounds like Ciel `af_heart`, with only cadence, emotion, pacing, and sentence-ending variety borrowed from the reference style.

## Configuring IndexTTS2 Adapter

The local adapter module is:

```env
INDEXTTS2_ADAPTER_MODULE=index_tts2_adapter
```

Required runtime configuration:

```env
TTS_ENGINE=kokoro
TTS_EXPRESSIVE_ENGINE_ENABLED=true
TTS_EXPRESSIVE_ENGINE=index_tts2
TTS_EXPRESSIVE_FALLBACK_TO_KOKORO=true

INDEXTTS2_ENABLED=true
INDEXTTS2_ADAPTER_MODULE=index_tts2_adapter
INDEXTTS2_REPO_PATH=
INDEXTTS2_MODEL_DIR=models/indextts2/checkpoints
INDEXTTS2_CONFIG_PATH=models/indextts2/checkpoints/config.yaml
INDEXTTS2_DEVICE=auto
INDEXTTS2_OUTPUT_SAMPLE_RATE=24000

TTS_REFERENCE_MANIFEST_PATH=storage/tts/references/manifest.json
TTS_REFERENCE_AUDIO_ROOT=storage/tts/references
TTS_CACHE_ENABLED=true
TTS_CACHE_BYPASS=false
TTS_CACHE_ROOT=storage/tts/cache
TTS_COMPARISON_OUTPUT_ROOT=storage/tts/cache/comparisons
```

`INDEXTTS2_REPO_PATH` is optional. Set it only when IndexTTS2 is present as a local folder and its `indextts` package is not installed in the virtual environment.

The adapter loads `indextts.infer_v2.IndexTTS2` once and reuses the model. It calls `infer()` with:

- target text
- Kokoro Ciel speaker reference: `storage/tts/cache/kokoro/reference_voice/ciel_af_heart.wav`
- selected Ciel expressive reference WAV
- emotion prompt, when the local API supports it

Adapter modes:

- `dual_reference`: speaker reference plus expressive reference
- `single_reference`: local API only accepted the speaker reference fields
- `text_prompt_only`: local API accepted text emotion prompt but not separate expressive audio
- `fallback_kokoro`: expressive generation did not run and Kokoro fallback was used

Common blocker meanings:

- `package_missing:indextts.infer_v2`: install IndexTTS2 in the virtual environment or set `INDEXTTS2_REPO_PATH`
- `model_dir_missing:<path>`: set `INDEXTTS2_MODEL_DIR` to the local model directory
- `config_path_missing:<path>`: set `INDEXTTS2_CONFIG_PATH` to the local IndexTTS2 config file
- `adapter_import_failed:<error>`: the adapter module or package import failed
- `model_load_failed:<error>`: IndexTTS2 imported but did not initialize
- `unsupported_api_shape:<detail>`: the local IndexTTS2 API does not expose the expected class or `infer()` call
- `style_reference_missing:<path>`: the manifest points to a missing expressive reference
- `speaker_reference_missing:<path>`: the Kokoro Ciel neutral reference was not generated or could not be read
- `device_cuda_error:<error>`: CUDA/device initialization failed; try `INDEXTTS2_DEVICE=cpu`
- `generation_failed:<error>`: IndexTTS2 started generation but failed inside inference

Run the health endpoint to inspect the current adapter state:

```text
http://127.0.0.1:8002/health
```

Run Ciel comparison generation:

```powershell
python scripts\generate_ciel_reference_comparison.py --force
```

Expressive outputs are saved to:

```text
storage/tts/cache/comparisons/ciel/
```

## Ciel Expressive Output Generation

Ciel uses Kokoro `af_heart` for identity. Ciel category reference files guide cadence, emotion, pacing, intonation, tone variation, and sentence-ending variation.

Reference behavior:

- `intro` uses `ciel/intro/ciel_intro_01.wav`
- `friendly_encouragement` rotates among available friendly files
- `gentle_reassurance` rotates among available gentle files
- `happy_praise` rotates among available happy files
- `focused_instruction` rotates among available instruction files
- `playful_friend` rotates among available playful files

Selection is deterministic and duration-weighted for cache stability. The selected reference file, reference duration, reference weighting version, Kokoro speaker reference, text, curated/humanized text, intent, voice ID, speed, emotion prompt, engine, and cache version are included in the cache key.

## Curated 7s Output Generation

Run:

```powershell
python scripts\generate_curated_7s_comparisons.py --force
```

Outputs are saved to:

```text
storage/tts/cache/comparisons/curated_7s/
```

The script writes:

- raw Kokoro baseline
- old humanizer comparison output
- curated prompt Kokoro output
- IndexTTS2 expressive output attempt
- `GENERATION_REPORT.md`

The report includes the agent, intent, selected curated prompt, reference file, reference duration, priority, weight, output path, output duration, target-range status, fallback status, and exact error when generation fails.

## Database-Backed Two-Stage Voice Lines

Deployment and defense playback now use a database-first voice line registry in the main Laravel app. ReaDirect-TTS exposes `POST /voice-lines/generate-batch` so Laravel can pre-generate both stages into `storage/app/public/tts/generated_voice_lines/`.

- Stage 1 `reference_style`: generated from the selected expressive reference audio directly, used as default defense audio.
- Stage 2 `kokoro_identity`: generated from the fixed Kokoro speaker identity with Stage 1 as the cadence/style guide.

The active playback stage is controlled by `READIRECT_TTS_ACTIVE_STAGE`. See `../READIRECT_VOICE_LINE_DATABASE.md` for the database schema, commands, fallback order, and report locations.

## Deployment Notes

Kokoro remains the required stable path. IndexTTS2 is optional and experimental. Missing dependencies, missing models, an empty manifest, or empty reference folders must not make the TTS service fail. With fallback enabled, the service returns Kokoro audio through the same Laravel contract.
