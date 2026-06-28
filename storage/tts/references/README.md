# Expressive TTS Reference Files

Place future IndexTTS2 expressive reference audio in this folder tree. These files are style references only: cadence, pacing, emotion, and delivery. They must not replace the fixed ReaDirect agent voice identities.

Kokoro remains the source of who is speaking:

- Miss Ciel: `af_heart`
- Miss Vivian: `af_bella`
- Miss Estelle: `bf_isabella`

Ciel supports these style folders:

- `ciel/intro/`
- `ciel/gentle_reassurance/`
- `ciel/friendly_encouragement/`
- `ciel/happy_praise/`
- `ciel/focused_instruction/`
- `ciel/playful_friend/`

Supported source formats:

- Preferred: `.wav`
- Optional: `.mp3`, when the local audio stack can decode and normalize it

Preferred normalized internal format:

- WAV
- Mono
- 24 kHz
- Reasonable loudness normalization
- No clipping

Create a real `manifest.json` beside `manifest.example.json` when references are ready. If `manifest.json` is missing or a folder is empty, expressive TTS falls back safely to Kokoro.
