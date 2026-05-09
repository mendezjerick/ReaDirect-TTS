# ReaDirect-TTS

Local FastAPI service for ReaDirect agent voices. It uses Kokoro to generate short WAV files for learner-facing agent messages and caches repeated lines. Kokoro is the only spoken TTS provider; ReaDirect falls back to text-only messages if this service is unavailable.

Location:

```text
C:\Users\Lost\Documents\holder-ReaDirect\ReaDirect-TTS
```

Selected voices:

- Miss Vivian: `af_bella`, speed `0.95`
- Miss Ciel: `af_heart`, speed `1.00`
- Miss Estelle: `bf_isabella`, speed `0.95`

## Setup

```powershell
cd "C:\Users\Lost\Documents\holder-ReaDirect\ReaDirect-TTS"
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Run

```powershell
python tts_service.py
```

or double-click:

```text
start_tts_service.bat
```

Service URL:

```text
http://127.0.0.1:8002
```

Health check:

```text
http://127.0.0.1:8002/health
```

## Endpoints

- `GET /health`
- `GET /voices`
- `POST /synthesize`

`POST /synthesize` returns `audio/wav` bytes. Laravel stores those bytes in its own private TTS cache and serves them through a Laravel route. Browsers should not load files directly from this service.

Example:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8002/synthesize" -Method Post -ContentType "application/json" -Body '{"agent":"miss_ciel","text":"Good try! Let us practice again.","cache":true}' -OutFile "sample.wav"
```

## Laravel `.env`

```env
TTS_ENABLED=true
TTS_PROVIDER=kokoro
TTS_BASE_URL=http://127.0.0.1:8002
TTS_TIMEOUT_SECONDS=10
TTS_FALLBACK_TO_TEXT=true
TTS_CACHE_ENABLED=true
TTS_DEBUG=false
TTS_VOICE_VIVIAN=af_bella
TTS_VOICE_CIEL=af_heart
TTS_VOICE_ESTELLE=bf_isabella
TTS_SPEED_VIVIAN=0.95
TTS_SPEED_CIEL=1.00
TTS_SPEED_ESTELLE=0.95
```

## Do Not Commit

- `.venv/`
- `generated_audio/`
- model caches
- test WAV files
- `__pycache__/`

## Troubleshooting

- `No module named kokoro`: activate `.venv` and run `python -m pip install -r requirements.txt`.
- Torch missing or slow: install CPU Torch with the command above.
- PowerShell blocks activation: run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.
- Service not reachable: confirm `python tts_service.py` is running on port `8002`.
- Model or voice errors: confirm Kokoro is installed and the selected voices are available.

## Client Handoff

Kokoro is installed locally on the deployment computer. ReaDirect calls the local TTS service at `http://127.0.0.1:8002`. The selected voices are `af_bella` for Miss Vivian, `af_heart` for Miss Ciel, and `bf_isabella` for Miss Estelle. Model files and generated audio cache are local runtime assets and should not be committed to GitHub.
