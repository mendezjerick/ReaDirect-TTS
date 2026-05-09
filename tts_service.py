from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


SERVICE_NAME = "readirect-tts"
PROVIDER = "kokoro"
SAMPLE_RATE = 24000
MAX_TEXT_LENGTH = 300
CACHE_DIR = Path(__file__).resolve().parent / "generated_audio"

AGENTS = {
    "miss_vivian": {
        "display_name": "Miss Vivian",
        "voice": "af_bella",
        "speed": 0.95,
    },
    "miss_ciel": {
        "display_name": "Miss Ciel",
        "voice": "af_heart",
        "speed": 1.0,
    },
    "miss_estelle": {
        "display_name": "Miss Estelle",
        "voice": "bf_isabella",
        "speed": 0.95,
    },
}

pipeline = None

app = FastAPI(title="ReaDirect TTS", version="1.0.0")


class SynthesizeRequest(BaseModel):
    agent: str
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None
    cache: bool = True


def normalize_agent(agent: str) -> str:
    aliases = {
        "assessment": "miss_vivian",
        "coach_feedback": "miss_ciel",
        "evaluator": "miss_estelle",
        "evaluator_recommendation": "miss_estelle",
    }
    key = aliases.get((agent or "").strip(), (agent or "").strip())
    if key not in AGENTS:
        raise HTTPException(status_code=422, detail="Unknown agent.")
    return key


def sanitize_text(text: str) -> str:
    cleaned = re.sub(r"<[^>]*>", " ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace("`", "").replace("{", "").replace("}", "")

    if not cleaned:
        raise HTTPException(status_code=422, detail="Text is required.")

    if len(cleaned) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=422, detail=f"Text must be {MAX_TEXT_LENGTH} characters or fewer.")

    return cleaned


def cache_key(agent: str, voice: str, speed: float, text: str) -> str:
    source = f"{PROVIDER}|{agent}|{voice}|{speed:.2f}|{text}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def get_pipeline():
    global pipeline
    if pipeline is None:
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code="a")
    return pipeline


def generate_audio(text: str, voice: str, speed: float, output_path: Path) -> None:
    chunks = []
    generator = get_pipeline()(text, voice=voice, speed=speed)

    for _, _, audio in generator:
        chunks.append(np.asarray(audio, dtype=np.float32))

    if not chunks:
        raise RuntimeError("No audio was generated.")

    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, SAMPLE_RATE)


def audio_path_for(request: SynthesizeRequest) -> tuple[str, Path, str, float, str]:
    agent = normalize_agent(request.agent)
    text = sanitize_text(request.text)
    profile = AGENTS[agent]
    voice = profile["voice"]
    speed = float(request.speed or profile["speed"])
    key = cache_key(agent, voice, speed, text)

    return key, CACHE_DIR / f"{key}.wav", voice, speed, agent


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": SERVICE_NAME,
        "provider": PROVIDER,
        "voices": {
            "miss_vivian": AGENTS["miss_vivian"]["voice"],
            "miss_ciel": AGENTS["miss_ciel"]["voice"],
            "miss_estelle": AGENTS["miss_estelle"]["voice"],
        },
    }


@app.get("/voices")
def voices():
    return {"agents": AGENTS}


@app.post("/synthesize")
def synthesize(request: SynthesizeRequest):
    key, output_path, voice, speed, agent = audio_path_for(request)

    if not request.cache or not output_path.exists():
        try:
            generate_audio(sanitize_text(request.text), voice, speed, output_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="TTS generation failed.") from exc

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"{key}.wav",
        headers={
            "X-ReaDirect-TTS-Provider": PROVIDER,
            "X-ReaDirect-TTS-Agent": agent,
            "X-ReaDirect-TTS-Voice": voice,
            "X-ReaDirect-TTS-Cache-Key": key,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)
