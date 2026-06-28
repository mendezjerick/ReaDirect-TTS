from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = SERVICE_ROOT / "storage" / "tts" / "cache" / "comparisons" / "ciel"

CIEL_LINES = {
    "intro": [
        "Hi, I'm Miss Ciel. I'll read with you today.",
    ],
    "friendly_encouragement": [
        "Take your time, then read this one out loud.",
        "You can do it. Let's try it slowly.",
        "Ready? Let's read this one together.",
    ],
    "gentle_reassurance": [
        "That's okay, let's try that one more time.",
        "No worries, we can slow it down together.",
        "That was close. Let's say it again.",
    ],
    "happy_praise": [
        "Nice work! You said that clearly.",
        "Great job! You got that one.",
        "Good job, that was clear.",
    ],
    "focused_instruction": [
        "Listen carefully first, then say the word after me.",
        "When you're ready, say the sound clearly.",
        "Look at the word, then read it out loud.",
    ],
    "playful_friend": [
        "Ready? Let's go slowly.",
        "Let's give this one a try.",
        "Nice, let's move to the next one.",
    ],
}


def get_json(base_url: str, path: str) -> dict:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def post_audio(base_url: str, payload: dict) -> tuple[bytes, dict[str, str]]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/synthesize",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TTS request failed: HTTP {exc.code} {detail}") from exc


def expressive_status(health: dict) -> tuple[bool, bool, Optional[str]]:
    index = (health.get("engines") or {}).get("index_tts2") or {}
    enabled = bool(index.get("enabled"))
    available = bool(index.get("available"))
    reason = index.get("availability_reason")
    return enabled, available, str(reason) if reason else None


def write_audio(base_url: str, output_dir: Path, output_name: str, payload: dict) -> dict[str, str]:
    audio, headers = post_audio(base_url, payload)
    output_path = output_dir / output_name
    output_path.write_bytes(audio)
    print(f"Saved {output_path}")
    return headers


def header_value(headers: dict[str, str], name: str) -> str:
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Ciel raw Kokoro, humanized Kokoro, and optional expressive reference comparison WAVs.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true", help="Bypass the TTS service cache while generating comparison files.")
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    health = get_json(args.base_url, "/health")
    expressive_enabled, expressive_available, expressive_reason = expressive_status(health)
    if not expressive_enabled:
        print("Expressive IndexTTS2 output skipped: expressive_engine_disabled")
    elif not expressive_available:
        print(f"Expressive IndexTTS2 blocker: {expressive_reason or 'adapter_unavailable'}")

    for intent, lines in CIEL_LINES.items():
        for index, line in enumerate(lines, start=1):
            common = {
                "agent": "miss_ciel",
                "text": line,
                "cache": not args.force,
                "context": "agent_narration",
                "metadata": {"prosody_intent": intent},
            }
            raw_payload = {
                **common,
                "engine": "kokoro",
                "humanize": False,
                "delivery_control": False,
                "audio_humanizer": False,
                "pause_control": False,
            }
            humanized_payload = {
                **common,
                "engine": "kokoro",
                "humanize": True,
                "delivery_control": True,
                "audio_humanizer": True,
                "pause_control": True,
            }

            base_name = f"{intent}_{index:02d}"
            write_audio(args.base_url, output_dir, f"{base_name}_kokoro_raw.wav", raw_payload)
            write_audio(args.base_url, output_dir, f"{base_name}_kokoro_humanized.wav", humanized_payload)

            if expressive_enabled:
                expressive_payload = {
                    **humanized_payload,
                    "engine": "index_tts2",
                    "expressive": True,
                }
                audio, headers = post_audio(args.base_url, expressive_payload)
                reference = header_value(headers, "X-ReaDirect-TTS-Reference")
                fallback = header_value(headers, "X-ReaDirect-TTS-Fallback-Reason")
                if fallback:
                    print(f"  expressive blocked for {base_name}: {fallback}")
                elif header_value(headers, "X-ReaDirect-TTS-Expressive") == "1":
                    output_path = output_dir / f"{base_name}_index_tts2_expressive.wav"
                    output_path.write_bytes(audio)
                    print(f"Saved {output_path}")
                    if reference:
                        print(f"  reference: {reference}")
                    mode = header_value(headers, "X-ReaDirect-TTS-Mode")
                    if mode:
                        print(f"  mode: {mode}")
                else:
                    print(f"  expressive blocked for {base_name}: expressive_not_used")

    print()
    print(f"Done. Listen to WAV files in {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
