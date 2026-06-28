from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


TEST_LINES = {
    "miss_ciel": [
        "That's okay, let's try that one more time.",
        "Nice work! You said that clearly.",
        "Listen carefully first, then say the word after me.",
        "Ready? Let's try this one slowly.",
    ],
    "miss_vivian": [
        "Listen carefully first, then say the sound out loud.",
        "Nice, let's move to the next one.",
        "When you're ready, submit your answer.",
    ],
    "miss_estelle": [
        "Let's look at your result together.",
        "You did well in this part, and we'll keep practicing the tricky ones.",
        "That's okay, this helps us know what to practice next.",
    ],
}


def slug(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return lowered[:56] or "line"


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


def write_output(base_url: str, output_dir: Path, agent: str, line: str, suffix: str, payload: dict) -> dict[str, str]:
    audio, headers = post_audio(base_url, payload)
    output_path = output_dir / f"{agent}_{slug(line)}_{suffix}.wav"
    output_path.write_bytes(audio)
    print(f"Saved {output_path}")
    return headers


def expressive_status(health: dict) -> tuple[bool, Optional[str]]:
    index = (health.get("engines") or {}).get("index_tts2") or {}
    enabled = bool(index.get("enabled"))
    available = bool(index.get("available"))
    reason = index.get("availability_reason")
    return enabled and available, str(reason) if reason else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate raw Kokoro, humanized Kokoro, and optional expressive TTS comparison WAV files.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--out-dir", default="storage/tts/cache/comparisons")
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    health = get_json(args.base_url, "/health")
    expressive_available, expressive_reason = expressive_status(health)
    if not expressive_available:
        print(f"Expressive IndexTTS2 output skipped: {expressive_reason or 'disabled_or_unavailable'}")

    for agent, lines in TEST_LINES.items():
        for line in lines:
            common = {
                "agent": agent,
                "text": line,
                "cache": False,
                "context": "agent_narration",
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

            write_output(args.base_url, output_dir, agent, line, "raw_kokoro", raw_payload)
            write_output(args.base_url, output_dir, agent, line, "humanized_kokoro", humanized_payload)

            if expressive_available:
                expressive_payload = {
                    **humanized_payload,
                    "engine": "index_tts2",
                    "expressive": True,
                }
                headers = write_output(args.base_url, output_dir, agent, line, "expressive_index_tts2", expressive_payload)
                fallback = headers.get("X-ReaDirect-TTS-Fallback-Reason")
                if fallback:
                    print(f"  Expressive fallback for {agent}: {fallback}")

    print()
    print(f"Done. Listen to WAV files in {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
