from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_LINES = [
    "Good job.",
    "Try again.",
    "Listen carefully.",
    "That was close.",
    "Take your time and read the word out loud.",
    "You did well today. Let's look at your result together.",
]

AGENTS = ["miss_ciel", "miss_vivian", "miss_estelle"]


def slug(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return lowered[:48] or "line"


def post_audio(base_url: str, payload: dict) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/synthesize",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TTS request failed: HTTP {exc.code} {detail}") from exc


def write_pair(base_url: str, output_dir: Path, agent: str, line: str) -> None:
    common = {
        "agent": agent,
        "text": line,
        "cache": False,
        "context": "agent_narration",
    }
    raw = {
        **common,
        "humanize": False,
        "delivery_control": False,
        "audio_humanizer": False,
        "pause_control": False,
    }
    humanized = {
        **common,
        "humanize": True,
        "delivery_control": True,
        "audio_humanizer": True,
        "pause_control": True,
    }

    line_slug = slug(line)
    raw_path = output_dir / f"{agent}_{line_slug}_raw.wav"
    humanized_path = output_dir / f"{agent}_{line_slug}_humanized.wav"

    raw_path.write_bytes(post_audio(base_url, raw))
    humanized_path.write_bytes(post_audio(base_url, humanized))
    print(f"Saved {raw_path}")
    print(f"Saved {humanized_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate raw and humanized ReaDirect TTS comparison WAV files.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--out-dir", default="generated_audio/humanization_comparison")
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for agent in AGENTS:
        for line in DEFAULT_LINES:
            write_pair(args.base_url, output_dir, agent, line)

    print()
    print(f"Done. Listen to WAV files in {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
