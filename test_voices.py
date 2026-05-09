from kokoro import KPipeline
import soundfile as sf
from pathlib import Path
import csv

pipeline = KPipeline(lang_code="a")

output_dir = Path("voice_tests_female")
output_dir.mkdir(exist_ok=True)

# Female voices only
voices = [
    # American English female voices
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "af_alloy",
    "af_aoede",
    "af_jessica",
    "af_kore",
    "af_nova",
    "af_river",

    # British English female voices
    "bf_emma",
    "bf_isabella",
    "bf_alice",
    "bf_lily",
]

# 0.90 = calmer/slower
# 1.00 = normal
# 1.08 = slightly energetic
speeds = [0.90, 1.00, 1.08]

samples = {
    "miss_vivian": "Hello! I am Miss Vivian. I will guide you through your reading assessment. Try your best and answer one step at a time.",
    "miss_ciel": "Hi! I am Miss Ciel. I will help you practice reading. Mistakes are okay. I am here to guide you.",
    "miss_estelle": "Hello! I am Miss Estelle. I will help explain your results. Great job finishing your activity.",
}

manifest_path = output_dir / "voice_test_manifest.csv"

with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_file:
    writer = csv.writer(manifest_file)
    writer.writerow(["agent", "voice", "speed", "filename", "status"])

    for voice in voices:
        for speed in speeds:
            for agent, text in samples.items():
                filename = f"{agent}_{voice}_speed_{str(speed).replace('.', '_')}.wav"
                filepath = output_dir / filename

                try:
                    print(f"Generating: {agent} | {voice} | speed={speed}")

                    generator = pipeline(
                        text,
                        voice=voice,
                        speed=speed,
                    )

                    audio_saved = False

                    for _, _, audio in generator:
                        sf.write(filepath, audio, 24000)
                        audio_saved = True
                        break

                    if audio_saved:
                        writer.writerow([agent, voice, speed, filename, "saved"])
                        print(f"Saved: {filepath}")
                    else:
                        writer.writerow([agent, voice, speed, filename, "no audio generated"])
                        print(f"No audio generated for {voice}")

                except Exception as e:
                    writer.writerow([agent, voice, speed, filename, f"skipped: {e}"])
                    print(f"Skipped {voice} at speed {speed}: {e}")

print()
print(f"Done. Check the folder: {output_dir}")
print(f"Manifest saved as: {manifest_path}")