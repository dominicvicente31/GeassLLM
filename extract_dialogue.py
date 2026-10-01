import re
from pathlib import Path

# --- Paths ---
scripts_dir = Path(__file__).parent / "data" / "full_scripts"
output_dir  = Path(__file__).parent / "data" / "LelouchQAScripts"
# In Colab:
# scripts_dir = Path("/content/drive/MyDrive/GeassLLM/data/full_scripts")
# output_dir  = Path("/content/drive/MyDrive/GeassLLM/data/LelouchQAScripts")

LELOUCH_NAMES = {"lelouch", "zero", "lelouch vi britannia"}

def parse_script(path):
    """Parse a script file into (speaker, line) tuples.
    Expects lines in the format:  SPEAKER: dialogue
    Speaker names are case-insensitive.
    """
    pairs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        match = re.match(r"^([A-Za-z][A-Za-z '\-\.]*?):\s+(.+)$", raw)
        if match:
            speaker = match.group(1).strip().lower()
            line    = match.group(2).strip()
            pairs.append((speaker, line))
    return pairs


def extract_pairs(pairs):
    """Yield (prompt, response) where response is a Lelouch line
    and prompt is the immediately preceding non-Lelouch line.
    """
    for i, (speaker, line) in enumerate(pairs):
        if speaker not in LELOUCH_NAMES:
            continue
        if i == 0:
            continue
        prev_speaker, prev_line = pairs[i - 1]
        if prev_speaker in LELOUCH_NAMES:
            continue
        yield prev_line, line


def _natural_key(p):
    parts = re.split(r'(\d+)', p.stem)
    return [int(x) if x.isdigit() else x.lower() for x in parts]

def main():
    script_files = sorted((f for f in scripts_dir.glob("*.txt") if f.stat().st_size > 0), key=_natural_key)
    if not script_files:
        raise FileNotFoundError(f"No .txt files found in {scripts_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    for i, path in enumerate(script_files, start=1):
        pairs = list(extract_pairs(parse_script(path)))
        output_file = output_dir / f"Lelouch#{i:02d}.txt"
        with output_file.open("w", encoding="utf-8") as f:
            for prompt, response in pairs:
                f.write(f"[USER]: {prompt}\n[LELOUCH]: {response}\n\n")
        print(f"Saved {len(pairs):,} pairs -> {output_file.name}")
        total_pairs += len(pairs)

    print(f"\nTotal: {total_pairs:,} pairs from {len(script_files)} episodes")

    # Preview first 5 pairs from the first file
    first_pairs = list(extract_pairs(parse_script(script_files[0])))
    print("\n--- Sample pairs ---")
    for prompt, response in first_pairs[:5]:
        print(f"[USER]: {prompt}")
        print(f"[LELOUCH]: {response}")
        print()


if __name__ == "__main__":
    main()
