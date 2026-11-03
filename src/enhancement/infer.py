"""Run the configured speech enhancement model on a WAV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.audio.loader import load_wav, save_wav
from src.enhancement.factory import create_enhancer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input WAV file")
    parser.add_argument("--output", type=Path, required=True, help="Enhanced WAV output")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/enhancement.yaml"),
        help="Enhancement component YAML config",
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError("Enhancement config root must be a mapping")

    enhancer = create_enhancer(config)
    enhanced = enhancer.enhance(load_wav(args.input))
    save_wav(enhanced, args.output)
    print(f"Enhanced audio saved to: {args.output}")


if __name__ == "__main__":
    main()
