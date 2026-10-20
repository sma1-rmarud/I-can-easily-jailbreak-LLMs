"""Construct an HF export using the starting model's context and local tokenizer."""

import json
from pathlib import Path
import sys


def export_command(core, base_checkpoint, dataset_path, final_checkpoint, output, sequence_length):
    source = json.loads((Path(base_checkpoint) / "source.json").read_text())
    original = json.loads((Path(source["snapshot"]) / "config.json").read_text())
    context = original["max_position_embeddings"]
    if isinstance(context, bool) or not isinstance(context, int) or context < sequence_length:
        raise ValueError("Starting HF config has an invalid or too short context length")
    tokenizer = Path(dataset_path) / "tokenizer"
    if not (tokenizer / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Tokenized data's saved tokenizer is missing: {tokenizer}")
    return [
        sys.executable, str(Path(core) / "src/examples/huggingface/convert_checkpoint_to_hf.py"),
        "-i", str(final_checkpoint), "-o", str(output),
        "-s", str(context), "-t", str(tokenizer),
        "--skip-validation",
    ]
