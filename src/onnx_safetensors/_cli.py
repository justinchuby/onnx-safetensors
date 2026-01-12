"""Command line interface for onnx-safetensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx_ir as ir

import onnx_safetensors


def convert_command(args: argparse.Namespace) -> None:
    """Convert an ONNX model to use safetensors format.

    Args:
        args: Command line arguments.
    """
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load the ONNX model
    print(f"Loading ONNX model from {input_path}...")
    model = ir.load(input_path)

    # Save the model with safetensors
    print("Converting model to safetensors format...")
    onnx_safetensors.save_model(
        model,
        output_path,
        max_shard_size=args.max_shard_size,
    )

    print(f"Model saved to {output_path}")


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to use safetensors format"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert an ONNX model to use safetensors format"
    )
    convert_parser.add_argument("input", type=str, help="Path to the input ONNX model")
    convert_parser.add_argument(
        "output", type=str, help="Path to the output ONNX model"
    )
    convert_parser.add_argument(
        "--max-shard-size",
        type=str,
        default=None,
        help='Maximum size for each shard (e.g., "5GB", "100MB"). If not specified, no sharding is performed.',
    )

    args = parser.parse_args()

    if args.command == "convert":
        convert_command(args)
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
