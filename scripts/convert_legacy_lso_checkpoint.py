import argparse
import os
import sys
from os.path import abspath, dirname

script_dir = dirname(abspath(__file__))
project_root = dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a legacy full-model checkpoint to an LSO state_dict checkpoint")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sf", type=int, required=True)
    parser.add_argument("--legacy_root", default=None, help="Optional source repo root needed to import legacy pickle modules")
    parser.add_argument("--source_label", default="converted_from_legacy_full_model_checkpoint", help="Sanitized label stored in output metadata")
    parser.add_argument("--num_token", type=int, default=8)
    parser.add_argument("--num_basis", type=int, default=24)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--n_resblocks", type=int, default=4)
    parser.add_argument("--guide_dim", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.legacy_root:
        sys.path.insert(0, args.legacy_root)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"].state_dict()
    else:
        raise ValueError(f"Unknown checkpoint format: {list(checkpoint.keys())}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    config = vars(args).copy()
    config.pop("legacy_root", None)
    config.pop("source", None)
    config.pop("output", None)
    config.pop("source_label", None)
    torch.save(
        {
            "epoch": checkpoint.get("epoch"),
            "model_state_dict": state_dict,
            "model_name": args.model_name,
            "config": config,
            "source_checkpoint": args.source_label,
        },
        args.output,
    )
    print(f"Saved converted checkpoint: {args.output}")


if __name__ == "__main__":
    main()
