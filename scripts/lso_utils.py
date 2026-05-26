import os
from os.path import exists

import torch

DEFAULT_LSO_ARGS = {
    "num_token": 8,
    "num_basis": 24,
    "width": 64,
    "patch_size": 4,
    "n_resblocks": 4,
    "guide_dim": 128,
}

CAVE_CONFIG = {
    "cave_x4": {"n_bands": 31, "image_size": 512, "n_bands_rgb": 3, "sf": 4},
    "cave_x8": {"n_bands": 31, "image_size": 512, "n_bands_rgb": 3, "sf": 8},
    "cave_x12": {"n_bands": 31, "image_size": 512, "n_bands_rgb": 3, "sf": 12},
    "cave_x16": {"n_bands": 31, "image_size": 512, "n_bands_rgb": 3, "sf": 16},
    "cave_x32": {"n_bands": 31, "image_size": 512, "n_bands_rgb": 3, "sf": 32},
}

HARVARD_CONFIG = {
    "Harvard_x4": {"n_bands": 31, "image_size": 1000, "n_bands_rgb": 3, "sf": 4},
    "Harvard_x8": {"n_bands": 31, "image_size": 1000, "n_bands_rgb": 3, "sf": 8},
}


def add_lso_args(parser):
    parser.add_argument("--num_token", type=int, default=None)
    parser.add_argument("--num_basis", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--n_resblocks", type=int, default=None)
    parser.add_argument("--guide_dim", type=int, default=None)


def fill_lso_defaults(opt):
    for key, value in DEFAULT_LSO_ARGS.items():
        if getattr(opt, key, None) is None:
            setattr(opt, key, value)
    return opt


def apply_dataset_config(opt, dataset_config):
    cfg = dataset_config[opt.dataset]
    opt.n_bands = cfg["n_bands"]
    opt.image_size = cfg["image_size"]
    opt.n_bands_rgb = cfg["n_bands_rgb"]
    opt.sf = cfg["sf"]
    return opt


def get_device(gpu="0"):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_lso_model(opt, device):
    from model.LSO import LSOModel

    model = LSOModel(
        args=opt,
        bilinear=True,
        n_select_bands=opt.n_bands_rgb,
        n_bands=opt.n_bands,
        sf=opt.sf,
    )
    return model.to(device)


def normalize_state_dict(state_dict):
    out = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        if key.startswith("model."):
            key = key[len("model."):]
        out[key] = value
    return out


def load_lso_checkpoint(checkpoint_path, model, device):
    if not exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"].state_dict()
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
    model.load_state_dict(normalize_state_dict(state_dict))
    return checkpoint.get("epoch"), checkpoint


def save_lso_checkpoint(path, model, epoch, model_name, config, source_checkpoint=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "config": dict(config),
            "source_checkpoint": source_checkpoint,
        },
        path,
    )
