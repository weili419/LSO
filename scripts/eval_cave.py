import argparse
import os
import sys
from os.path import abspath, dirname, join

script_dir = dirname(abspath(__file__))
project_root = dirname(script_dir)
os.chdir(project_root)
for path in [project_root, script_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.lso_utils import CAVE_CONFIG, add_lso_args, apply_dataset_config, build_lso_model, fill_lso_defaults, get_device, load_lso_checkpoint
from tools.dataset import DatasetFromHdf5
from tools.metrics import ERGAS, PSNR, SAM

try:
    from thop import clever_format, profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LSO on CAVE")
    parser.add_argument("--dataset", type=str, default="cave_x4", choices=list(CAVE_CONFIG.keys()))
    parser.add_argument("--dataroot", type=str, default=join(project_root, "data", "cave_x4"))
    parser.add_argument("--model", type=str, default="lso_x4", choices=["lso_x4", "lso_x8"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--testBatchSize", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--save_results", action="store_true", default=False)
    add_lso_args(parser)
    return parser.parse_args()


def get_test_set(root_dir, dataset_name):
    return DatasetFromHdf5(join(root_dir, "test_cave(with_up)x4.h5"), dataset_name)


def print_model_complexity(model, opt, device):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.3f} M ({total_params:,})")
    if not THOP_AVAILABLE:
        print("FLOPs: skipped because thop is not installed")
        return
    batch_size = 32
    hr_size = 64
    lr_size = max(1, hr_size // int(opt.sf))
    import copy

    input_rgb = torch.randn(batch_size, opt.n_bands_rgb, hr_size, hr_size, device=device)
    input_hsi = torch.randn(batch_size, opt.n_bands, lr_size, lr_size, device=device)
    model_copy = copy.deepcopy(model)
    try:
        flops, params = profile(model_copy, inputs=(input_rgb, input_hsi), verbose=False)
    finally:
        del model_copy
    flops_fmt, _ = clever_format([flops, params], "%.3f")
    print(f"FLOPs (per sample): {flops_fmt}")


def main():
    opt = fill_lso_defaults(apply_dataset_config(parse_args(), CAVE_CONFIG))
    device = get_device(opt.gpu)
    print("=" * 70)
    print("LSO CAVE Evaluation")
    print(f"Dataset: {opt.dataset}")
    print(f"Model: {opt.model}")
    print(f"Data root: {opt.dataroot}")
    print(f"Model path: {opt.model_path}")
    print(f"Scale factor: {opt.sf}")
    print("=" * 70)

    model = build_lso_model(opt, device)
    epoch, _ = load_lso_checkpoint(opt.model_path, model, device)
    print(f"Loaded checkpoint epoch: {epoch}")
    model.eval()
    print_model_complexity(model, opt, device)

    test_set = get_test_set(opt.dataroot, opt.dataset)
    loader = DataLoader(test_set, batch_size=opt.testBatchSize, shuffle=False, num_workers=opt.threads, pin_memory=torch.cuda.is_available(), persistent_workers=opt.threads > 0)
    print(f"Test set size: {len(test_set)}")

    results_dir = join(project_root, "RESULTS", opt.dataset, opt.model)
    os.makedirs(results_dir, exist_ok=True)
    total = {"PSNR": 0.0, "SAM": 0.0, "ERGAS": 0.0}
    all_metrics = {"PSNR": [], "SAM": [], "ERGAS": []}

    with torch.no_grad():
        for index, batch in enumerate(loader):
            input_rgb = batch[0].to(device, non_blocking=True)
            ms = batch[1].to(device, non_blocking=True)
            ref = batch[2].to(device, non_blocking=True)
            if index == 0:
                print(f"First batch RGB shape: {tuple(input_rgb.shape)}")
                print(f"First batch LR HSI shape: {tuple(ms.shape)}")
                print(f"First batch GT shape: {tuple(ref.shape)}")
            out = model(input_rgb, ms).clamp(0.0, 1.0)
            if index == 0:
                print(f"First batch output shape: {tuple(out.shape)}")
            ref_np = ref[0].permute(1, 2, 0).cpu().numpy()
            out_np = out[0].permute(1, 2, 0).cpu().numpy()
            metrics = {
                "PSNR": PSNR(ref_np, out_np)[0],
                "SAM": SAM(ref_np, out_np),
                "ERGAS": ERGAS(ref_np, out_np),
            }
            print(f"Sample {index + 1}/{len(test_set)}: PSNR: {metrics['PSNR']:.4f} dB, SAM: {metrics['SAM']:.4f}, ERGAS: {metrics['ERGAS']:.4f}")
            for key, value in metrics.items():
                total[key] += float(value)
                all_metrics[key].append(float(value))
            if opt.save_results:
                import scipy.io as io
                io.savemat(join(results_dir, f"sample_{index:04d}.mat"), {"output": out_np, **metrics})

    count = len(all_metrics["PSNR"])
    print("=" * 70)
    print(f"Test Results (Average over {count} samples)")
    print(f"PSNR:  {total['PSNR'] / count:.4f} +/- {np.std(all_metrics['PSNR']):.4f} dB")
    print(f"SAM:   {total['SAM'] / count:.4f} +/- {np.std(all_metrics['SAM']):.4f}")
    print(f"ERGAS: {total['ERGAS'] / count:.4f} +/- {np.std(all_metrics['ERGAS']):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
