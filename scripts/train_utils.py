import argparse
import datetime
import os
import random
import sys
import time
from os.path import abspath, dirname, join

script_dir = dirname(abspath(__file__))
project_root = dirname(script_dir)
os.chdir(project_root)
for path in [project_root, script_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from scripts.lso_utils import add_lso_args, apply_dataset_config, build_lso_model, fill_lso_defaults, get_device, save_lso_checkpoint
from tools.dataset import DatasetFromHdf5
from tools.metrics import PSNR


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def build_parser(description, default_dataset, dataset_choices, default_dataroot, default_model):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=str, default=default_dataset, choices=dataset_choices)
    parser.add_argument("--dataroot", type=str, default=default_dataroot)
    parser.add_argument("--model", type=str, default=default_model, choices=["lso_x4", "lso_x8"])
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--testBatchSize", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--val_threads", type=int, default=2)
    parser.add_argument("--start_epochs", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--save_root", type=str, default=join(project_root, "checkpoints"))
    parser.add_argument("--max_train_batches", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--no_save", action="store_true", default=False, help="Disable checkpoint writes during smoke tests")
    add_lso_args(parser)
    return parser


def run_training(opt, dataset_config, train_file, val_file, checkpoint_group):
    set_random_seed(opt.seed)
    opt = fill_lso_defaults(apply_dataset_config(opt, dataset_config))
    device = get_device(opt.gpu)
    train_set = DatasetFromHdf5(join(opt.dataroot, train_file), opt.dataset)
    val_set = DatasetFromHdf5(join(opt.dataroot, val_file), opt.dataset)
    train_loader = DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=opt.threads, pin_memory=torch.cuda.is_available(), persistent_workers=opt.threads > 0)
    val_loader = DataLoader(val_set, batch_size=opt.testBatchSize, shuffle=False, num_workers=opt.val_threads, pin_memory=torch.cuda.is_available())

    model = build_lso_model(opt, device)
    loss_fn = nn.L1Loss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.6)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print(f"LSO training: dataset={opt.dataset}, model={opt.model}, sf={opt.sf}")
    print(f"Data root: {opt.dataroot}")
    print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}")
    print(f"Trainable parameters: {trainable:,} ({trainable / 1e6:.2f}M)")
    print(f"Total parameters: {total:,} ({total / 1e6:.2f}M)")
    print("=" * 70)

    timestamp = time.strftime("%Y%m%d%H%M")
    run_name = f"{opt.model}_{opt.dataset}_{timestamp}"
    checkpoint_dir = join(opt.save_root, checkpoint_group, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_psnr = 0.0

    for epoch in range(opt.start_epochs, opt.n_epochs + 1):
        model.train()
        print(f"Train_Epoch_{epoch}: lr={optimizer.param_groups[0]['lr']}")
        for iteration, batch in enumerate(train_loader, 1):
            if opt.max_train_batches is not None and iteration > opt.max_train_batches:
                break
            input_rgb = batch[0].to(device, non_blocking=True)
            ms = batch[1].to(device, non_blocking=True)
            ref = batch[2].to(device, non_blocking=True)
            out = model(input_rgb, ms)
            loss = loss_fn(out, ref)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % opt.val_interval == 0 or epoch == opt.n_epochs:
            model.eval()
            values = []
            with torch.no_grad():
                for val_iteration, batch in enumerate(val_loader, 1):
                    if opt.max_val_batches is not None and val_iteration > opt.max_val_batches:
                        break
                    input_rgb = batch[0].to(device, non_blocking=True)
                    ms = batch[1].to(device, non_blocking=True)
                    ref = batch[2].to(device, non_blocking=True)
                    out = model(input_rgb, ms).clamp(0.0, 1.0)
                    ref_np = ref[0].permute(1, 2, 0).cpu().numpy()
                    out_np = out[0].permute(1, 2, 0).cpu().numpy()
                    values.append(float(PSNR(ref_np, out_np)[0]))
            current_psnr = float(np.mean(values))
            print(f"Validation PSNR: {current_psnr:.4f}")
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                if opt.no_save:
                    print("Checkpoint saving disabled by --no_save")
                else:
                    path = join(checkpoint_dir, "model_best.pth.tar")
                    save_lso_checkpoint(path, model, epoch, opt.model, vars(opt))
                    with open(join(checkpoint_dir, "best_epoch_info.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Best Model Epoch: {epoch}\n")
                        f.write(f"Best PSNR: {best_psnr:.4f}\n")
                        f.write(f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    print(f"Best checkpoint saved to {path}")
    print(f"Best PSNR: {best_psnr:.4f}")
    return best_psnr
