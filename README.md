<div align="center">

# Latent Spectral Operators

### Solving Spatial-Spectral Fusion with Latent Spectral Operators

**Wei Li**, **Jieyuan Pei**, **Junnan Xu**, **Xuanfeng Ding**, **Junwei Zhu**, **Wanjun Chen**, **Jianwei Zheng***

Zhejiang University of Technology

<sub>* Corresponding author</sub>

<a href="https://weili419.github.io/latent-spectral-operators/">
  <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-4FD1C5?style=for-the-badge">
</a>
<a href="https://weili419.github.io/latent-spectral-operators/assets/paper.pdf">
  <img alt="Paper" src="https://img.shields.io/badge/Paper-PDF-8A7DFF?style=for-the-badge">
</a>
<a href="https://github.com/weili419/LSO/blob/main/LICENSE">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-111111?style=for-the-badge">
</a>

</div>

<p align="center">
  <img src="https://weili419.github.io/latent-spectral-operators/assets/overall.png" alt="LSO framework" width="95%">
</p>

## Highlights

Latent Spectral Operators (LSO) formulates spatial-spectral fusion as an operator between spectral functions in a compact latent space. Instead of binding the fusion mapping to one coordinate grid, LSO projects observations into latent spectral tokens, solves the latent operator with a controllable trigonometric basis, and projects the result back to the image domain.

- Compact LSO implementation for hyperspectral image super-resolution and spatial-spectral fusion.
- Training and evaluation scripts for CAVE and Harvard benchmarks.
- Converted `state_dict` checkpoints for CAVE x4/x8 and Harvard x4/x8.
- Cross-scale evaluation support from CAVE x4 training to larger degradation factors.

## Repository Layout

```text
LSO/
├── model/                 # Latent Spectral Operator network
├── scripts/               # Training, evaluation, checkpoint conversion
├── tools/                 # Dataset and metric utilities
├── checkpoints/           # Released LSO checkpoints
├── run_train_*.sh         # Reproducible training entry points
├── run_eval_*.sh          # Reproducible evaluation entry points
└── requirements.txt
```

## Installation

```bash
git clone git@github.com:weili419/LSO.git
cd LSO

conda create -n lso python=3.10 -y
conda activate lso

pip install -r requirements.txt
```

If your machine uses CUDA, install the PyTorch build matching your CUDA version before running the experiments.

## Data Preparation

Datasets are not bundled in this repository. Put the CAVE and Harvard HDF5 files under `data/`, or pass a custom path with `DATAROOT=/path/to/dataset`.

```text
data/
├── cave_x4/
│   ├── train_cave(with_up)x4.h5
│   ├── validation_cave(with_up)x4.h5
│   └── test_cave(with_up)x4.h5
└── Harvard_x4/
    ├── train_harvard(with_up)x4.h5
    ├── validation_harvard(with_up)x4.h5
    └── test_harvardv3(with_up)x4.h5
```

Training scripts use the validation files for model selection. Evaluation scripts use the test files.

The loader uses `GT` and `RGB` from each HDF5 file and regenerates the low-resolution HSI by blur-and-downsample according to the selected dataset setting, such as `cave_x4`, `cave_x8`, `Harvard_x4`, or `Harvard_x8`.

## Model Zoo

| Dataset | Scale | Checkpoint | Evaluation |
|---|---:|---|---|
| CAVE | x4 | `checkpoints/cave/lso_x4_cave_x4_202601210644/model_best.pth.tar` | `./run_eval_cave_x4.sh` |
| CAVE | x8 | `checkpoints/cave/lso_x8_cave_x8_202601210442/model_best.pth.tar` | `./run_eval_cave_x8.sh` |
| Harvard | x4 | `checkpoints/harvard/lso_x4_Harvard_x4_202601221430/model_best.pth.tar` | `./run_eval_harvard_x4.sh` |
| Harvard | x8 | `checkpoints/harvard/lso_x8_Harvard_x8_202601210930/model_best.pth.tar` | `./run_eval_harvard_x8.sh` |

## Evaluation

Run the released checkpoints with the default local data layout:

```bash
./run_eval_cave_x4.sh
./run_eval_cave_x8.sh
./run_eval_harvard_x4.sh
./run_eval_harvard_x8.sh
```

Or specify your dataset and GPU explicitly:

```bash
DATAROOT=/path/to/cave_x4 GPU=0 ./run_eval_cave_x4.sh
DATAROOT=/path/to/Harvard_x4 GPU=0 ./run_eval_harvard_x8.sh
```

For cross-scale generalization on CAVE, the x8 model can be evaluated under multiple degradation factors while resizing every LR-HSI input to `64x64`:

```bash
DATAROOT=/path/to/cave_x4 ./run_eval_cave_vary_target64.sh

DATASETS="cave_x4 cave_x8 cave_x12 cave_x16 cave_x32" \
DATAROOT=/path/to/cave_x4 \
./run_eval_cave_vary_target64.sh
```

Add `--save_results` to the Python evaluation commands if you want to save `.mat` outputs under `RESULTS/`.

## Training

```bash
DATAROOT=/path/to/cave_x4 GPU=0 ./run_train_cave_x4.sh
DATAROOT=/path/to/Harvard_x4 GPU=0 ./run_train_harvard_x8.sh
```

The default configuration uses:

| Option | Value |
|---|---:|
| `num_token` | 8 |
| `num_basis` | 24 |
| `width` | 64 |
| `patch_size` | 4 |
| `n_resblocks` | 4 |
| `guide_dim` | 128 |

Checkpoints from new runs are saved to `checkpoints/<dataset_group>/<run_name>/`.

## Useful Commands

```bash
# Evaluate a checkpoint with the Python entry directly
python scripts/eval_cave.py \
  --dataset cave_x4 \
  --dataroot /path/to/cave_x4 \
  --model lso_x4 \
  --model_path checkpoints/cave/lso_x4_cave_x4_202601210644/model_best.pth.tar

# Convert a legacy full-model checkpoint to the released state_dict format
python scripts/convert_legacy_lso_checkpoint.py \
  --source /path/to/legacy_checkpoint.pth.tar \
  --output checkpoints/cave/custom_lso/model_best.pth.tar \
  --model_name lso_x4 \
  --dataset cave_x4 \
  --sf 4
```

## Citation

If this repository is useful for your research, please cite:

```bibtex
@inproceedings{li2026lso,
  title     = {Solving Spatial-Spectral Fusion with Latent Spectral Operators},
  author    = {Li, Wei and Pei, Jieyuan and Xu, Junnan and Ding, Xuanfeng and Zhu, Junwei and Chen, Wanjun and Zheng, Jianwei},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
