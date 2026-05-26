import os
import sys
from os.path import abspath, dirname, join

script_dir = dirname(abspath(__file__))
project_root = dirname(script_dir)
for path in [project_root, script_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.lso_utils import HARVARD_CONFIG
from scripts.train_utils import build_parser, project_root, run_training


def main():
    parser = build_parser(
        description="Train LSO on Harvard",
        default_dataset="Harvard_x8",
        dataset_choices=list(HARVARD_CONFIG.keys()),
        default_dataroot=join(project_root, "data", "Harvard_x4"),
        default_model="lso_x8",
    )
    opt = parser.parse_args()
    run_training(opt, HARVARD_CONFIG, "train_harvard(with_up)x4.h5", "validation_harvard(with_up)x4.h5", "harvard")


if __name__ == "__main__":
    main()
