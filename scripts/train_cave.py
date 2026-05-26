import os
import sys
from os.path import abspath, dirname, join

script_dir = dirname(abspath(__file__))
project_root = dirname(script_dir)
for path in [project_root, script_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.lso_utils import CAVE_CONFIG
from scripts.train_utils import build_parser, project_root, run_training


def main():
    parser = build_parser(
        description="Train LSO on CAVE",
        default_dataset="cave_x4",
        dataset_choices=list(CAVE_CONFIG.keys()),
        default_dataroot=join(project_root, "data", "cave_x4"),
        default_model="lso_x4",
    )
    opt = parser.parse_args()
    run_training(opt, CAVE_CONFIG, "train_cave(with_up)x4.h5", "test_cave(with_up)x4.h5", "cave")


if __name__ == "__main__":
    main()
