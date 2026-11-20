"""Train the joint verb+noun model with ActionFormer's stock training loop.

Same arguments as ActionFormer's train.py; run from anywhere (paths in the config are relative to
the ActionFormer checkout, which this script switches into):
    python train_joint.py configs/epic_joint_holdout.yaml --output joint
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import joint_actionformer  # noqa: E402,F401  (registers dataset + model)

os.chdir(joint_actionformer.AF_DIR)
import train as af_train  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the joint verb+noun ActionFormer")
    parser.add_argument("config")
    parser.add_argument("-p", "--print-freq", default=10, type=int)
    parser.add_argument("-c", "--ckpt-freq", default=5, type=int)
    parser.add_argument("--output", default="", type=str)
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    if not os.path.isabs(args.config) and not os.path.exists(args.config):
        args.config = os.path.join(HERE, args.config)
    af_train.main(args)
