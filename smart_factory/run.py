import argparse
import os
import random
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf

from ml_deeco.utils import setVerboseLevel


def run(args):
    initialize(args)
    pass


def initialize(args):
    setVerboseLevel(args.verbose)

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Set number of threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)


def main():
    parser = argparse.ArgumentParser(description='TODO')  # TODO
    parser.add_argument('-v', '--verbose', type=int, help='the verboseness between 0 and 4.', required=False, default="0")
    parser.add_argument('-s', '--seed', type=int, help='Random seed.', required=False, default=42)
    parser.add_argument('--threads', type=int, help='Number of CPU threads TF can use.', required=False, default=4)
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
