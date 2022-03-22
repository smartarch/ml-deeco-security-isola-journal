import argparse
import os
import random
from typing import List
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf

from ml_deeco.simulation import Component, run_experiment
from ml_deeco.utils import setVerboseLevel

from configuration import CONFIGURATION


def run(args):
    initialize(args)
    run_experiment(1, 1, CONFIGURATION.steps, prepareSimulation)


def initialize(args):
    setVerboseLevel(args.verbose)

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Set number of threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)


def prepareSimulation(_i, _s):
    from components import Factory, WorkPlace, Shift, Worker
    from ensembles import getEnsembles

    components: List[Component] = []
    shifts = []

    factory = Factory()
    components.append(factory)

    for i in range(3):
        workPlace = WorkPlace(factory)
        # TODO: workers
        shift = Shift(workPlace)
        components += [workPlace, shift]
        shifts.append(shift)

    return components, getEnsembles(shifts)


def main():
    parser = argparse.ArgumentParser(description='TODO')  # TODO
    parser.add_argument('-v', '--verbose', type=int, help='the verboseness between 0 and 4.', required=False, default="0")
    parser.add_argument('-s', '--seed', type=int, help='Random seed.', required=False, default=42)
    parser.add_argument('--threads', type=int, help='Number of CPU threads TF can use.', required=False, default=4)
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
