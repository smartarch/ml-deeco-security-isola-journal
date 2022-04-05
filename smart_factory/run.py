import argparse
import os
import random
from typing import List
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf

from ml_deeco.simulation import Component, run_experiment
from ml_deeco.utils import setVerboseLevel, verbosePrint, Log

from configuration import CONFIGURATION, createFactory, setArrivalTime
from components import Shift, Worker
from ensembles import getEnsembles
from helpers import DayOfWeek


arrivedAtWorkplaceTimeAvgTimes = []
workerLogs = {}


def run(args):
    initialize(args)
    run_experiment(2, 7, CONFIGURATION.steps, prepareSimulation,
                   stepCallback=stepCallback, simulationCallback=simulationCallback, iterationCallback=iterationCallback)


def initialize(args):
    setVerboseLevel(args.verbose)

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Set number of threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)


def prepareSimulation(_i, simulation):
    global workerLogs
    workerLogs = {}

    CONFIGURATION.dayOfWeek = DayOfWeek(simulation)

    components: List[Component] = []
    shifts = []

    factory, workplaces, busStop = createFactory()
    components.append(factory)

    for workplace in workplaces:
        workers = [Worker(workplace, busStop) for _ in range(CONFIGURATION.workersPerShift)]
        for worker in workers:
            setArrivalTime(worker, simulation)
        standbys = [Worker(workplace, busStop) for _ in range(CONFIGURATION.standbysPerShift)]
        shift = Shift(workplace, workers, standbys)
        components += [workplace, shift, *workers, *standbys]
        shifts.append(shift)

        for worker in workers + standbys:
            workerLogs[worker] = Log(["x", "y", "state", "atFactory", "headGear"])

    return components, getEnsembles(shifts)


def stepCallback(components, _ens, _step):
    for worker in filter(lambda c: isinstance(c, Worker), components):
        workerLogs[worker].register([int(worker.location.x), int(worker.location.x), worker.state, worker.isAtFactory, worker.hasHeadGear])


def simulationCallback(components, _ens, i, s):
    shifts = filter(lambda c: isinstance(c, Shift), components)
    for shift in shifts:
        arrivedWorkers = list(filter(lambda w: w.arrivedAtWorkplaceTime is not None, shift.workers))
        avgArriveTime = sum(map(lambda w: w.arrivedAtWorkplaceTime, arrivedWorkers)) / len(arrivedWorkers)
        arrivedAtWorkplaceTimeAvgTimes.append(avgArriveTime)
        standbysCount = len(shift.calledStandbys)
        verbosePrint(f"{shift}: arrived {len(arrivedWorkers)} workers ({standbysCount} standbys), avg. time = {avgArriveTime:.2f}", 2)

    workersAtFactory = list(filter(lambda c: isinstance(c, Worker) and c.arrivedAtFactoryTime is not None, components))
    if workersAtFactory:
        avgFactoryArrivalTime = sum(map(lambda w: w.arrivedAtFactoryTime, workersAtFactory)) / len(workersAtFactory)
        verbosePrint(f"Average arrival at factory = {avgFactoryArrivalTime:.2f}", 2)

    os.makedirs(f"results/workers/{i+1}/{s+1}", exist_ok=True)
    for worker in filter(lambda c: isinstance(c, Worker), components):
        workerLogs[worker].export(f"results/workers/{i+1}/{s+1}/{worker}.csv")


def iterationCallback(_i):
    global arrivedAtWorkplaceTimeAvgTimes
    avgTimesAverage = sum(arrivedAtWorkplaceTimeAvgTimes) / len(arrivedAtWorkplaceTimeAvgTimes)
    verbosePrint(f"Average arrival time in the iteration: {avgTimesAverage:.2f}", 1)
    arrivedAtWorkplaceTimeAvgTimes = []


def main():
    parser = argparse.ArgumentParser(description='TODO')  # TODO
    parser.add_argument('-v', '--verbose', type=int, help='the verboseness between 0 and 4.', required=False, default="0")
    parser.add_argument('-s', '--seed', type=int, help='Random seed.', required=False, default=42)
    parser.add_argument('--threads', type=int, help='Number of CPU threads TF can use.', required=False, default=4)
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
