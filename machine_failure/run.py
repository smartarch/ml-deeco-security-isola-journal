import argparse
import os
import random
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf

from ml_deeco.estimators import NeuralNetworkEstimator
from ml_deeco.simulation import Experiment, Configuration
from ml_deeco.utils import setVerboseLevel, Log, setVerbosePrintFile, closeVerbosePrintFile, verbosePrint

from configuration import CONFIGURATION
from plots import plotFailureRate


class ProductionMachineExperiment(Experiment):

    def __init__(self, args):
        config = Configuration(**args.__dict__, name="machine", steps=CONFIGURATION.steps)
        super().__init__(config)

        # initialize output path
        CONFIGURATION.outputFolder = Path(config.output)
        os.makedirs(CONFIGURATION.outputFolder, exist_ok=True)
        outputFile = open(CONFIGURATION.outputFolder / "output.txt", "w")

        # initialize configuration
        CONFIGURATION.timeToFailureEstimator = NeuralNetworkEstimator(
            self, hidden_layers=[128], fit_params={"batch_size": 64}, baseline=None,
            name="time_to_failure", outputFolder=CONFIGURATION.outputFolder / "time_to_failure"
        )

        # initialize verbose printing
        setVerboseLevel(args.verbose)
        setVerbosePrintFile(outputFile)

        # import the component with estimate after the `CONFIGURATION.timeToFailureEstimator` is created
        from components import ProductionMachine
        self.machineLogs = {}

    def prepareSimulation(self, _i, _s):
        """Prepares the components for the simulation"""
        from components import ProductionMachine

        machines = [ProductionMachine(self) for _ in range(CONFIGURATION.machineCount)]

        self.machineLogs = {}
        for machine in machines:
            self.machineLogs[machine] = Log(["step", "timeSinceLastFailure", "isRunning", "failureRate"])

        return machines, []

    def stepCallback(self, components, _ensembles, step):
        for machine in components:
            self.machineLogs[machine].register([step, machine.timeSinceLastRepair, machine.isRunning, machine.failureRate])

    def computeMachinesRunning(self):
        totalRunningTime = 0
        for log in self.machineLogs.values():
            for _, _, isRunning, _ in log.records:
                if isRunning:
                    totalRunningTime += 1
        return totalRunningTime

    def simulationCallback(self, components, _ens, i, s):
        os.makedirs(CONFIGURATION.outputFolder / f"machines/{i+1}/{s+1}", exist_ok=True)
        for machine in components:
            self.machineLogs[machine].export(CONFIGURATION.outputFolder / f"machines/{i+1}/{s+1}/{machine}.csv")
            machine.maintenanceLog.export(CONFIGURATION.outputFolder / f"machines/{i+1}/{s+1}/{machine}_maintenance.csv")
            machine.repairLog.export(CONFIGURATION.outputFolder / f"machines/{i+1}/{s+1}/{machine}_repair.csv")
        plotFailureRate(self.machineLogs, filename=CONFIGURATION.outputFolder / f"machines/{i+1}/{s+1}/failure_rate.png")
        verbosePrint(f"Total running time: {self.computeMachinesRunning()}", 2)

    def iterationCallback(self, i):
        return i == self.config.iterations - 1  # do not train after last iteration

    def trainingCallback(self, i):
        # save the ML model
        CONFIGURATION.timeToFailureEstimator.saveModel(i)


def run():
    parser = argparse.ArgumentParser(description='Smart factory simulation')
    parser.add_argument('-v', '--verbose', type=int, help='the verboseness between 0 and 4.', required=False, default=3)
    parser.add_argument('--seed', type=int, help='Random seed.', required=False, default=42)
    parser.add_argument('--threads', type=int, help='Number of CPU threads TF can use.', required=False, default=4)
    parser.add_argument('-o', '--output', type=str, help='Output folder for the logs.', required=False, default='results')
    parser.add_argument('-i', '--iterations', type=int, help="Number of iterations to run.", required=False, default=2)
    parser.add_argument('-s', '--simulations', type=int, help="Number of simulations to run in each iteration.", required=False, default=1)
    args = parser.parse_args()

    # Fix random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Set number of threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    experiment = ProductionMachineExperiment(args)
    experiment.run()

    closeVerbosePrintFile()


if __name__ == "__main__":
    run()
