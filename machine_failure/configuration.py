import math
import random

import numpy as np


class Configuration:

    steps = 1000
    outputFolder = None
    timeToFailureEstimator = None
    failureThreshold = 0.499  # 0.5 with a margin for float rounding errors
    timeToRepair = 30
    machineCount = 100
    # machineCount = 10

    def getFailureRate(self, lastFailureRate, timeSinceLastRepair):
        # return self._failureRateSigmoid(timeSinceLastRepair)
        # return self._failureRateExp(timeSinceLastRepair)
        return self._failureRateCategorical(lastFailureRate)

    @staticmethod
    def _failureRateSigmoid(timeSinceLastRepair):
        mean = 0.5 / (1 + math.exp(-0.1 * (timeSinceLastRepair - 60)))  # sigmoid, > 0.4 around x == 75
        var = 0.01 + timeSinceLastRepair / 2500
        return np.random.normal(mean, var)

    @staticmethod
    def _failureRateExp(timeSinceLastRepair):
        mean = (1.1 ** (timeSinceLastRepair - 100)) / 2  # exponential, at x == 100 => return 0.5 (which is failThreshold)
        var = 0.01 + timeSinceLastRepair / 2500
        return np.random.normal(mean, var)

    @staticmethod
    def _failureRateCategorical(lastFailureRate):
        return lastFailureRate + np.random.choice([0, 0.05], p=[0.9, 0.1])

    def __init__(self):
        if 'CONFIGURATION' in locals():
            raise RuntimeError("Do not create a new instance of the Configuration. Use the CONFIGURATION global variable instead.")


CONFIGURATION = Configuration()
