import math
import random


class Configuration:

    steps = 500
    outputFolder = None
    timeToFailureEstimator = None
    failureThreshold = 0.5
    timeToRepair = 30
    machineCount = 10

    @staticmethod
    def failureRateMean(timeSinceLastRepair):
        # return (1.1 ** (timeSinceLastFailure - 100)) / 2  # exponential, at x == 100 => return 0.5 (which is failThreshold)
        return 0.5 / (1 + math.exp(-0.1 * (timeSinceLastRepair - 60)))  # sigmoid, > 0.4 around x == 75

    @staticmethod
    def failureRateVariance(timeSinceLastRepair):
        return 0.01 + timeSinceLastRepair / 2500

    # @staticmethod
    # def failureRateAdd(timeSinceLastFailure):
    #     if timeSinceLastFailure < 50:
    #         return random.random() / 500
    #     return random.random() / 100

    def __init__(self):
        if 'CONFIGURATION' in locals():
            raise RuntimeError("Do not create a new instance of the Configuration. Use the CONFIGURATION global variable instead.")


CONFIGURATION = Configuration()
