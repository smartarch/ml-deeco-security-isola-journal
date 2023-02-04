import numpy as np

from ml_deeco.estimators import TimeEstimate, NumericFeature
from ml_deeco.simulation import Component

from configuration import CONFIGURATION


class ProductionMachine(Component):

    def __init__(self):
        super().__init__()
        self.isRunning = True
        self.failureRate = 0

        self.timeToRepair = 0
        self.timeSinceLastFailure = 0

    timeToFailure = TimeEstimate().using(CONFIGURATION.timeToFailureEstimator)

    @timeToFailure.input(NumericFeature(0, 1))
    def failure_rate(self):
        return self.failureRate

    @timeToFailure.inputsValid
    def is_running(self):
        return self.isRunning

    @timeToFailure.condition
    def failed(self):
        return self.failureRate > CONFIGURATION.failureThreshold

    def fail(self):
        self.isRunning = False
        self.timeToRepair = CONFIGURATION.timeToRepair

    def repair(self):
        self.isRunning = True
        self.failureRate = 0
        self.timeSinceLastFailure = 0

    def simulateFailureRate(self):
        self.failureRate = np.random.normal(CONFIGURATION.failureRateMean(self.timeSinceLastFailure), CONFIGURATION.failureRateVariance)
        self.timeSinceLastFailure += 1

    def actuate(self):
        if self.isRunning:
            self.simulateFailureRate()
            if self.failureRate > CONFIGURATION.failureThreshold:
                self.fail()

        if self.timeToRepair > 0:
            self.timeToRepair -= 1
            if self.timeToRepair == 0:
                self.repair()
