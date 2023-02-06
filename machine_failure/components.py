from ml_deeco.estimators import TimeEstimate, NumericFeature
from ml_deeco.simulation import Component

from configuration import CONFIGURATION
from ml_deeco.utils import verbosePrint, Log


class ProductionMachine(Component):

    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

        self.isRunning = True
        self.failureRate = 0

        self.timeToRepair = 0
        self.timeSinceLastRepair = 0

        self.maintenanceLog = Log(["step", "timeSinceLastRepair", "isRunning", "expectedFailure"])
        self.repairLog = Log(["step", "timeSinceLastRepair", "isRunning"])

    timeToFailure = TimeEstimate().using(CONFIGURATION.timeToFailureEstimator)

    @timeToFailure.input(NumericFeature(0, 1))
    def failure_rate(self):
        return self.failureRate

    # TODO: use timeSinceLastRepair as input?

    @timeToFailure.inputsValid
    def is_running(self):
        return self.isRunning

    @timeToFailure.condition
    def failed(self):
        return self.failureRate > CONFIGURATION.failureThreshold

    def fail(self):
        self.isRunning = False
        self.callMaintenance()

    def repair(self):
        self.repairLog.register([self.experiment.currentTimeStep, self.timeSinceLastRepair, self.isRunning])
        self.isRunning = True
        self.failureRate = 0
        self.timeSinceLastRepair = 0
        self.timeToRepair = 0

    def callMaintenance(self, expectedFailure=None):
        # Simulate maintenance arriving soon
        if self.timeToRepair == 0:
            self.timeToRepair = CONFIGURATION.timeToRepair
            self.maintenanceLog.register([self.experiment.currentTimeStep, self.timeSinceLastRepair, self.isRunning, expectedFailure])
            if expectedFailure:
                verbosePrint(f"{self.id}: Expecting failure in {expectedFailure:.0f} time steps, calling maintenance.", 3)
            else:
                verbosePrint(f"{self.id}: Machine failed, calling maintenance.", 3)
        else:
            pass  # maintenance is already called

    def simulateFailureRate(self):
        self.failureRate = CONFIGURATION.getFailureRate(self.failureRate, self.timeSinceLastRepair)
        self.failureRate = max(self.failureRate, 0)
        self.timeSinceLastRepair += 1

    def preventFailure(self):
        """Call the maintenance if we predict the machine will fail soon"""
        timeToFailure = self.timeToFailure()
        if timeToFailure is None:  # baseline without failure prevention
            # if self.timeSinceLastRepair >= 50:
            #     self.callMaintenance()
            return

        if timeToFailure < CONFIGURATION.timeToRepair:
            # Call maintenance
            self.callMaintenance(timeToFailure)

    def actuate(self):
        if self.timeToRepair > 0:
            self.timeToRepair -= 1
            if self.timeToRepair == 0:
                self.repair()

        if self.isRunning:
            self.simulateFailureRate()
            self.preventFailure()
            if self.failureRate > CONFIGURATION.failureThreshold:
                self.fail()
