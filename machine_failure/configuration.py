class Configuration:

    steps = 500
    outputFolder = None
    timeToFailureEstimator = None
    failureThreshold = 0.5
    timeToRepair = 30
    machineCount = 10

    @staticmethod
    def failureRateMean(x):
        return (1.1 ** (x - 100)) / 2  # exponential, at x == 100 => return 0.5 (which is failThreshold)

    failureRateVariance = 0.05

    def __init__(self):
        if 'CONFIGURATION' in locals():
            raise RuntimeError("Do not create a new instance of the Configuration. Use the CONFIGURATION global variable instead.")


CONFIGURATION = Configuration()
