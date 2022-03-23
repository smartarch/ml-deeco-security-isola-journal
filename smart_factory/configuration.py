
class Configuration:

    steps = 50
    shiftStart = 30
    shiftEnd = 50
    workersPerShift = 3
    standbysPerShift = 3

    def __init__(self):
        if 'CONFIGURATION' in locals():
            raise RuntimeError("Do not create a new instance of the Configuration. Use the CONFIGURATION global variable instead.")


CONFIGURATION = Configuration()
