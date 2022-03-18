from ml_deeco.simulation import SIMULATION_GLOBALS


def allow(subjects, action, object):
    print(f"Allowing {subjects} {action} {object}")


def now():
    return SIMULATION_GLOBALS.currentTimeStep


# TODO: unused
class Time:

    def __init__(self, minutes: int):
        self.minutes = minutes

    def __repr__(self):
        return f"Time({self.minutes})"
