from ml_deeco.simulation import SIMULATION_GLOBALS
from ml_deeco.utils import verbosePrint
from components import SecurityComponent


def allow(subjects, action, object: SecurityComponent):
    subjects = list(subjects)
    for s in subjects:
        object.allow(s, action)
    if len(subjects) > 0:
        verbosePrint(f"Allowing {subjects} '{action}' '{object}'", 4)


def now():
    return SIMULATION_GLOBALS.currentTimeStep


# TODO: unused
class Time:

    def __init__(self, minutes: int):
        self.minutes = minutes

    def __repr__(self):
        return f"Time({self.minutes})"
