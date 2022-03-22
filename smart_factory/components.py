from typing import List, Set

from ml_deeco.simulation import StationaryComponent2D, MovingComponent2D, Component

from configuration import CONFIGURATION


class Door(StationaryComponent2D):
    pass


class Dispenser(StationaryComponent2D):
    pass


class Factory(Component):

    entryDoor: Door
    dispenser: Dispenser

    def __init__(self):
        super().__init__()
        self.dispenser = Dispenser(None)


class WorkPlace(Component):

    factory: Factory
    entryDoor: Door

    def __init__(self, factory: Factory):
        super().__init__()
        self.factory = factory


class Shift(Component):

    def __init__(self, workPlace: WorkPlace):
        super().__init__()
        self.workPlace = workPlace
        self.startTime = CONFIGURATION.shiftStart
        self.endTime = CONFIGURATION.shiftEnd
        self.assigned: Set['Worker'] = set()  # originally assigned for the shift
        self.standbys: Set['Worker'] = set()
        self.cancelled: Set['Worker'] = set()
        self.calledStandbys: Set['Worker'] = set()
        self.workers: Set['Worker'] = set()  # actually working (subset of assigned and standbys)

    @property
    def availableStandbys(self):
        return self.standbys - self.calledStandbys


class Worker(MovingComponent2D):

    hasHeadGear: bool
    isAtFactory: bool
