from typing import List, Set

from ml_deeco.simulation import StationaryComponent2D, MovingComponent2D, Component


class Door(StationaryComponent2D):
    pass


class Dispenser(StationaryComponent2D):
    pass


class Factory(Component):
    pass


class WorkPlace(Component):

    factory: Factory


class Shift(Component):

    workPlace: WorkPlace
    startTime: int
    endTime: int
    assigned: Set['Worker']  # originally assigned for the shift
    standbys: Set['Worker']
    cancelled: Set['Worker']
    calledStandbys: Set['Worker']
    workers: Set['Worker']   # actually working (subset of assigned and standbys)

    @property
    def availableStandbys(self):
        return self.standbys - self.calledStandbys


class Worker(MovingComponent2D):

    hasHeadGear: bool
    isAtFactory: bool

    pass
