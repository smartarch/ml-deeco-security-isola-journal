from typing import List

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
    assigned: List['Worker']  # originally assigned for the shift
    cancelled: List['Worker']
    standbys: List['Worker']
    working: List['Worker']   # actually working (subset of assigned and standbys)


class Worker(MovingComponent2D):

    hasHeadGear: bool

    pass
