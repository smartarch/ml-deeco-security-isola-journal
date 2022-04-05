import random
from typing import Tuple, List

from helpers import DayOfWeek
from ml_deeco.simulation import Point2D, SIMULATION_GLOBALS

from components import WorkPlace, Factory, Door, Dispenser, Worker


class Configuration:

    steps = 80
    shiftStart = 40
    shiftEnd = 80
    workersPerShift = 50
    standbysPerShift = 30
    dayOfWeek = None

    outputFolder = None

    def __init__(self):
        if 'CONFIGURATION' in locals():
            raise RuntimeError("Do not create a new instance of the Configuration. Use the CONFIGURATION global variable instead.")


CONFIGURATION = Configuration()


def createFactory() -> Tuple[Factory, List[WorkPlace], Point2D]:
    factory = Factory()
    factory.entryDoor = Door(20, 90)
    factory.dispenser = Dispenser(30, 90)

    workplace1 = WorkPlace(factory)
    workplace1.entryDoor = Door(40, 50)
    workplace1.pathTo = [Point2D(30, 50)]

    workplace2 = WorkPlace(factory)
    workplace2.entryDoor = Door(120, 50)
    workplace2.pathTo = [Point2D(110, 90), Point2D(110, 50)]

    workplace3 = WorkPlace(factory)
    workplace3.entryDoor = Door(120, 110)
    workplace3.pathTo = [Point2D(110, 90), Point2D(110, 110)]

    busStop = Point2D(0, 90)

    return factory, [workplace1, workplace2, workplace3], busStop


weekDayMean, weekDayStd = 20, 10
weekEndMean, weekEndStd = 10, 10
standbyMean, standbyStd = 60, 10


def setArrivalTime(worker: Worker, dayOfWeek):
    dayOfWeek = DayOfWeek(dayOfWeek % 7)

    if dayOfWeek in (DayOfWeek.SATURDAY, DayOfWeek.SUNDAY):
        worker.busArrivalTime = int(random.gauss(weekEndMean, weekEndStd))
    else:
        worker.busArrivalTime = int(random.gauss(weekDayMean, weekDayStd))


# we will not simulate the standby, just assume they will start working about an hour after they are called
def setStandbyArrivedAtWorkplaceTime(standby: Worker):
    standby.arrivedAtWorkplaceTime = SIMULATION_GLOBALS.currentTimeStep + int(random.gauss(standbyMean, standbyStd))
