from typing import List

from components import Shift, Worker
from helpers import allow, now

from ml_deeco.simulation import Ensemble, someOf


class ShiftTeam(Ensemble):

    shift: Shift

    def __init__(self, shift: Shift):
        super().__init__()
        self.shift = shift

    def priority(self):
        return 5

    workers = someOf(Worker)

    @workers.select
    def workers(self, worker, otherEnsembles):
        return worker in (self.shift.assigned - self.shift.cancelled) or \
            worker in self.shift.calledStandbys

    def actuate(self):
        self.shift.workers = set(self.workers)


class AccessToFactory(Ensemble):

    shift: Shift

    def __init__(self, shift: Shift):
        super().__init__()
        self.shift = shift
        self.factory = shift.workPlace.factory

    def priority(self):
        return 4

    def situation(self):
        startTime = self.shift.startTime
        endTime = self.shift.endTime
        return startTime - 30 < now() < endTime + 30

    def actuate(self):
        allow(self.shift.workers, "enter", self.factory)


class AccessToDispenser(Ensemble):

    shift: Shift

    def __init__(self, shift: Shift):
        super().__init__()
        self.shift = shift
        self.dispenser = shift.workPlace.factory.dispenser

    def priority(self):
        return 4

    def situation(self):
        startTime = self.shift.startTime
        endTime = self.shift.endTime
        return startTime - 15 < now() < endTime

    def actuate(self):
        allow(self.shift.workers, "use", self.dispenser)


class AccessToWorkPlace(Ensemble):

    shift: Shift

    def __init__(self, shift: Shift):
        super().__init__()
        self.shift = shift
        self.workPlace = shift.workPlace

    def priority(self):
        return 3

    def situation(self):
        startTime = self.shift.startTime
        endTime = self.shift.endTime
        return startTime - 30 < now() < endTime + 30

    workers = someOf(Worker)  # subset of self.shift.workers

    @workers.select
    def workers(self, worker, otherEnsembles):
        return worker in self.shift.workers and worker.hasHeadGear

    def actuate(self):
        allow(self.workers, "enter", self.workPlace)


class CancelLateWorkers(Ensemble):

    shift: Shift

    def __init__(self, shift: Shift):
        super().__init__()
        self.shift = shift

    def priority(self):
        return 2

    def situation(self):
        startTime = self.shift.startTime
        endTime = self.shift.endTime
        return startTime - 15 < now() < endTime

    # region late workers

    lateWorkers = someOf(Worker).withTimeEstimate(collectOnlyIfMaterialized=False)
    # TODO: if the worker will come in time
    # TODO: inTimeSteps( from, to )

    @lateWorkers.select
    def lateWorkers(self, worker, otherEnsembles):
        if self.potentiallyLate(worker):
            estimatedArrival = now() + self.lateWorkers.estimate(worker)
            return estimatedArrival > self.shift.startTime
        return False

    @lateWorkers.conditionValid
    def belongsToShift(self, worker):
        return worker in self.shift.assigned - self.shift.cancelled

    @lateWorkers.inputsValid
    def potentiallyLate(self, worker):
        return not worker.isAtFactory and self.belongsToShift(worker)

    @lateWorkers.input()
    def alreadyPresentWorkers(self, worker):
        return len(list(filter(lambda w: w.isAtFactory, self.shift.workers)))

    # TODO: which shift
    # TODO: day of week

    @lateWorkers.estimate.condition
    def arrived(self, worker):
        return worker.isAtFactory

    # endregion

    def actuate(self):
        self.shift.cancelled.update(self.lateWorkers)
        # TODO: notify


# TODO: do the replacement as matching for all the shifts simultaneously
class ReplaceLateWithStandbys(Ensemble):

    lateWorkersEnsemble: CancelLateWorkers  # TODO: list
    shift: Shift

    def __init__(self, lateWorkersEnsemble: CancelLateWorkers):
        super().__init__()
        self.lateWorkersEnsemble = lateWorkersEnsemble
        self.shift = lateWorkersEnsemble.shift

    def priority(self):
        return 1

    def situation(self):
        return self.lateWorkersEnsemble.materialized
        # return any(map(lambda e: e.materialized, self.lateWorkersEnsembles))  # TODO: replace with named abstraction

    standbys = someOf(Worker)  # TODO: matching

    @standbys.select
    def standbys(self, worker, otherEnsembles):
        # TODO: not in other shift
        return worker in self.shift.availableStandbys

    @standbys.cardinality
    def standbys(self):
        return 0, len(self.lateWorkersEnsemble.lateWorkers)

    def actuate(self):
        print(self.standbys)
        # TODO: self.shift.calledStandbys.append(self.standbys)
        # TODO: notify


def getEnsembles(shifts: List[Shift]):
    ensembles: List[Ensemble] = []

    ensembles.extend((ShiftTeam(shift) for shift in shifts))
    ensembles.extend((AccessToFactory(shift) for shift in shifts))
    ensembles.extend((AccessToDispenser(shift) for shift in shifts))
    ensembles.extend((AccessToWorkPlace(shift) for shift in shifts))

    for shift in shifts:
        lateWorkersEnsemble = CancelLateWorkers(shift)
        ensembles.append(lateWorkersEnsemble)
        ensembles.append(ReplaceLateWithStandbys(lateWorkersEnsemble))

    return ensembles
