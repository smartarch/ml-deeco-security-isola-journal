from typing import List

from configuration import setStandbyArrivedAtWorkplaceTime, CONFIGURATION
from ml_deeco.estimators import ConstantEstimator, NeuralNetworkEstimator, CategoricalFeature, BinaryFeature
from ml_deeco.simulation import Ensemble, someOf
from ml_deeco.utils import verbosePrint

from components import Shift, Worker, WorkerState
from helpers import allow, now, DayOfWeek


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
        return startTime - 30 <= now() <= endTime + 30

    def actuate(self):
        allow(self.shift.workers, "enter", self.factory.entryDoor)


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
        return startTime - 15 <= now() <= endTime

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
        return startTime - 30 <= now() <= endTime + 30

    workers = someOf(Worker)  # subset of self.shift.workers

    @workers.select
    def workers(self, worker, otherEnsembles):
        return worker in self.shift.workers and worker.hasHeadGear

    def actuate(self):
        allow(self.workers, "enter", self.workPlace.entryDoor)


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
        return startTime - 20 <= now() <= endTime

    # region late workers

    def willArriveBaseline(self, _worker, afterTime):
        """As a baseline, we assume that the worker will not arrive if they are not present 10 minutes before the shift starts."""
        return self.shift.startTime - 10 <= now() + afterTime

    # lateWorkers = someOf(Worker).withTimeEstimate(collectOnlyIfMaterialized=False).using(ConstantEstimator(10))

    lateWorkers = someOf(Worker)\
        .withValueEstimate(collectOnlyIfMaterialized=False)\
        .inTimeStepsRange(1, 20)\
        .using(NeuralNetworkEstimator([32, 32], name="worker_arrives", outputFolder="results/worker_arrives"))\
        .withBaseline(willArriveBaseline)

    @lateWorkers.select
    def lateWorkers(self, worker, otherEnsembles):
        if self.potentiallyLate(worker):
            # estimatedArrival = now() + self.lateWorkers.estimate(worker)
            # return estimatedArrival > self.shift.startTime
            timeToShift = self.shift.startTime - now()
            return self.lateWorkers.estimate(worker, timeToShift)
        return False

    # @lateWorkers.estimate.conditionsValid
    @lateWorkers.estimate.targetsValid
    def belongsToShift(self, worker):
        return worker in self.shift.assigned - self.shift.cancelled

    @lateWorkers.estimate.inputsValid
    def potentiallyLate(self, worker):
        return not worker.isAtFactory and self.belongsToShift(worker)

    @lateWorkers.estimate.input()
    def alreadyPresentWorkers(self, worker):
        return len(list(filter(lambda w: w.isAtFactory, self.shift.workers)))

    @lateWorkers.estimate.input(CategoricalFeature(DayOfWeek))
    def dayOfWeek(self, worker):
        return CONFIGURATION.dayOfWeek

    @lateWorkers.estimate.input()
    def currentTime(self, worker):
        return now()

    # @lateWorkers.estimate.condition
    # def arrived(self, worker):
    #     return worker.isAtFactory

    @lateWorkers.estimate.target(BinaryFeature())
    def workerArrived(self, worker):
        return worker.isAtFactory

    # endregion

    def actuate(self):
        self.shift.cancelled.update(self.lateWorkers)
        for worker in self.lateWorkers:
            worker.state = WorkerState.CANCELLED  # this is instead of the notification


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
        verbosePrint(str(self.standbys), 5)
        self.shift.calledStandbys.update(self.standbys)
        for standby in self.standbys:
            standby.state = WorkerState.CALLED_STANDBY  # this is instead of the notification of the standby
            setStandbyArrivedAtWorkplaceTime(standby)


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
