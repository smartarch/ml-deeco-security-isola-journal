from components import Shift, Worker
from helpers import allow, now
from ml_deeco.simulation import Ensemble, someOf


class ShiftTeam(Ensemble):

    shift: Shift

    def __init__(self, shift: Shift):
        super().__init__()
        self.shift = shift

    workers = someOf(Worker)

    @workers.select
    def workers(self, worker, otherEnsembles):
        return worker in (self.shift.assigned - self.shift.cancelled) or \
            worker in self.shift.calledStandbys

    def actuate(self):
        self.shift.workers = set(self.workers)


class AccessToFactory(Ensemble):

    # parent ensemble
    shiftTeam: ShiftTeam
    # TODO: shiftTeam: ShiftTeam = ParentEnsemble()

    def __init__(self, shiftTeam: ShiftTeam):
        super().__init__()
        self.shiftTeam = shiftTeam
        self.factory = shiftTeam.shift.workPlace.factory

    def priority(self):
        return self.shiftTeam.priority() - 1

    def situation(self):
        startTime = self.shiftTeam.shift.startTime
        endTime = self.shiftTeam.shift.endTime
        return startTime - 30 < now() < endTime + 30

    def actuate(self):
        allow(self.shiftTeam.workers, "enter", self.factory)


class AccessToDispenser(Ensemble):

    # parent ensemble
    shiftTeam: ShiftTeam

    def __init__(self, shiftTeam: ShiftTeam):
        super().__init__()
        self.shiftTeam = shiftTeam
        self.dispenser = shiftTeam.shift.workPlace.factory.dispenser

    def priority(self):
        return self.shiftTeam.priority() - 1

    def situation(self):
        startTime = self.shiftTeam.shift.startTime
        endTime = self.shiftTeam.shift.endTime
        return startTime - 15 < now() < endTime

    def actuate(self):
        allow(self.shiftTeam.workers, "use", self.dispenser)


class AccessToWorkPlace(Ensemble):

    # parent ensemble
    shiftTeam: ShiftTeam

    def __init__(self, shiftTeam):
        super().__init__()
        self.shiftTeam = shiftTeam
        self.workPlace = shiftTeam.shift.workPlace

    def priority(self):
        return self.shiftTeam.priority() - 1

    def situation(self):
        startTime = self.shiftTeam.shift.startTime
        endTime = self.shiftTeam.shift.endTime
        return startTime - 30 < now() < endTime + 30

    workers = someOf(Worker)  # subset of self.shiftTeam.workers

    @workers.select
    def workers(self, worker, otherEnsembles):
        return worker in self.shiftTeam.workers and worker.hasHeadGear

    def actuate(self):
        allow(self.workers, "enter", self.workPlace)


class NotificationAboutWorkersThatArePotentiallyLate(Ensemble):
    pass


class LateWorkersReplacement(Ensemble):

    # parent ensemble
    shiftTeam: ShiftTeam

    def __init__(self, shiftTeam):
        super().__init__()
        self.shiftTeam = shiftTeam
        self.shift = shiftTeam.shift

    def priority(self):
        return self.shiftTeam.priority() + 1

    def situation(self):
        startTime = self.shiftTeam.shift.startTime
        endTime = self.shiftTeam.shift.endTime
        return startTime - 15 < now() < endTime

    # region late workers

    lateWorkers = someOf(Worker).withTimeEstimate(collectOnlyIfMaterialized=False)

    @lateWorkers.select
    def lateWorkers(self, worker, otherEnsembles):
        if self.potentiallyLate(worker):
            estimatedArrival = now() + self.lateWorkers.estimate(worker)
            return estimatedArrival > self.shift.endTime
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

    @lateWorkers.estimate.condition
    def arrived(self, worker):
        return worker.isAtFactory

    # endregion

    # region select standbys for late workers

    standbys = someOf(Worker)

    @standbys.select
    def standbys(self, worker, otherEnsembles):
        # TODO: not in other shift
        return worker in self.shift.availableStandbys

    @standbys.cardinality
    def standbys(self):
        return 0, len(self.lateWorkers)

    # endregion

    def actuate(self):
        if len(self.standbys) < len(self.lateWorkers):
            raise RuntimeError("Not enough standbys")  # TODO: replace error with a notification?

        self.shift.cancelled.append(self.lateWorkers)
        self.shift.calledStandbys.append(self.standbys)
        # TODO: notify
