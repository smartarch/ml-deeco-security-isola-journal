from components import Shift, Worker
from helpers import allow, now
from ml_deeco.simulation import Ensemble, someOf


class ShiftTeam(Ensemble):

    shift: Shift

    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    # TODO
    workers = someOf(Worker)


class AccessToFactory(Ensemble):

    # parent ensemble
    shiftTeam: ShiftTeam

    def __init__(self, shiftTeam):
        super().__init__()
        self.shiftTeam = shiftTeam
        self.factory = shiftTeam.shift.workPlace.factory

    def priority(self):
        return self.shiftTeam.priority() + 1

    def situation(self):
        startTime = self.shiftTeam.shift.startTime
        endTime = self.shiftTeam.shift.endTime
        return startTime - 30 < now() < endTime + 30

    def actuate(self):
        allow(self.shiftTeam.workers, "enter", self.factory)


class AccessToDispenser(Ensemble):

    # parent ensemble
    shiftTeam: ShiftTeam

    def __init__(self, shiftTeam):
        super().__init__()
        self.shiftTeam = shiftTeam
        self.dispenser = shiftTeam.shift.workPlace.factory.dispenser

    def priority(self):
        return self.shiftTeam.priority() + 1

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
        return self.shiftTeam.priority() + 1

    def situation(self):
        startTime = self.shiftTeam.shift.startTime
        endTime = self.shiftTeam.shift.endTime
        return startTime - 30 < now() < endTime + 30

    workers = someOf(Worker)  # subset of self.shiftTeam.workers

    @workers.select
    def workers(self, worker, otherEnsembles):
        return worker in self.shiftTeam.workers and worker.hasHeadGear

    # TODO: can we implement unlimited cardinality? -> also useful for 'drone charging'
    @workers.cardinality
    def workers(self):
        return len(self.shiftTeam.workers)

    def actuate(self):
        allow(self.workers, "enter", self.workPlace)


class NotificationAboutWorkersThatArePotentiallyLate(Ensemble):
    pass


class LateWorkersReplacement(Ensemble):
    pass
