from hanser.train.callbacks import Callback


class PrintGenotype(Callback):

    def __init__(self, from_epoch):
        super().__init__()
        self.from_epoch = from_epoch

    def after_epoch(self, state):
        epoch = self.learner.epoch
        if epoch + 1 < self.from_epoch:
            return
        p = """Genotype(
    normal=[
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
    ], normal_concat=[2, 3, 4, 5],
    reduce=[
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
    ], reduce_concat=[2, 3, 4, 5],
)"""
        g = self.learner.model.genotype()
        print(p % (tuple(g.normal) + tuple(g.reduce)))


class TrainArch(Callback):

    def __init__(self, from_epoch):
        super().__init__()
        self.from_epoch = from_epoch

    def begin_epoch(self, state):
        epoch = self.learner.epoch + 1
        self.learner.train_arch = epoch >= self.from_epoch
        if epoch == self.from_epoch:
            self.learner.remake()


class TauSchedule(Callback):

    def __init__(self, tau_max, tau_min):
        super().__init__()
        self.tau_max, self.tau_min = tau_max, tau_min

    def begin_epoch(self, state):
        epoch = self.learner.epoch
        epochs = state['epochs']
        tau_max = self.tau_max
        tau_min = self.tau_min
        tau = tau_max - (tau_max - tau_min) * (epoch / epochs)
        self.learner.model.tau.assign(tau)