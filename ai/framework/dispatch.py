from typing import List

from framework.contracts import HookAdapter


class Dispatcher(HookAdapter):
    def __init__(self, callbacks: List[HookAdapter] = None):
        self.callbacks = callbacks or []

    def add_callback(self, cb: HookAdapter):
        self.callbacks.append(cb)

    def on_train_start(self, state):
        for cb in self.callbacks:
            cb.on_train_start(state)

    def on_epoch_start(self, state):
        for cb in self.callbacks:
            cb.on_epoch_start(state)

    def on_batch_start(self, state):
        for cb in self.callbacks:
            cb.on_batch_start(state)

    def on_batch_end(self, state):
        for cb in self.callbacks:
            cb.on_batch_end(state)

    def on_epoch_end(self, state):
        for cb in self.callbacks:
            cb.on_epoch_end(state)

    def on_train_end(self, state):
        for cb in self.callbacks:
            cb.on_train_end(state)
