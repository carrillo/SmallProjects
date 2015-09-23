__author__ = 'carrillo'

import numpy as np

class UpdateParameter(object):
    def __init__(self, name, start, stop):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def float32(x):
        return np.cast['float32'](x)


    def __call__(self, nn, train_history):
        """
        1. Generate array of parameters from start to stop over all epochs.
        2. Assign the epoch specific value at each epoch.
        :param nn:
        :param train_history:
        :return:
        """
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        new_value = np.cast['float32'](self.ls[train_history[-1]['epoch']-1])
        #epoch = train_history[-1]['epoch']
        #new_value = self.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
