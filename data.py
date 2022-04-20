import collections
import os
import time
import os

import matplotlib.pyplot as plt
import numpy as np

import nn


class Dataset(object):
    def __init__(self) -> None:
        dict = getDict()
        data = dict[b'data']
        labels = dict[b'labels']

        labels_one_hot = np.zeros((len(data), 10))
        labels_one_hot[range(len(data)), labels] = 1

        self.x = data
        self.y = labels_one_hot

    def iterate_once(self, batch_size):
        assert isinstance(batch_size, int) and batch_size > 0, (
            "Batch size should be a positive integer, got {!r}".format(
                batch_size))
        assert self.x.shape[0] % batch_size == 0, (
            "Dataset size {:d} is not divisible by batch size {:d}".format(
                self.x.shape[0], batch_size))
        index = 0
        while index < self.x.shape[0]:
            x = self.x[index:index + batch_size]
            y = self.y[index:index + batch_size]
            yield nn.Constant(x), nn.Constant(y)
            index += batch_size

    def iterate_forever(self, batch_size):
        while True:
            yield from self.iterate_once(batch_size)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getDict():
    file = "data/data_batch_1"
    dict = unpickle(file)
    return dict

data = getDict()
x = data[b'data']
y = data[b'labels']
dataset = Dataset()
for example in dataset.iterate_forever(100):
    x = example[0]
    y = example[1]