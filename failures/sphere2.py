import numpy as np
from random import choice
from somoclu import Somoclu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataset import DataSet


sphere = None


class Sphere:
    def __init__(self, data: DataSet):
        self.dimensions = (100, 100)  # Dimensiones de neuronas para la red
        self.net = Somoclu(*self.dimensions,
                           initialization='random', neighborhood='bubble', compactsupport=False)  # Red SOM
        self.data = data

    def train(self, data_set: str, first: bool = True, epochs: int = 200):
        if first:
            data = np.float32(getattr(self.data, data_set)).transpose()[:100].transpose()
        else:
            data = np.float32(getattr(self.data, data_set)).transpose()[100:].transpose()

        self.net.train(data, epochs=epochs, radius0=5, radiusN=1, scale0=1, scaleN=0.005)

    def plot(self):
        self.net.view_umatrix(bestmatches=True, labels=['BMU']*100)

    @staticmethod
    def _get_colours():
        options = ['red', 'green', 'blue', 'yellow']

        for i in range(0, 5):
            yield [choice(options)] * 20


def get_sphere(data: DataSet):
    global sphere

    if sphere is None:
        sphere = Sphere(data)

    return sphere
