import numpy as np
from random import uniform

from dataset import DataSet


class Sphere:
    def __init__(self, cluster_units: int, epochs: int, data_set: DataSet):
        self.cluster_units = cluster_units  # Cantidad de cluster units que vamos a tener
        self.epochs = epochs  # Épocas que el algoritmo correrá
        self.data_set = data_set  # Diccionario de Data Sets
        self.weight = None

        # p10, p20, p30 son tuplas.
        # En las cuales el índice 0 indica el cluster F1 y el índice 1 el cluster F2
        self.p10 = self._build_tdata_clusters(data_set.P10)
        self.p20 = self._build_tdata_clusters(data_set.P20)
        self.p30 = self._build_tdata_clusters(data_set.P30)

        # Validar la forma de las matrices
        assert self.p10[0].shape == (10, 10, 3) and self.p10[1].shape == (10, 10, 3)
        assert self.p20[0].shape == (10, 10, 3) and self.p20[1].shape == (10, 10, 3)
        assert self.p30[0].shape == (10, 10, 3) and self.p30[1].shape == (10, 10, 3)

        # Inicializar arreglo de pesos
        self._init_weight_array()

    def run(self):
        epoch = 0

        while epoch < self.epochs:
            def _iterate_data_tuple(data: tuple):
                counter = 0
                for cluster in range(0, 2):  # Clusters F1 y F2
                    for r in range(0, 10):  # Los cluster ya fueron separados en una tupla, los índices son [0,99]
                        for c in range(0, 10):
                            print(data[cluster][r][c])

            _iterate_data_tuple(self.p10)
            epoch += 1

    def _build_tdata_clusters(self, data: list) -> tuple:
        f1_cluster = self._get_cluster(data[0][:100], data)  # Los 100 primeros
        f2_cluster = self._get_cluster(data[0][100:], data, add_to_index=100)  # Los 100 últimos

        return np.array(f1_cluster), np.array(f2_cluster)

    def _init_weight_array(self):
        _w = []
        for i in range(0, self.cluster_units):
            _w.append(self._get_cluster_unit())

        self.weight = np.array(_w)

    @staticmethod
    def _get_cluster_unit():
        return np.array([uniform(0.9, 1.0), uniform(0.9, 1.0), uniform(0.9, 1.0)])

    @staticmethod
    def _get_radius(epoch: int) -> int:
        if epoch == 0:
            return 5
        elif epoch == 19:
            return 4
        elif epoch == 39:
            return 3
        elif epoch == 59:
            return 2
        elif epoch == 79:
            return 1
        else:
            return 0

    @staticmethod
    def _get_cluster(cluster: list, data: list, add_to_index: int = 0) -> list:
        _cluster = []
        _regrouped_cluster = []

        for i, _ in enumerate(cluster):
            if add_to_index == 0:
                _cluster.append([data[0][i], data[1][i], data[2][i]])
            else:
                _i = i + add_to_index
                _cluster.append([data[0][_i], data[1][_i], data[2][_i]])

        _group = []
        for i, _ in enumerate(_cluster):
            _group.append(_cluster[i])

            if (i + 1) % 10 == 0:
                _regrouped_cluster.append(_group)
                _group = []

        del _group, _cluster

        return _regrouped_cluster
