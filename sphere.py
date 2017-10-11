import numpy as np
from dataset import DataSet


class Sphere:
    def __init__(self, cluster_units: int, epochs: int, data_set: DataSet):
        self.cluster_units = cluster_units  # Cantidad de cluster units que vamos a tener
        self.epochs = epochs  # Épocas que el algoritmo correrá
        self.data_set = data_set  # Diccionario de Data Sets

        p10 = self._build_tdata_clusters(data_set.P10)
        p20 = self._build_tdata_clusters(data_set.P20)
        p30 = self._build_tdata_clusters(data_set.P30)

    def run(self):
        epoch = 0

        while epoch < self.epochs:
            # ...

            epoch += 1

    def _build_tdata_clusters(self, data: list) -> tuple:
        f1_cluster = self._get_cluster(data[0][:100], data)  # Los 100 primeros
        f2_cluster = self._get_cluster(data[0][100:], data, add_to_index=100)  # Los 100 últimos

        return np.array(f1_cluster), np.array(f2_cluster)

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
