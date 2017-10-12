import numpy as np
from typing import Generator
from random import uniform

from dataset import DataSet


class Sphere:
    def __init__(self, cluster_units: int, epochs: int, data_set: DataSet):
        self.cluster_units = cluster_units  # Cantidad de cluster units que vamos a tener
        self.epochs = epochs  # Épocas que el algoritmo correrá
        self.data_set = data_set  # Diccionario de Data Sets
        self.weights = None  # Vector de pesos declarado como None
        self.learning_rate = 1  # Tasa de aprendizaje inicial
        self.decay = 0.005  # Decay para la tasa de aprendizaje. Sacado del primer ejemplo de Tutorial003
        self.winning_units = {
            20: {
                'p10': list(),
                'p20': list(),
                'p30': list()
            },
            50: {
                'p10': list(),
                'p20': list(),
                'p30': list()
            },
            100: {
                'p10': list(),
                'p20': list(),
                'p30': list()
            },
        }

        # p10, p20, p30 son tuplas.
        # En las cuales el índice 0 indica el cluster F1 y el índice 1 el cluster F2
        self.p10 = self._build_tdata_clusters(data_set.P10)
        self.p20 = self._build_tdata_clusters(data_set.P20)
        self.p30 = self._build_tdata_clusters(data_set.P30)

        # Validar la forma de las matrices
        assert self.p10[0].shape == (10, 10, 3) and self.p10[1].shape == (10, 10, 3)
        assert self.p20[0].shape == (10, 10, 3) and self.p20[1].shape == (10, 10, 3)
        assert self.p30[0].shape == (10, 10, 3) and self.p30[1].shape == (10, 10, 3)

        # Inicializar vector de pesos
        self._init_weight_array()

    def run(self):
        epoch = 0

        while epoch < self.epochs:
            def _iterate_data(data: tuple, cluster: int) -> Generator:
                for r in range(0, 10):
                    for c in range(0, 10):
                        yield data[cluster][r][c], (cluster, r, c)

            # ----------
            # | P10    |
            # ----------
            for e, pos_tuple in _iterate_data(self.p10, 0):  # F1
                weight_index = self._calc_winning_unit(e)
                self._update_weight(e, weight_index)

            for e, pos_tuple in _iterate_data(self.p10, 1):  # F2
                weight_index = self._calc_winning_unit(e)
                self._update_weight(e, weight_index)

            # ----------
            # | P20    |
            # ----------
            for e, pos_tuple in _iterate_data(self.p20, 0):  # F1
                weight_index = self._calc_winning_unit(e)
                self._update_weight(e, weight_index)

            for e, pos_tuple in _iterate_data(self.p20, 1):  # F2
                weight_index = self._calc_winning_unit(e)
                self._update_weight(e, weight_index)

            # ----------
            # | P30    |
            # ----------
            for e, pos_tuple in _iterate_data(self.p30, 0):  # F1
                weight_index = self._calc_winning_unit(e)
                self._update_weight(e, weight_index)

            for e, pos_tuple in _iterate_data(self.p30, 1):  # F2
                weight_index = self._calc_winning_unit(e)
                self._update_weight(e, weight_index)

            self._update_learning_rate()  # Se terminó una época, actualizar tasa de aprendizaje
            epoch += 1

    def _get_neighbours(self, data: tuple, cluster: int, row: int, column: int, epoch: int) -> Generator:
        radius = self._get_radius(epoch)

        x = [row - radius + 1, row + radius + 1]
        y = [column - radius + 1, column + radius + 1]

        # No pueden ser negativos, mínimo 0.
        if x[0] < 0:
            x[0] = 0

        if y[0] < 0:
            y[0] = 0

        # No pueden ser mayor a 10, máximo 10.
        if x[1] > 10:
            x[1] = 10

        if y[1] > 10:
            y[1] = 10

        for i in range(x[0], x[1]):
            for j in range(y[0], y[1]):
                yield data[cluster][i][j]

    def _update_weight(self, input_data, wi):
        self.weights[wi] = self.weights[wi] + (1 - self.learning_rate) * (input_data - self.weights[wi])

    def _update_learning_rate(self):
        self.learning_rate = self.decay * self.learning_rate

    def _calc_winning_unit(self, input_data: list) -> int:
        distances = []

        for weight in self.weights:
            distances.append(self._calc_euclidean_distance(input_data, weight))

        # El índice de la distancia más alta es el mismo índice del vector
        # de pesos que vamos a actualizar.
        return distances.index(min(distances))

    @staticmethod
    def _calc_euclidean_distance(input_data: list, weight: list):
        # Calcular la distancia euclidiana de un vector de entrada vs un vector de pesos
        def _pow() -> Generator:
            for _i, w in enumerate(weight):
                yield (w - input_data[_i]) ** 2

        return sum(_pow())

    def _build_tdata_clusters(self, data: list) -> tuple:
        f1_cluster = self._get_cluster(data[0][:100], data)  # Los 100 primeros
        f2_cluster = self._get_cluster(data[0][100:], data, add_to_index=100)  # Los 100 últimos

        return np.array(f1_cluster), np.array(f2_cluster)

    def _init_weight_array(self):
        _w = []
        for i in range(0, self.cluster_units):
            _w.append(self._get_cluster_unit())

        self.weights = np.array(_w)

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
            return 1

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
