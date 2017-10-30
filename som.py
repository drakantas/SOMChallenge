import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Som:
    def __init__(self, starting_radius: int = 5, shape: tuple = (10, 10, 3), grid_type: int = 0):
        self.weight = np.random.rand(*shape)
        self.grid_type = grid_type
        self.radius = starting_radius
        self.learning_rate = 1
        self._latest_trained_dataset = None

    def train(self, data, epochs: int = 100):
        assert len(data.shape) == 2
        assert data.shape[1] == 3

        self._latest_trained_dataset = data

        counter = 0

        while counter < epochs:
            for e in data:
                winner = None

                for row, _ in enumerate(self.weight):
                    for column, __ in enumerate(self.weight):
                        if winner is None:
                            winner = (self._euclidean_distance(e, self.weight[row][column]), row, column)
                        else:
                            euclidean_distance = self._euclidean_distance(e, self.weight[row][column])

                            if euclidean_distance > winner[0]:
                                winner = (euclidean_distance, row, column)

                neighbourhood = self._find_neighbourhood(winner[1], winner[2])

                for row in range(*neighbourhood[0]):
                    for column in range(*neighbourhood[1]):
                        self._update_weight(e, row, column)

            # Si el contador más uno es un múltiplo de 20 y el radio es mayor a 1, reducimos
            # el radio en 1.
            if (counter + 1) % 20 == 0 and self.radius > 1:
                self.radius -= 1

            counter += 1

    def plot(self):
        if self._latest_trained_dataset is None:
            print('No hay data disponible para plot(), entrena la red neuronal primero antes de usar este '
                  'método.')
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self._latest_trained_dataset[:, 0]
        y = self._latest_trained_dataset[:, 1]
        z = self._latest_trained_dataset[:, 2]

        ax.plot_trisurf(x, y, z, linewidth=0)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        weights = np.array(list(self._get_plottable_weights()))

        ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='r', marker='o')

        plt.show()

    def _get_plottable_weights(self):
        for row in self.weight:
            for e in row:
                yield e

    def _find_neighbourhood(self, row: int, column: int):
        if self.grid_type == 0:
            return self._rectangular_neighbourhood(row, column)
        elif self.grid_type == 1:
            return self._hexagonal_neighbourhood(row, column)

        raise ValueError('Tipo de grilla no soportado.')

    def _rectangular_neighbourhood(self, row: int, column: int):
        lookup_range = ([row - self.radius, row + 1 + self.radius],
                        [column - self.radius, column + 1 + self.radius])

        for i, _range in enumerate(lookup_range):
            if _range[0] < 0:
                lookup_range[i][0] = 0

            if _range[1] > 10:
                lookup_range[i][1] = 10

        return lookup_range

    def _hexagonal_neighbourhood(self, row: int, column: int):
        pass

    def _update_weight(self, input_array, row: int, column: int):
        self.weight[row][column] += self.learning_rate * (input_array - self.weight[row][column])

    @staticmethod
    def _euclidean_distance(input_array, weight_array):
        assert input_array.shape == weight_array.shape == (3,)

        distance = 0.0

        for i in range(0, 3):
            distance += (input_array[i] - weight_array[i]) ** 2

        return distance
