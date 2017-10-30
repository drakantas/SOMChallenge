import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataset import DataSet
from som import Som


# 0 = Mapear data de esfera
# 1 = What if? Mapear data RGB
experiment = 1


if experiment == 0:
    data = DataSet(Path('./datasets/sphere_data.mat'))
elif experiment == 1:
    data = DataSet(Path('./datasets/rgb_data.mat'))
else:
    raise Exception('experiment debe ser 0 ó 1')


def run_sphere():
    # Inicializamos SOM
    som = Som()  # Radio 5, y forma de vector de pesos 10x10 son asumidos por defecto

    # P10
    som.train(data.P10.transpose(), epochs=200)
    som.plot('Dataset P10, iteraciones=200')

    # P20
    som.train(data.P20.transpose(), epochs=400)
    som.plot('Dataset P20, iteraciones=400')

    # P30
    som.train(data.P30.transpose(), epochs=600)
    som.plot('Dataset P30, iteraciones=600')


def run_rgb():
    som = Som(learning_rate=0.25, decay=0.0005)

    som.train(data.RGB.transpose(), epochs=10)

    title = 'Data RGB'

    fig = plt.figure()
    fig.canvas.set_window_title(title)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')

    weights = np.array(list(som._get_plottable_weights()))

    ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c='r', marker='o')

    plt.show()


if __name__ == '__main__':
    if experiment == 0:
        run_sphere()
    else:
        run_rgb()
