from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataset import DataSet
from som import Som


# 0 = Mapear data de esfera
# 1 = What if? Mapear data RGB
experiment = 0


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
    som.train(data.P10.transpose(), epochs=100)
    som.plot('Dataset P10, iteraciones=100')

    # P20
    som.train(data.P20.transpose(), epochs=100)
    som.plot('Dataset P20, iteraciones=200')

    # P30
    som.train(data.P30.transpose(), epochs=100)
    som.plot('Dataset P30, iteraciones=300')


def run_rgb():
    som = Som()

    #som.train(data.RGB.transpose(), epochs=1000)




if __name__ == '__main__':
    if experiment == 0:
        run_sphere()
    else:
        run_rgb()
