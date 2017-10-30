from pathlib import Path
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
    som = Som(starting_radius=5, shape=(10, 10, 3), grid_type=0)

    # P10
    som.train(data.P10.transpose(), epochs=1000)
    som.plot()

    # P20
    som.train(data.P20.transpose(), epochs=1000)
    som.plot()

    # P30
    som.train(data.P30.transpose(), epochs=1000)
    som.plot()


def run_rgb():
    return


if __name__ == '__main__':
    if experiment == 0:
        run_sphere()
    else:
        run_rgb()
