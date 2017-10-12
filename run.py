from pathlib import Path
from dataset import DataSet


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
    return


def run_rgb():
    return


if __name__ == '__main__':
    if experiment == 0:
        run_sphere()
    else:
        run_rgb()
