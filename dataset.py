from pathlib import Path
from scipy.io import loadmat
from typing import Generator, Any


class DataSet(dict):
    """
        El motivo principal por el cual hacemos esta abstracción es para:
        - Deshacernos de la metadata innecesaria.
        - Interactuar con la data de manera más elegante:
            data = DataSet(Path('./datasets/sphere_data.mat'))
            data.P20 retornará el índice P20 de la matriz
    """
    def __init__(self, path: Path):
        super().__init__(self._load_data(path))

    def __getitem__(self, item: Any):
        raise KeyError('No soportado')

    def __setitem__(self, key: Any, value: Any):
        raise ValueError('No soportado')

    def __setattr__(self, key: str, value: Any):
        raise AttributeError('No soportado')

    def __getattr__(self, item: str):
        return super().__getitem__(item)

    @staticmethod
    def _load_data(path: Path) -> Generator:
        data = loadmat(str(path.resolve()))

        for k, v in data.items():
            if k in ('__header__', '__version__', '__globals__'):  # Ignorar metadata
                continue

            yield k, v
