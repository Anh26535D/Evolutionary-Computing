import numpy as np


class BaseTSPSolver():
    
    def __init__(self) -> None:
        pass

    def read_dmatrix(self, file_path) -> np.ndarray:
        with open(file_path, 'r') as rfile:
            data = [[int(x) for x in line.split()] for line in rfile]
            return np.array(data)
        