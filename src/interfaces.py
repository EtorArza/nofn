import numpy as np
import numpy.typing as npt
import csv

class policy_nn:

    def __init__(self, n_in: int, n_out: int):
        raise NotImplementedError()

    def get_output(input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        raise NotImplementedError()



class optimization_strat:
    
    def __init__(self, n_in: int, n_out: int, seed: int):
        raise NotImplementedError()

    def show(self) -> policy_nn:
        raise NotImplementedError()

    def tell(self, objective_value: float) -> None:
        raise NotImplementedError()
    
    def log(self, resfilepath, f, evaluations, steps, time):
        import os
        file_exists = os.path.exists(resfilepath)
        with open(resfilepath, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(['f', 'evaluations', 'steps', 'time'])
            csvwriter.writerow([f, evaluations, steps, time])

