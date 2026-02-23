from typing import Optional, Tuple

import numpy as np
from joblib import load, dump
from sklearn.tree import DecisionTreeClassifier

from pysr import PySRRegressor 


# Wrapper around our extracted decision tree, mostly so that we can use the sb policy evaluator
class PySRWrapper:
    def __init__(self, sr: PySRRegressor):
        self.sr = sr

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.sr.predict(observation), None

    @classmethod
    def load(cls, path: str):
        clf = load(path)
        return PySRWrapper(clf)

    def save(self, path: str):
        print(f"Saving to\t{path}")
        dump(self.tree, path)

    def print_info(self):
        # TODO: implement info print here. Complexity....
        print("Ain't been done")
        # print(f"Max depth:\t{self..get_depth()}")
        # print(f"# Leaves:\t{self.tree.get_n_leaves()}")