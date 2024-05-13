from typing import Dict,Any
from numpy.typing import NDArray
def assign_env_config(self:Any, kwargs:Dict):
    """
    Set self.key = value, for each key in kwargs
    """
    for key, value in kwargs.items():
        setattr(self, key, value)


def dist(loc1:NDArray, loc2:NDArray):
    return ((loc1[:, 0] - loc2[:, 0]) ** 2 + (loc1[:, 1] - loc2[:, 1]) ** 2) ** 0.5