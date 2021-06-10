import numpy as np
from pyts.datasets import make_cylinder_bell_funnel


class CbfCreator:
    def __init__(self, sample_amount):
        self.cbf_data_set,_ = make_cylinder_bell_funnel(n_samples=sample_amount)

    def get_sts_cbf_data_set(self):
        return to_sts_matrix(np.ravel(self.cbf_data_set), 128)


def to_sts_matrix(ts: np.array, w: int):
    shape = ts.shape[:-1] + (ts.shape[-1] - w + 1, w)
    strides = ts.strides + (ts.strides[-1],)
    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)