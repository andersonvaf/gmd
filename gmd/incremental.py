import numpy as np
from libgmdc import subspace_slice, set_seed, kstest, subspace_slice_oldest
from .incsortedindex import IncSortedIndex


class IncSubspaceContrast:
    def __init__(self, data, subspace, ref_dim, alpha=0.1, seed=1234):
        self.subspace = subspace
        self.ref_dim = np.int32(ref_dim)
        self.alpha = alpha
        self.window_size = len(data)
        self.res = np.zeros((self.window_size, 2))

        self.sorted_index = IncSortedIndex(data)

        self.seed = seed
        set_seed(seed)

        for i in range(self.window_size):
            curr_slice, oldest = subspace_slice_oldest(
                self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
            )
            self.res[i, 1] = oldest
            self.res[i, 0] = kstest(
                curr_slice, self.sorted_index.sorted[:, self.ref_dim]
            )

    def insert_and_shift(self, new_point, strategy_func, **args):
        self.sorted_index.del_and_ins_sorted(new_point)
        return strategy_func(self, **args)


def evicted_strategy(self):
    self.res[:, 1] -= 1
    if not hasattr(self, "len_evicted"):
        self.len_evicted = []
    if not hasattr(self, "variances"):
        self.variances = []

    evicted = np.where(self.res[:, 1] == -1)[0]
    self.len_evicted.append(len(evicted))
    self.variances.append(self.res[evicted, 0])
    for i in evicted:
        curr_slice, oldest = subspace_slice_oldest(
            self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
        )
        # print(sum(curr_slice))
        self.res[i, 1] = oldest
        self.res[i, 0] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])
    return np.mean(self.res[:, 0])


def replace_oldest_strategy(self):
    for i in range(len(self.res) - 1):
        self.res[i] = self.res[i + 1]
    curr_slice, oldest = subspace_slice_oldest(
        self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
    )
    self.res[-1, 1]
    self.res[-1, 0] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])
    return np.mean(self.res[:, 0])


def gmd_strategy(self):
    for i in range(self.window_size):
        curr_slice, oldest = subspace_slice_oldest(
            self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
        )
        self.res[i, 1] = oldest
        self.res[i, 0] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])
    return np.mean(self.res[:, 0])


def random_strategy(self, draws=100):
    to_replace = np.random.randint(0, self.window_size, draws)
    for i in to_replace:
        curr_slice, oldest = subspace_slice_oldest(
            self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
        )
        self.res[i, 1] = oldest
        self.res[i, 0] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])

    return np.mean(self.res[:, 0])

