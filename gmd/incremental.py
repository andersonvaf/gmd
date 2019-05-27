import numpy as np
from libgmdc import subspace_slice, set_seed, kstest, subspace_slice_oldest
from .incsortedindex import IncSortedIndex


class IncSubspaceContrast:
    def __init__(self, data, subspace, ref_dim, alpha=0.1, seed=1234):
        self.subspace = subspace
        self.ref_dim = np.int32(ref_dim)
        self.alpha = alpha
        self.window_size = len(data)

        self.sorted_index = IncSortedIndex(data)

        self.seed = seed
        set_seed(seed)

        self.init_result()

    def insert_and_shift(self, new_point):
        self.sorted_index.del_and_ins_sorted(new_point)
        return self.shift(new_point)

    def shift(self, new_point):
        pass

    def init_result(self):
        self.res = np.zeros((self.window_size, 1))
        for i in range(self.window_size):
            curr_slice = subspace_slice(
                self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
            )
            self.res[i] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])


class EvictedSubspaceContrast(IncSubspaceContrast):
    def __init__(self, data, subspace, ref_dim, alpha=0.1, seed=1234):
        super().__init__(data, subspace, ref_dim, alpha, seed)
        self.len_evicted = []
        self.variances = []

    def init_result(self):
        self.res = np.zeros((self.window_size, 2))
        for i in range(self.window_size):
            curr_slice, oldest = subspace_slice_oldest(
                self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
            )
            self.res[i, 1] = oldest
            self.res[i, 0] = kstest(
                curr_slice, self.sorted_index.sorted[:, self.ref_dim]
            )

    def shift(self, new_point):
        self.res[:, 1] -= 1
        evicted = np.where(self.res[:, 1] == -1)[0]
        self.len_evicted.append(len(evicted))
        self.variances.append(self.res[evicted, 0])
        for i in evicted:
            curr_slice, oldest = subspace_slice_oldest(
                self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
            )
            self.res[i, 1] = oldest
            self.res[i, 0] = kstest(
                curr_slice, self.sorted_index.sorted[:, self.ref_dim]
            )
        return np.mean(self.res[:, 0])


class ReplaceOldestSubspaceContrast(IncSubspaceContrast):
    def shift(self, new_point):
        for i in range(len(self.res) - 1):
            self.res[i] = self.res[i + 1]
        curr_slice = subspace_slice(
            self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
        )
        self.res[-1] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])
        return np.mean(self.res)


class OriginalGMDSubspaceContrast(IncSubspaceContrast):
    def shift(self, new_point):
        for i in range(self.window_size):
            curr_slice = subspace_slice(
                self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
            )
            self.res[i] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])
        return np.mean(self.res)


class ReplaceRandomSubspaceContrast(IncSubspaceContrast):
    def __init__(self, data, subspace, ref_dim, alpha=0.1, seed=1234, draws=50):
        super().__init__(data, subspace, ref_dim, alpha, seed)
        self.draws = draws

    def shift(self, new_point):
        to_replace = np.random.randint(0, self.window_size, self.draws)
        for i in to_replace:
            curr_slice = subspace_slice(
                self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
            )
            self.res[i] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])

        return np.mean(self.res)

