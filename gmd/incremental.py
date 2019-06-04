from collections import deque
import numpy as np
from libgmdc import subspace_slice, set_seed, kstest, subspace_slice_oldest
from .incsortedindex import IncSortedIndex


class IncSubspaceContrast:
    def __init__(self, data, subspace, ref_dim, iterations=100, alpha=0.1, seed=1234):
        self.subspace = subspace
        self.ref_dim = np.int32(ref_dim)
        self.alpha = alpha

        self.sorted_index = IncSortedIndex(data)

        self.seed = seed
        set_seed(seed)

        self.iterations = iterations
        self.init_result()

    def insert_and_shift(self, new_point):
        self.sorted_index.del_and_ins_sorted(new_point)
        return self.shift(new_point)

    def shift(self, new_point):
        pass

    def init_result(self):
        self.res = np.zeros((self.iterations, 1))
        for i in range(self.iterations):
            curr_slice = subspace_slice(
                self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
            )
            self.res[i] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])


class EvictedSubspaceContrast(IncSubspaceContrast):
    def __init__(self, data, subspace, ref_dim, iterations=100, alpha=0.1, seed=1234):
        super().__init__(data, subspace, ref_dim, iterations, alpha, seed)
        self.len_evicted = []
        self.variances = []

    def init_result(self):
        self.res = np.zeros((self.iterations, 2))
        for i in range(self.iterations):
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
    def __init__(
        self, data, subspace, ref_dim, iterations=100, alpha=0.1, seed=1234, k=10
    ):
        super().__init__(data, subspace, ref_dim, iterations, alpha, seed)
        self.k = k

    def shift(self, new_point):
        for i in range(self.k):
            curr_slice = subspace_slice(
                self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
            )
            self.res.append(
                kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])
            )
        return np.mean(self.res)

    def init_result(self):
        res = np.zeros((self.iterations, 1))
        for i in range(self.iterations):
            curr_slice = subspace_slice(
                self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
            )
            res[i] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])
        self.res = deque(res, self.iterations)


class OriginalGMDSubspaceContrast(IncSubspaceContrast):
    def shift(self, new_point):
        for i in range(self.iterations):
            curr_slice = subspace_slice(
                self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
            )
            self.res[i] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])
        return np.mean(self.res)


class ReplaceRandomSubspaceContrast(IncSubspaceContrast):
    def __init__(
        self, data, subspace, ref_dim, iterations=100, alpha=0.1, seed=1234, draws=50
    ):
        super().__init__(data, subspace, ref_dim, iterations, alpha, seed)
        self.draws = draws

    def shift(self, new_point):
        to_replace = np.random.randint(0, self.iterations, self.draws)
        for i in to_replace:
            curr_slice = subspace_slice(
                self.sorted_index.sorted, self.subspace, self.ref_dim, self.alpha
            )
            self.res[i] = kstest(curr_slice, self.sorted_index.sorted[:, self.ref_dim])

        return np.mean(self.res)

