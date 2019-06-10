import numpy as np


def jaccard_coeff(s1: set, s2: set) -> float:
    return len(s1.intersection(s2)) / len(s1.union(s2))


def similarity_matrix(df1, df2):
    assert df1.shape == df2.shape
    assert df1.columns.tolist() == df2.columns.tolist()

    matrix = np.zeros(df1.shape)
    for i, col in enumerate(df1.columns):
        matrix[:, i] = [jaccard_coeff(*s) for s in zip(df1[col], df2[col])]
    return matrix
