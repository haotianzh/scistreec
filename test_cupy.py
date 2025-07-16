import cupy as cp

def pairwise_expected_distance(A):
    C, S, num_states = A.shape
    print(A.shape)
    # Expand dimensions to compute pairwise probabilities efficiently
    # A1 = A[:, :, cp.newaxis, :]  # Shape: (C, 1, S, 4)
    # expected_distances1 = cp.matmul(A1, 1-A1.transpose(0, 1, 3, 2))
    expected_distances1 = cp.einsum('ipq,jpq->ij', A, 1-A)
    print(expected_distances1.shape)
    return expected_distances1
    # expected_distances2 = cp.matmul(1-A1, A1.transpose(0, 1, 3, 2))
    # expected_distances = expected_distances1 + expected_distances2
    # pairwise_dist = expected_distances.squeeze()
    # Sum over sites to get final pairwise distances
    # pairwise_dist = cp.sum(expected_distances, axis=-1)  # Shape: (C, C)
    
    # return pairwise_dist

# Example usage
C, S = 4, 4  # 10 cells, 100 sites
A = cp.random.rand(C, S, 2)
A /= A.sum(axis=-1, keepdims=True)  # Normalize to make it a probability distribution
# print(A)
# # Compute pairwise distances
# pairwise_distances = pairwise_expected_distance(A)

import numpy as np 
mat = np.zeros([4, 4])
for i, c1 in enumerate(A):
    for j, c2 in enumerate(A):
        ss = 0
        for s1, s2 in zip(c1, c2):
           for k in range(len(s1)):
                ss += s1[k] * (1-s2[k]) + (1-s1[k])*s2[k]
        mat[i, j] = ss

print(mat)



print(2*pairwise_expected_distance(A))