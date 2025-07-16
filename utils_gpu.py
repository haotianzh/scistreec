import popgen
import numpy as np
import cupy as cp
import scipy as sp
from cupyx.scipy.special import logsumexp, log_softmax
from time import time
import popgen.utils


"""
    Log of dot product of a mat and a vec.

    Args:
        mat: (num_site, num_state, num_state)
        vec: (num_site, 1, num_state)

    Returns:
        (num_site, num_state)
"""
def log_mat_vec_mul(mat, vec):
    # assert mat.shape[1] == vec.shape[0], "shape is not match."
    v = mat + vec # broadcasting
    # print(v[0].round(2), logsumexp(v, axis=-1).round(2)[0])
    return logsumexp(v, axis=-1)

"""
    Log of maximum of pointwise multiplication. 
    Consider all sites when picking the maximal one.

    Args:
        mat: (num_site, num_state, num_state)
        vec: (num_site, 1, num_state)
    
    Returns:
        (num_site, num_state)
"""
def log_mat_vec_max(mat, vec):
    # assert mat.shape[1] == vec.shape[0], "shape is not match."
    v = mat + vec # broadcasting
    # sum_v = v.sum(axis=0) # (num_state, num_state)
    # arg = sum_v.argmax(axis=-1) # (num_state)
    # arg = arg.reshape(1, -1, 1) # (1, num_state, 1)
    # val = np.take_along_axis(v, arg, axis=-1) # (num_site, num_state)
    val = v.max(axis=-1)
    arg = v.argmax(axis=-1)
    return val, arg

def log_matmul(logA, logB):
    """
    Compute matrix multiplication in log space: log(A @ B)   
    logA: (N, p, p) array 
    logB: (N, p, p) array 
    Returns:
    logC: (N, p, p) array, where logC = log(A @ B)
    """
    # logA has shape (N, p, p)
    # logB has shape (N, p, p)
    # Compute log(A @ B) using broadcasting
    product = logsumexp(logA[:, :, :, cp.newaxis] + logB[:, cp.newaxis, :, :], axis=2)
    return product


def timimg(func):
    """
    A decorator to time the function.
    """
    def wrapper(*args, **kwargs):
        start_time = cp.cuda.Event()
        end_time = cp.cuda.Event()
        start_time.record()
        result = func(*args, **kwargs)
        end_time.record()
        end_time.synchronize()
        elapsed_time = cp.cuda.get_elapsed_time(start_time, end_time)
        print(f"Elapsed time: {elapsed_time} ms")
        return result
    return wrapper

def cpu_time(func):
    """
    A decorator to time the function.
    """
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        elapsed_time = end_time - start_time
        # print(f"Function {func.__name__}, Elapsed time: {elapsed_time} s")
        return result
    return wrapper

@cpu_time
def neighbor_joining_np(disMatrix):
    import numpy as np
    D = np.array(disMatrix, dtype = float)
    n = D.shape[0]
    clusters = [i for i in range(n)]
    adj = [[] for i in range(n)]
    nodes = [popgen.Node(name=str(i+1)) for i in range(n)]
    if len(D) <= 1:
        return adj
    while True:
        if n == 2:
            adj[len(adj)-1].append((len(adj)-2, D[0][1]))
            adj[len(adj)-2].append((len(adj)-1, D[0][1]))
            break
        totalDist = np.sum(D, axis = 0)
        D1 = (n-2) * D
        D1 = D1 - totalDist
        D1 = D1 - totalDist.reshape((n, 1))
        np.fill_diagonal(D1, 0.)
        #print(D1)
        index = np.argmin(D1)
        i = index // n
        j = index % n
        if j < i:
            i, j = j, i
        delta = (totalDist[i] - totalDist[j])/(n-2)
        li = (D[i, j]+delta)/2
        lj = (D[i, j]-delta)/2
        d_new = (D[i, :]+D[j, :]-D[i, j])/2
        D = np.insert(D, n, d_new, axis = 0)
        d_new = np.insert(d_new, n, 0., axis = 0)
        D = np.insert(D, n, d_new, axis = 1)
        D = np.delete(D, [i, j], 0)
        D = np.delete(D, [i, j], 1)
        m = len(adj)
        node_new = popgen.Node()
        nodes[i].set_parent(node_new)
        nodes[j].set_parent(node_new)
        node_new.add_child(nodes[i])
        node_new.add_child(nodes[j])
        nodes.remove(nodes[j])
        nodes.remove(nodes[i])
        nodes.append(node_new)

        adj.append([])
        adj[m].append((clusters[i], li))
        adj[clusters[i]].append((m, li))
        adj[m].append((clusters[j], lj))
        adj[clusters[j]].append((m, lj))
        if i < j:
            del clusters[j]
            del clusters[i]
        else:
            del clusters[i]
            del clusters[j]
        clusters.append(m)
        n -= 1
    # merge the last two nodes
    root = popgen.Node()
    nodes[0].set_parent(root)
    nodes[1].set_parent(root)
    root.add_child(nodes[1])
    root.add_child(nodes[0])
    return popgen.utils.from_node(root)

@cpu_time
def neighbor_joining(disMatrix):
    '''
    Haotian redistributed this code from the following link:

    In 1987, Naruya Saitou and Masatoshi Nei developed the neighbor-joining algorithm for evolutionary tree reconstruction. Given 
    an additive distance matrix, this algorithm, which we call NeighborJoining, finds a pair of neighboring leaves and substitutes 
    them by a single leaf, thus reducing the size of the tree. NeighborJoining can thus recursively construct a tree fitting the 
    additive matrix. This algorithm also provides a heuristic for non-additive distance matrices that performs well in practice.

    The central idea of NeighborJoining is that although finding a minimum element in a distance matrix D is not guaranteed to 
    yield a pair of neighbors in Tree(D), we can convert D into a different matrix whose minimum element does yield a pair of 
    neighbors. First, given an n × n distance matrix D, we define TotalDistanceD(i) as the sum ∑1≤k≤n Di,k of distances from 
    leaf i to all other leaves. The neighbor-joining matrix D* (see below) is defined such that for any i and j, D*i,i = 0 
    and D*i,j = (n - 2) · Di,j - TotalDistanceD(i) - TotalDistanceD(j).

    Implement NeighborJoining.
        Input: An integer n, followed by an n x n distance matrix.
        Output: An adjacency list for the tree resulting from applying the neighbor-joining algorithm. Edge-weights should be 
        accurate to two decimal places (they are provided to three decimal places in the sample output below).

    Note on formatting: The adjacency list must have consecutive integer node labels starting from 0. The n leaves must be 
    labeled 0, 1, ..., n - 1 in order of their appearance in the distance matrix. Labels for internal nodes may be labeled 
    in any order but must start from n and increase consecutively.

    Sample Input:
    0	23	27	20
    23	0	30	28
    27	30	0	30
    20	28	30	0
    Sample Output:
    0->4:8.000
    1->5:13.500
    2->5:16.500
    3->4:12.000
    4->5:2.000
    4->0:8.000
    4->3:12.000
    5->1:13.500
    5->2:16.500
    5->4:2.000
    '''
    D = cp.array(disMatrix, dtype=cp.float32)
    n = D.shape[0] 
    nodes = [popgen.Node(identifier=str(i)) for i in range(n)]
    while True:
        size = D.shape[0]
        if size == 2:
            break

        totalDist = cp.sum(D, axis=0)
        D1 = (size - 2) * D - totalDist - totalDist.reshape((size, 1))
        cp.fill_diagonal(D1, 0.0)
        index = cp.argmin(D1)
        i = int(index // size)
        j = int(index % size)
        if i > j:
            i, j = j, i
        delta = (totalDist[i] - totalDist[j]) / (size - 2)
        li = (D[i, j] + delta) / 2
        lj = (D[i, j] - delta) / 2
        d_new = (D[i, :] + D[j, :] - D[i, j]) / 2
        d_new = cp.delete(d_new, [i, j])  
        d_new = cp.append(d_new, 0.0)
        mask = cp.ones(size, dtype=bool)
        mask[[i, j]] = False
        D = D[mask][:, mask] 
        D = cp.vstack((D, d_new[:-1].reshape(1, -1)))
        new_col = cp.append(d_new[:-1], 0.0).reshape(-1, 1)
        D = cp.hstack((D, new_col))
        node_new = popgen.Node()
        nodes[i].set_parent(node_new)
        nodes[j].set_parent(node_new)
        node_new.add_child(nodes[i])
        node_new.add_child(nodes[j])
        nodes.remove(nodes[j])
        nodes.remove(nodes[i])
        nodes.append(node_new)
        root = popgen.Node()
    nodes[0].set_parent(root)
    nodes[1].set_parent(root)
    root.add_child(nodes[1])
    root.add_child(nodes[0])

    return popgen.utils.from_node(root)



def read_scistree2_input(filename):
    arr = []
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            row = line.strip().split()[1:]
            row = [float(v) for v in row]
            arr.append(row)
    return np.array(arr, dtype=np.float64)


if __name__ == '__main__':
    disMatrix = [
    [0,	23,	27,	20],
    [23,	0,	30,	28],
    [27,	30,	0,	30],
    [20,	28,	30,	0],
    ]

    tree = neighbor_joining_np(disMatrix)
    print(tree)

    a = neighbor_joining(disMatrix)
    print(a)