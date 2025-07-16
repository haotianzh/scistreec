import popgen
import transition_solver
import numpy as np 
import time
import sys
from scipy.special import logsumexp
from numpy.linalg import matrix_power


def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('time: ', end-start)
    return wrapper

@time_wrapper
def test1():
    a = np.arange(100 * 100).reshape((100, 100))
    b = np.array([a]* 10000)
    print(b.__sizeof__())
    return b 

@time_wrapper
def test2():
    a = np.arange(100 * 100).reshape((100, 100))
    
    b = np.vstack([a] * 10000)
    print(b.__sizeof__())
    return b 


@time_wrapper
def test3():
    a = np.arange(100 * 100).reshape((100, 100))
    b = np.tile(a, (10000, 1))
    print(b.__sizeof__())
    return b

@time_wrapper
def test4():
    a = np.arange(100 * 100).reshape((100, 100))
    b = np.repeat(a, 10000, axis=0).reshape(100, 10000, 100)
    c = np.ones([10000, 1, 100])
    b = b.transpose((1, 0 ,2))
    b = b + c
    print(b.__sizeof__())
    return b


def test5():
    a = np.arange(32).reshape(2, 4, 4)
    # indices = np.array([[[3], [2], [1], [1]]])
    # indices = np.broadcast_to(indices, (2, 4, 1))
    indices = np.array([0, 1, 2, 3])
    indices = indices.reshape(1, 4, 1)
    indices = np.broadcast_to(indices, (2, 4, 1))
    print(indices)
    # axis = 0
    # Ni, M, Nk = a.shape[:axis], a.shape[axis], a.shape[axis+1:]
    # J = indices.shape[axis]  # Need not equal M
    # out = np.empty(Ni + (J,) + Nk)
    # for ii in np.ndindex(Ni):
    #     for kk in np.ndindex(Nk):
    #         # print(ii, kk, np.s_[:,])
    #         a_1d       = a      [ii + np.s_[:,] + kk]
    #         indices_1d = indices[ii + np.s_[:,] + kk]
    #         out_1d     = out    [ii + np.s_[:,] + kk]
    #         print(a_1d)
    #         print(indices_1d)
    #         print('---')
    # # print(a[:, :, 0], indices[:, :, 0])
    # # print(a[:, :, 1])
    b = np.take_along_axis(a, indices, axis=-1)
    print(b.shape)
    # print(b.__sizeof__())

def log_space_product(A,B):
    Astack = np.stack([A]*A.shape[0]).transpose(2,1,0)
    Bstack = np.stack([B]*B.shape[1]).transpose(1,0,2)
    return logsumexp(Astack+Bstack, axis=0)

def log_power(a, times):
    res = np.eye(a.shape[0], a.shape[1])
    res = np.log(res)
    for i in range(times):
        res = log_space_product(res, a)
    return res 



def test6():
    a = np.random.rand(5, 5) / 10
    print(np.log(matrix_power(a, 5)))
    # print(log_space_product(np.log(a), np.log(a)))
    print(log_power(np.log(a), 5))
    


def log_matmul(logA, logB):
    """
    Compute matrix multiplication in log space: log(A @ B)
    
    logA: (N, p, p) array (log-space representation of A)
    logB: (N, p, p) array (log-space representation of B)

    Returns:
    logC: (N, p, p) array, where logC = log(A @ B)
    """
    # logA has shape (N, p, p)
    # logB has shape (N, p, p)

    # Compute log(A @ B) using broadcasting
    logC = logsumexp(logA[:, :, :, np.newaxis] + logB[:, np.newaxis, :, :], axis=2)

    return logC


def test7():
    # Example usage
    N, p = 100, 10  # Example dimensions
    A = np.random.rand(N, p, p)  # Random log-space matrices
    B = np.random.rand(N, p, p)

    logC = log_matmul(np.log(A), np.log(B))
    C_actual = np.log(np.matmul(A, B))  # True log(A @ B)

    # Check if values are close
    print(np.allclose(logC, C_actual, atol=1e-6))  # Should print True


def test8():
    tree = popgen.utils.get_random_binary_tree(10)
    tree.draw()
    traversor = popgen.utils.TraversalGenerator()
    for n in traversor(tree, order='pre'):
        print(n.name)



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    test8()

