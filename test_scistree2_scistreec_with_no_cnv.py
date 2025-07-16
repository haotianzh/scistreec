import popgen.utils
from scistreec_simple_gpu_modification import ScisTreeC, init_prob_leaves, init_prob_leaves_gpu
import numpy as np 
import cupy as cp
import popgen


def random_reads(n_leaves=10, n_sites=1):
    reads = []
    for site in range(n_sites):
        read = []
        for leave in range(n_leaves):
            ref = np.random.randint(low=2, high=5)
            alt = max(0, np.random.randint(low=-2, high=2))
            read.append((ref, alt, 2))
            # read.append((0, 5, 2))
        reads.append(read)
    return reads


np.set_printoptions(precision=8, suppress=True)
scistreec = ScisTreeC(verbose=True)
scistree2 = popgen.utils.ScisTree2(nni=True)


# reads = [[(4, 0, 2), (5, 0, 2), (2, 4, 2), (5, 3, 2), (3, 1, 2), (4, 0, 2)]]

# n = 500
# reads = random_reads(n, 20)

# load reads from file
reads = np.load('reads.npy', allow_pickle=True)
print(reads.shape)
# probs = np.array([[0.9, 0.9, 0.1, 0.1]])
probs = init_prob_leaves_gpu(reads).get()
# print(probs)
# nwk = '((1,2),(3,4));'
# nwk = '(((1,2),3),4);'
# nwk = '(1,(2,(3,4)));'
# tree = popgen.utils.get_random_binary_tree(n_leave=n)
tree = popgen.utils.get_random_binary_tree(n_leave=20)
# tree = popgen.utils.from_newick(nwk)
# tree = popgen.utils.relabel(tree, offset=-1)
# tree.draw()
sc_in = np.exp(probs)[:, :, -1].T
# print(sc_in, tree.output())
imputed_genotype, scistree2_likelihood = scistree2.evaluate(sc_in,  tree.output(), offset=0)
scistreec_likelihood = scistreec.marginal_evaluate_dp(cp.asarray(probs), tree)
print('scistree2', scistree2_likelihood)
print('scistreec', scistreec_likelihood)
print('diff', scistree2_likelihood - scistreec_likelihood)
# print(scistreec.maximal_path_decoding(cp.asarray(probs), tree, site_index=0))