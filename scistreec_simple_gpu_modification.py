import os
import popgen
import popgen.utils
from transition_solver import TransitionProbability
from utils_gpu import *
from find_deletion_gpu import *
import numpy as np
import cupy as cp 
from scipy.stats import binom
from scipy.special import comb, logsumexp, log_softmax
from cupyx.scipy.special import logsumexp as logsumexp_gpu 
from cupyx.scipy.special import log_softmax as log_softmax_gpu 
from time import time
from kernels import *
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
from popgen.utils.metric import tree_accuracy

# ----------------------------- non-class version ------------------------------

CN_MAX = 2
CN_MIN = 0
LAMBDA_C = 1
LAMBDA_S = 0.1
LAMBDA_T = 1
N = int((CN_MAX-CN_MIN+1) * (CN_MAX+CN_MIN+2) / 2)


def index_gt_(i, j):
    return int((i+j) * (i+j+1) / 2 + i - int((CN_MIN) * (CN_MIN + 1) / 2))


def log_likelihood_with_ado_seqerr_gt(ref_counts, alt_counts, g0, g1, ado=0.2, seqerr=0.01):
    """
        L(G) = P(D|G), given G=(g0, g1)
    """
    if g0 == g1 == 0:
        return -np.inf
    p00, p01, p10, p11 = 1-seqerr, seqerr, seqerr, 1-seqerr
    logado, log1mado = np.log(ado), np.log(1-ado)
    L = []
    w = []
    for g0_ in range(0, g0+1):
        for g1_ in range(0, g1+1):
            if g0_ == g1_ == 0:
                continue
            q = g0_ / (g0_ + g1_)
            log_L = ref_counts*np.log(q*p00+(1-q)*p10) + alt_counts*np.log(q*p01+(1-q)*p11)
            L.append(log_L)
            # w.append(np.exp((g0-g0_)*np.log(ado)+(g1-g1_)*np.log(ado)+g0_*np.log(1-ado)+g1_*np.log(1-ado)))
            wei = np.log(comb(g0, g0-g0_)) + (g0-g0_)*logado + g0_*log1mado + np.log(comb(g1, g1-g1_)) + (g1-g1_)*logado + g1_*log1mado
            w.append(np.exp(wei))
    L = np.array(L)
    w = np.array(w)
    likelihood = logsumexp(L, b=w)
    return likelihood



def log_posterior_with_gt(ref_counts, alt_counts, cn, af):
    """
        P(G|D)
        # prior of genotype is sampled in the bino
        # mial distribution
    """
    cn = min(cn, CN_MAX)   # scale cn into the reasonable range
    cn = max(cn, CN_MIN)
    posterior = np.empty(N, dtype=float)
    posterior.fill(-np.inf)
    for g0 in range(CN_MAX+1):
        for g1 in range(CN_MAX+1):
            if g0 + g1 == cn:
                p = binom.logpmf(g0, cn, af)
                l = log_likelihood_with_ado_seqerr_gt(ref_counts, alt_counts, g0, g1)
                # print(p, l)
                posterior[index_gt_(g0, g1)] = p + l
    # print(posterior)
    posterior = log_softmax(posterior)
    return posterior



@cpu_time
def estimate_allele_frequency(reads):
    count_0 = (reads[:, :, 0] / (reads[:, :, 2] / 2)).sum(axis=1)
    count_1 = (reads[:, :, 1] / (reads[:, :, 2] / 2)).sum(axis=1)
    return count_0 / (count_0 + count_1)
    
        
@cpu_time
def init_prob_leaves(reads):
    afs = estimate_allele_frequency(reads)
    probs = []
    nsite, ncell = len(reads), len(reads[0])
    for i in range(ncell):
        prob = []
        for j in range(nsite):
            prob.append(log_posterior_with_gt(reads[j][i][0], reads[j][i][1], reads[j][i][2], afs[j]))
        probs.append(prob)
    return np.array(probs)


def init_prob_leaves_gpu(reads, CN_MAX=CN_MAX, CN_MIN=CN_MIN, ado=0.1, seqerr=0.001):
    reads = cp.asarray(reads, dtype=np.float32)
    nsite, ncell = reads.shape[:2]

    ref = reads[:, :, 0]
    alt = reads[:, :, 1]
    cn = reads[:, :, 2]
    # Estimate allele frequencies
    af = cp.sum(ref / (cn / 2), axis=1)
    bf = cp.sum(alt / (cn / 2), axis=1)
    afs = af / (af + bf)
    N = int((CN_MAX - CN_MIN + 1) * (CN_MAX + CN_MIN + 2) / 2)
    probs = cp.zeros((ncell * nsite, N), dtype=cp.float32)
    afs = cp.array(afs, dtype=cp.float32)

    threads_per_block = 128
    blocks_per_grid = (nsite * ncell + threads_per_block - 1) // threads_per_block
    print('threads per block', threads_per_block, 'blocks per grid', blocks_per_grid)
    compute_genotype_log_probs()((blocks_per_grid,), (threads_per_block,), (
        ref.ravel(), alt.ravel(), cn.ravel(), afs, probs.ravel(),
        np.float32(ado), np.float32(seqerr),
        np.int32(ncell), np.int32(nsite),
        np.int32(CN_MAX), np.int32(CN_MIN), np.int32(N)
    ))

    return probs.reshape((ncell, nsite, N))


class ScisTreeC():
    def __init__(self, 
                CN_MAX = CN_MAX, 
                CN_MIN = CN_MIN,
                LAMBDA_C = LAMBDA_C,
                LAMBDA_S = LAMBDA_S,
                LAMBDA_T = LAMBDA_T,
                verbose = True):
        self.CN_MAX = CN_MAX
        self.CN_MIN = CN_MIN
        self.LAMBDA_C = LAMBDA_C
        self.LAMBDA_S = LAMBDA_S
        self.LAMBDA_T = LAMBDA_T
        self.N = int((CN_MAX-CN_MIN+1) * (CN_MAX+CN_MIN+2) / 2)
        self._indexing()
        transition = TransitionProbability(CN_MAX=self.CN_MAX,
                                            CN_MIN=self.CN_MIN,
                                            LAMBDA_C=self.LAMBDA_C,
                                            LAMBDA_S=self.LAMBDA_S,
                                            LAMBDA_T=self.LAMBDA_T)

        self.tran_prob_mutation_free = cp.log(cp.asarray(transition.solve_no_mutation(verbose=False)))
        self.tran_prob_mutation = cp.log(cp.asarray(transition.solve_mutation(verbose=False)))
        self.traversor = popgen.utils.TraversalGenerator()
        if verbose:
            print('with mutation')
            print(transition.format(self.tran_prob_mutation.get()))
            print('mutation free')
            print(transition.format(self.tran_prob_mutation_free.get()))
        # print(self.tran_prob_mutation_free)
        # print(self.tran_prob_mutation)
        # print(self.tran_prob_mutation.shape)
        # print(self.tran_prob_mutation_free.shape)

    def valid(self, g0, g1):
        return min(g0, g1) >= 0 and self.CN_MIN <= g0 + g1 <= self.CN_MAX
    
    def _indexing(self):
        state2index = {}
        index2state = {}
        for i in range(0, self.CN_MAX+1):
            for j in range(0, self.CN_MAX+1):
                if self.valid(i, j):
                    index = int((i+j) * (i+j+1) / 2 + i - int((self.CN_MIN) * (self.CN_MIN + 1) / 2))
                    state2index[(i, j)] = index
                    index2state[index] = (i, j)
        self.state2index = state2index
        self.index2state = index2state
        # print(self.state2index)

    """
        Index for state
    """
    # def index_gt(self, i, j):
    #     return int((i+j) * (i+j+1) / 2 + i - int((self.CN_MIN) * (self.CN_MIN + 1) / 2))

    def index_gt(self, i, j):
        return self.state2index[(i, j)]

    """
        CN profie at index
    """

    def cn_profile_at_index(self, index):
        return self.index2state[index]

    """
        Index for transition matrix
    """
    # def index(self, i, j, p ,q):
    #     # use yang's tri indexing
    #     i1 = (i+j) * (i+j+1) / 2 + i - int((self.CN_MIN) * (self.CN_MIN + 1) / 2)
    #     i2 = (p+q) * (p+q+1) / 2 + p - int((self.CN_MIN) * (self.CN_MIN + 1) / 2)
    #     return int(i1*self.N + i2)
    def index(self, i, j, p ,q):
        # use yang's tri indexing
        i1 = self.index_gt(i, j)
        i2 = self.index_gt(p, q)
        return int(i1*self.N + i2)
    


    def get_true_tree(self, nwk, offset=-1):
        tree = popgen.utils.relabel(popgen.utils.from_newick(nwk), offset=offset)
        return tree
    

    def pairwise_distance_matrix(self, probs):
        ncell, nsite, num_states = probs.shape
        exp_probs = cp.exp(probs)
        expected_distances = cp.einsum('ipq,jpq->ij', exp_probs, 1-exp_probs)
        # print(expected_distances)
        return expected_distances


    def initial_tree(self, probs):
        """
            Build the initial tree by NJ
        """
        distance = self.pairwise_distance_matrix(probs)
        tree = neighbor_joining(distance)
        # print(sorted(tree.get_leaves()))
        return tree


    def marginal_evaluate(self, probs, tree):  # offset=-1 (1-index)
        """
            The naive approach in O(n^2)
        """
        nsite = probs.shape[1]
        tran_prob_mut_broadcast = cp.tile(self.tran_prob_mutation, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
        tran_prob_mut_free_broadcast = cp.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
        log_likelihoods = []
        for node1 in self.traversor(tree):
            start = time()
            if node1.is_root():
                continue
            t = tree.copy()
            for node2 in self.traversor(t):
                if node2.is_leaf():
                    node2.state = probs[int(node2.name)]
                    node2.reads = [r[int(node2.name)] for r in reads]
                else:
                    if node1.parent == node2:
                        state = t[node1.identifier].state
                        state = state.reshape(state.shape[0], 1, state.shape[1])
                        sibs = node1.get_siblings()
                        # print(tran_prob_mut_broadcast.shape, state.shape)
                        components = [log_mat_vec_mul(tran_prob_mut_broadcast, state)]
                        for sib in sibs:
                            state = t[sib.identifier].state
                            state = state.reshape(state.shape[0], 1, state.shape[1])
                            components.append(log_mat_vec_mul(tran_prob_mut_free_broadcast, state))
                    else:
                        components = []
                        for child in node2.get_children():
                            state = child.state
                            # print(state.shape)
                            state = state.reshape(state.shape[0], 1, state.shape[1])
                            components.append(log_mat_vec_mul(tran_prob_mut_free_broadcast, state))
                    components = cp.array(components)
                    node2.state = cp.sum(components, axis=0)
                # print(node1.name, node2.name, node2==node1.parent, node2.state[0].round(2).tolist(),)
                # print(node2.name, node2.state.tolist())
            log_likelihood = t.root.state[:, self.index_gt(2,0)]
            log_likelihoods.append(log_likelihood)
            end = time()
            print('evalute a single branch', end - start)
            # break
        log_likelihoods = cp.array(log_likelihoods)
        max_L = log_likelihoods.max(axis=0).sum()
        print(t.root.state[0])
        return max_L


    @cpu_time
    def calcualte_U(self, tree, probs):
        """
            Bottom-up 
        """
        nsite = probs.shape[1]
        tran_prob_mut_free_broadcast = cp.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, self.N, self.N)
        tran_prob_mut_broadcast = cp.tile(self.tran_prob_mutation, (nsite, 1)).reshape(nsite, self.N, self.N)
        for node in self.traversor(tree):
            if node.is_leaf():
                node.U = probs[int(node.name)]
                node.U_ = log_mat_vec_mul(tran_prob_mut_free_broadcast, node.U.reshape(nsite, 1, self.N))
                node._U = log_mat_vec_mul(tran_prob_mut_broadcast, node.U.reshape(nsite, 1, self.N))
                continue
            components = []
            for child in node.get_children():
                components.append(child.U_)
            components = cp.array(components)
            node.U = cp.sum(components, axis=0)
            node.U_ = log_mat_vec_mul(tran_prob_mut_free_broadcast, node.U.reshape(nsite, 1, self.N))
            node._U = log_mat_vec_mul(tran_prob_mut_broadcast, node.U.reshape(nsite, 1, self.N))

    @cpu_time
    def calculate_Q(self, tree):
        """
            calcuate Q for each site recursively, have to do after calculation of U, use this fashion instead of DFS
        """
        # assert 'U' in tree.root.__dict__, "fatal: calculate U first!"
        nsite = tree.root.U.shape[0]
        tran_prob_mut_free_broadcast = cp.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, self.N, self.N)
        for node in self.traversor(tree, order='pre'):
            if node.is_root():
                ident = cp.log(cp.identity(self.N))
                node.Q = cp.tile(ident, (nsite, 1)).reshape(nsite, self.N, self.N)
            else:
                # node.Q = log_matmul(node.parent.Q, tran_prob_mut_free_broadcast)
                node.Q = node.parent.Q.copy()
                for sib in node.get_siblings():
                    node.Q += sib.U_.reshape(nsite, 1, self.N)
                node.Q = log_matmul(node.Q, tran_prob_mut_free_broadcast)

    @cpu_time
    def marginal_evaluate_dp(self, probs, tree):
        """
            DP speedup
        """
        nsite = probs.shape[1]
        self.calcualte_U(tree, probs)
        # self.calculate_Q(tree.root)
        self.calculate_Q(tree)
        # loop all branches to place mutation
        likelihoods = []
        branches = []
        for node in self.traversor(tree):
            if node.is_root(): # currently do not consider mutation happens right above the root.
                likelihoods.append(node.U[:, self.index_gt(1, 1)])
                likelihoods.append(node.U[:, self.index_gt(0, 2)])
                likelihoods.append(node.U[:, self.index_gt(2, 0)])
                # print(node.name, likelihood)
                continue
            res = node._U
            for sib in node.get_siblings():
                res += sib.U_
            likelihood = log_mat_vec_mul(node.parent.Q, res.reshape(nsite, 1, self.N))
            # print(node.name, likelihood)
            likelihoods.append(likelihood[:, self.index_gt(2, 0)])
            branches.append(node.identifier)
        likelihoods = cp.array(likelihoods)
        max_L = likelihoods.max(axis=0).sum()
        # print(likelihoods.argmax(axis=0))
        # print(tree.root.U.max(axis=1).sum())
        # for ind in tree.root.U.argmax(axis=1):
        #     print(self.cn_profile_at_index(int(ind)))
        return max_L
    
    ## ---------------------------------- copy without storing U' -----------------------------

    # @cpu_time
    # def calcualte_U(self, tree, probs):
    #     """
    #         Bottom-up 
    #     """
    #     nsite = probs.shape[1]
    #     tran_prob_mut_free_broadcast = cp.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, self.N, self.N)
    #     for node in self.traversor(tree):
    #         if node.is_leaf():
    #             node.U = probs[int(node.name)]
    #             continue
    #         components = []
    #         for child in node.get_children():
    #             components.append(log_mat_vec_mul(tran_prob_mut_free_broadcast, child.U.reshape(nsite, 1, self.N)))
    #         components = cp.array(components)
    #         node.U = cp.sum(components, axis=0)

    
        
    # # this may not be correct. see below
    # # def calculate_Q(self, node):
    # #     """
    # #         calcuate Q for each site recursively, have to do after calculation of U
    # #     """
    # #     assert 'U' in node.__dict__, "fatal: no U!"
    # #     nsite = node.U.shape[0]
    # #     tran_prob_mut_free_broadcast = cp.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, self.N, self.N)
    # #     if node.is_root():
    # #         ident = cp.log(cp.identity(self.N))
    # #         node.Q = cp.tile(ident, (nsite, 1)).reshape(nsite, self.N, self.N)
    # #     else:
    # #         node.Q = log_matmul(node.parent.Q, tran_prob_mut_free_broadcast)
    # #         for sib in node.get_siblings():
    # #             s = log_mat_vec_mul(tran_prob_mut_free_broadcast, sib.U.reshape(nsite, 1, self.N))
    # #             node.Q += s.reshape(nsite, self.N, 1)
    # #     for child in node.get_children():
    # #         self.calculate_Q(child)

    # # @cpu_time
    # # def calculate_Q(self, node):
    # #     """
    # #         calcuate Q for each site recursively, have to do after calculation of U
    # #     """
    # #     assert 'U' in node.__dict__, "fatal: calculate U first!"
    # #     nsite = node.U.shape[0]
    # #     tran_prob_mut_free_broadcast = cp.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, self.N, self.N)
    # #     if node.is_root():
    # #         ident = cp.log(cp.identity(self.N))
    # #         node.Q = cp.tile(ident, (nsite, 1)).reshape(nsite, self.N, self.N)
    # #     else:
    # #         # node.Q = log_matmul(node.parent.Q, tran_prob_mut_free_broadcast)
    # #         node.Q = node.parent.Q.copy()
    # #         for sib in node.get_siblings():
    # #             s = log_mat_vec_mul(tran_prob_mut_free_broadcast, sib.U.reshape(nsite, 1, self.N))
    # #             node.Q += s.reshape(nsite, 1, self.N)
    # #         node.Q = log_matmul(node.Q, tran_prob_mut_free_broadcast)
    # #     for child in node.get_children():
    # #         self.calculate_Q(child)

    # @cpu_time
    # def calculate_Q(self, tree):
    #     """
    #         calcuate Q for each site recursively, have to do after calculation of U, use this fashion instead of DFS
    #     """
    #     # assert 'U' in tree.root.__dict__, "fatal: calculate U first!"
    #     nsite = tree.root.U.shape[0]
    #     t_total = 0 
    #     tran_prob_mut_free_broadcast = cp.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, self.N, self.N)
    #     for node in self.traversor(tree, order='pre'):
    #         if node.is_root():
    #             ident = cp.log(cp.identity(self.N))
    #             node.Q = cp.tile(ident, (nsite, 1)).reshape(nsite, self.N, self.N)
    #         else:
    #             # node.Q = log_matmul(node.parent.Q, tran_prob_mut_free_broadcast)
    #             node.Q = node.parent.Q.copy()
    #             for sib in node.get_siblings():
    #                 start = time()
    #                 s = log_mat_vec_mul(tran_prob_mut_free_broadcast, sib.U.reshape(nsite, 1, self.N))
    #                 t_total += time() - start
    #                 node.Q += s.reshape(nsite, 1, self.N)
    #             node.Q = log_matmul(node.Q, tran_prob_mut_free_broadcast)
    #     print('calculate Q total time', t_total)

    # @cpu_time
    # def marginal_evaluate_dp(self, probs, tree):
    #     """
    #         DP speedup
    #     """
    #     nsite = probs.shape[1]
    #     start = time()
    #     tran_prob_mut_broadcast = cp.tile(self.tran_prob_mutation, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
    #     tran_prob_mut_free_broadcast = cp.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
    #     self.calcualte_U(tree, probs)
    #     # self.calculate_Q(tree.root)
    #     self.calculate_Q(tree)
    #     # loop all branches to place mutation
    #     likelihoods = []
    #     branches = []
    #     for node in self.traversor(tree):
    #         if node.is_root(): # currently do not consider mutation happens right above the root.
    #             continue
    #         res = log_mat_vec_mul(tran_prob_mut_broadcast, node.U.reshape(nsite, 1, self.N))
    #         for sib in node.get_siblings():
    #             res += log_mat_vec_mul(tran_prob_mut_free_broadcast, sib.U.reshape(nsite, 1, self.N))
    #         likelihood = log_mat_vec_mul(node.parent.Q, res.reshape(nsite, 1, self.N))
    #         likelihoods.append(likelihood[:, self.index_gt(2, 0)])
    #         branches.append(node.identifier)
    #     likelihoods = cp.array(likelihoods)
    #     max_L = likelihoods.max(axis=0).sum()
    #     # print(likelihoods.argmax(axis=0))
    #     end = time()
    #     print('dp', end - start)
    #     return max_L

    # # def marginal_evaluate_dp(self, reads, nwk, offset=-1):
    # #     """
    # #         DP speedup, compatiable with calculate_Q(node) instead of calculate_Q(tree)
    # #     """
    # #     nsite, ncell = len(reads), len(reads[0])
    # #     tree = popgen.utils.relabel(popgen.utils.from_newick(nwk), offset=offset)
    # #     probs = self.init_prob_leaves(reads)
    # #     print('init prob done.')
    # #     start = time()
    # #     tran_prob_mut_broadcast = cp.tile(self.tran_prob_mutation, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
    # #     tran_prob_mut_free_broadcast = cp.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
    # #     self.calcualte_U(tree, probs)
    # #     self.calculate_Q(tree.root)
    # #     # loop all branches to place mutation
    # #     likelihoods = []
    # #     branches = []
    # #     for node in self.traversor(tree):
    # #         if node.is_root(): # currently do not consider mutation happens right above the root.
    # #             continue
    # #         res = log_mat_vec_mul(tran_prob_mut_broadcast, node.U.reshape(nsite, 1, self.N))
    # #         for sib in node.get_siblings():
    # #             res += log_mat_vec_mul(tran_prob_mut_free_broadcast, sib.U.reshape(nsite, 1, self.N))
    # #         likelihood = log_mat_vec_mul(node.parent.Q, res.reshape(nsite, 1, self.N))
    # #         likelihoods.append(likelihood[:, self.index_gt(2, 0)])
    # #         branches.append(node.identifier)
    # #     likelihoods = cp.array(likelihoods)
    # #     max_L = likelihoods.max(axis=0).sum()
    # #     print(likelihoods.argmax(axis=0))
    # #     end = time()
    # #     print('dp', end - start)
    # #     return max_L


    def maximal_evaluate(self, probs, tree, offset=-1, return_trees=False):
        nsite = probs.shape[1]
        # probs = init_prob_leaves(reads)
        # print('init prob done.')
        # print(probs.shape)
        tran_prob_mut_broadcast = np.tile(self.tran_prob_mutation, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
        tran_prob_mut_free_broadcast = np.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
        log_likelihoods = []
        if return_trees:
            trees = []
        for node1 in self.traversor(tree):
            if node1.is_root():
                continue
            t = tree.copy()
            for node2 in self.traversor(t):
                if node2.is_leaf():
                    node2.state = probs[int(node2.name)]
                    # node2.reads = [r[int(node2.name)] for r in reads]
                    # print(reads)
                else:
                    if node1.parent == node2:
                        state = t[node1.identifier].state
                        state = state.reshape(state.shape[0], 1, state.shape[1])
                        sibs = node1.get_siblings()
                        # print(tran_prob_mut_broadcast.shape, state.shape)
                        val, arg = log_mat_vec_max(tran_prob_mut_broadcast, state)
                        components = [val]
                        components_arg = {node1.identifier: arg}
                        for sib in sibs:
                            state = t[sib.identifier].state
                            state = state.reshape(state.shape[0], 1, state.shape[1])
                            val_, arg_ = log_mat_vec_max(tran_prob_mut_free_broadcast, state)
                            components.append(val_)
                            components_arg[sib.identifier] = arg_
                    else:
                        components = []
                        components_arg = {}
                        for child in node2.get_children():
                            state = child.state
                            # print(state.shape)
                            state = state.reshape(state.shape[0], 1, state.shape[1])
                            val_, arg_ = log_mat_vec_max(tran_prob_mut_free_broadcast, state)
                            components.append(val_)
                            components_arg[child.identifier] = arg_
                    node2.state = cp.sum(cp.asarray(components), axis=0)
                    node2.arg = components_arg
                # print(node1.name, node2.name, node2==node1.parent, node2.state[0].round(2).tolist(),)
                # print(node2.name, node2.state.tolist())
            log_likelihood = t.root.state[:, self.index_gt(2,0)]
            log_likelihoods.append(log_likelihood)
            if return_trees:
                trees.append(t)
            # break
        log_likelihoods = cp.array(log_likelihoods)
        max_L = log_likelihoods.max(axis=0).sum()
            # break
        if return_trees:
            return max_L, trees
        return max_L

    
    def _bfs(self, node, site_index, state_index):
        cn_profile = self.cn_profile_at_index(state_index)
        node.cn = f'{sum(cn_profile)}:{cn_profile}'
        if not node.is_leaf():
            for child in node.get_children():
                child_state_index = node.arg[child.identifier][site_index][state_index]
                self._bfs(child, site_index, child_state_index)
        else:
            node.cn = f'{node.name} [CN {node.cn}] [READ:{node.reads[site_index]}]'
            # print(node.reads)


    def maximal_path_decoding(self, probs, tree, site_index):
        max_L, trees = self.maximal_evaluate(probs, tree, return_trees=True)
        max_tree = trees[np.argmax([t.root.state.get()[site_index, self.index_gt(2, 0)] for t in trees])]
        self._bfs(max_tree.root, site_index, self.index_gt(2, 0))
        max_tree.draw(nameattr='cn')
        

    def nni_search(self, tree):
        """
            NNI neighbor
        """
        tree.draw()
        for node in tree.get_all_nodes():
            switch = not tree[node].is_leaf()
            for child in tree[node].get_children():
                if child.is_leaf():
                    switch = False
            if switch:
                lc1, lc2 = tree[node].get_children()[0].get_children()
                rc1, rc2 = tree[node].get_children()[1].get_children()
                print(lc1, lc2, rc1, rc2)



    def nni_search_non_optim_sinlge_round(self, probs, tree):
        """
            NNI neighbor
        """
        candidates = []
        # tree.draw()
        # TODO: quartet switch
        for node in tree.get_all_nodes():
            switch = not tree[node].is_leaf()
            for child in tree[node].get_children():
                if child.is_leaf():
                    switch = False
            if switch:
                t1 = tree.copy()
                p1 = t1[node].get_children()[0]
                p2 = t1[node].get_children()[1]
                lc1, lc2 = p1.get_children()
                rc1, rc2 = p2.get_children()
                p1.remove_child(lc1)
                p2.remove_child(rc1)
                lc1.set_parent(p2)
                rc1.set_parent(p1)
                p1.add_child(rc1)
                p2.add_child(lc1)
                candidates.append(t1)
                t2 = tree.copy()
                p1 = t2[node].get_children()[0]
                p2 = t2[node].get_children()[1]
                lc1, lc2 = p1.get_children()
                rc1, rc2 = p2.get_children()
                p1.remove_child(lc1)
                p2.remove_child(rc2)
                lc1.set_parent(p2)
                rc2.set_parent(p1)
                p1.add_child(rc2)
                p2.add_child(lc1)
                candidates.append(t2)
        # TODO: triplet switch
        for node in tree.get_all_nodes():
            switch = not tree[node].is_leaf() and not tree[node].is_root()
            if switch:
                t1 = tree.copy()
                sib = t1[node].get_siblings()[0]
                c1 = t1[node].get_children()[0]
                c2 = t1[node].get_children()[1]
                t1[node].remove_child(c1)
                t1[node].parent.remove_child(sib)
                t1[node].add_child(sib)
                sib.set_parent(t1[node])
                t1[node].parent.add_child(c1)
                c1.set_parent(t1[node].parent)
                candidates.append(t1)
                t2 = tree.copy()
                sib = t2[node].get_siblings()[0]
                c1 = t2[node].get_children()[0]
                c2 = t2[node].get_children()[1]
                t2[node].remove_child(c2)
                t2[node].parent.remove_child(sib)
                t2[node].add_child(sib)
                sib.set_parent(t2[node])
                t2[node].parent.add_child(c2)
                c2.set_parent(t2[node].parent)
                candidates.append(t2)
                    
        # local search:
        best_tree = tree.copy()
        best_likelihood = self.marginal_evaluate_dp(probs, best_tree)
        for t in candidates:
            likelihood = self.marginal_evaluate_dp(probs, t.copy())
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_tree = t
        return best_tree, best_likelihood
    

    def local_search(self, probs, ground_truth=None):
        tree = self.initial_tree(probs)
        L = -np.inf
        while True:
            better_tree, likelihood = self.nni_search_non_optim_sinlge_round(probs, tree)
            if likelihood <= L:
                print('converge, stop!')
                break
            else:
                L = likelihood
                tree = better_tree
                print('new tree evaluated', L, end=' ')
                if ground_truth:
                    print('acc:', tree_accuracy(ground_truth, tree))
        return tree, L



if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    s = ScisTreeC(verbose=True)
    # print(s.log_likelihood_with_ado_seqerr_gt(2, 0, 1, 0))
    # res = s.log_posterior_with_gt(2, 0, 2, 0.9)
    # print(res)
    # popgen.utils.ScisTree2
    # reads = [[(2,2,2), (3,2,2), (2,0,1), (1,0,1), (1,2,2), (2,1,2), (3,0,2), (2,0,2), (2,0,2), (1,0,2)], 
    #         [(2,0,2), (1,0,2), (2,0,2), (1,0,2), (1,0,2), (2,0,2), (3,0,2), (2,0,2), (1,2,2), (0,2,2)], 
    #         [(2,2,2), (3,2,2), (2,0,1), (1,0,1), (1,2,2), (2,1,2), (3,0,2), (2,0,2), (2,0,2), (1,0,2)]]
    
    # # reads = [[(2,2,2), (3,2,2), (2,0,2), (1,0,2), (1,2,2), (2,1,2), (3,0,2), (2,0,2), (2,0,2), (1,0,2)], 
    # #         [(2,0,2), (1,0,2), (2,0,2), (1,0,2), (1,0,2), (2,0,2), (3,0,2), (2,0,2), (1,2,2), (0,2,2)], 
    # #         [(2,2,2), (3,2,2), (2,0,2), (1,0,2), (1,2,2), (2,1,2), (3,0,2), (2,0,2), (2,0,2), (1,0,2)]]

    # reads = np.array(reads)
    # probs = s.init_prob_leaves(reads)
    # print(probs.shape)
    # nwk = popgen.utils.get_random_binary_tree(10, start_index=1).output()
    # print(nwk)
    # nwk = '((((1,2),(3,4)),(5,6)),(7,(8,(9,10))));'
    # nwk = '(((((10,9),4),1),(((3,8),6),(2,5))),7);'
    

    # print(nwk)
    # nwk = '((1,2), (3,4));'
    # likelihood = s.marginal_evaluate(reads, nwk)
    # print('marginal', likelihood)
    # likelihood = s.maximal_evaluate(reads, nwk)
    # print('maximal', likelihood)

    # s.maximal_path_decoding(reads, nwk, site_index=0)
    # z = s.log_likelihood_with_ado_seqerr_gt(2, 0, 0, 1)
    # print(z)


    
    # p = np.array([[-np.inf, -8.05, -0.0, -np.inf, -np.inf, -np.inf]]).reshape(1, -1)
    # print(s.tran_prob_mutation_free)
    # z = s.tran_prob_mutation_free + p
    # print(logsumexp(z, axis=-1))



    ### cellcoal simulation test

    # dirname = 'simulation/test_no_cn_d0.5_err0.05'
    dirname = 'simulation/test'
    i = 5

    reads, tree, tg = get_scistreec_input_with_cn(dirname, i)
    n_site, n_cell, _ = reads.shape
    print('#site', n_site, '#cell', n_cell)
    rename_dict = {f'cell{i+1:04}': str(i+1) for i in range(n_cell)}
    tree = popgen.utils.relabel(tree, name_map=rename_dict)
    nwk = tree.output()

    # reads = cp.asarray(reads, dtype=int)
    # random_tree = popgen.utils.get_random_binary_tree(100, start_index=0)
    tree = s.get_true_tree(nwk)

    ## test gpu init
    probs = init_prob_leaves_gpu(reads)
    # print('pp', probs)
    # np.savetxt('reads.txt', reads[:, :, -1].astype(int), fmt='%i')
    ##  write probs.txt
    visual_probs = []


    # np.savetxt('probs.txt', probs.get()[:, 57, :], fmt='%4f')
    nj_tree = s.initial_tree(probs)
    # print(probs1[0,0])
    # print(probs.shape)


    # probs = init_prob_leaves(reads)
    # np.save('probs1.npy', probs)
    # # print(probs.shape)
    # probs = cp.asarray(probs)

    # print(cp.isclose(probs, probs1))


    # spr move
    # tree = popgen.utils.spr_move(tree, 1)
    # tree = popgen.utils.get_random_binary_tree(100, start_index=0)
    # # s.maximal_path_decoding(reads, nwk, site_index=626)
    # # print(nwk)

    # tree = nj_tree

    # ------------------------------------------

    print('true tree', s.marginal_evaluate_dp(probs, tree), tree_accuracy(tree, tree))
    print('nj tree', s.marginal_evaluate_dp(probs, nj_tree), tree_accuracy(tree, nj_tree))
    # print('random tree', s.marginal_evaluate_dp(probs, random_tree), tree_accuracy(tree, random_tree))

    # --------------------------------------------
    # likelihood = s.marginal_evaluate_dp(probs, tree)

    
    # likelihood_nj_tree = s.marginal_evaluate_dp(probs, nj_tree)
    # likelihood_true_tree = s.marginal_evaluate_dp(probs, tree)
    # acc_nj_tree = tree_accuracy(tree, nj_tree)
    # print('*** true', likelihood_true_tree)
    # print('*** nj', likelihood_nj_tree, acc_nj_tree,)

    # neighbor_true_tree = popgen.utils.spr_move(tree, 2)
    # likelihood_neigh = s.marginal_evaluate_dp(probs, neighbor_true_tree)
    # acc_neigh = tree_accuracy(tree, neighbor_true_tree)
    # print('*** true neighbor', likelihood_neigh, acc_neigh)




    # # ----------------- ScisTree 2 same input-------------------
    # np.set_printoptions(precision=10)
    # # scistree2_in = read_scistree2_input(f'{dirname}/{i}.prob')
    # # print(scistree2_in)
    # # print(scistree2_in)
    # scistree2 = popgen.utils.ScisTree2()
    # sc_in = cp.exp(probs).get()[:, :, -1].T.astype(float)
    # sc_in[sc_in > 1-1e-10] = 1-1e-10
    # sc_in[sc_in < 1e-10] = 1e-10
    # res = scistree2.infer(sc_in)
    # scistree2_tree = popgen.utils.from_newick(res[1])
    # scistree2_tree = popgen.utils.relabel(scistree2_tree, offset=-1)
    # acc_scistree2_tree = tree_accuracy(tree, scistree2_tree)
    # likelihood_scistree2 = s.marginal_evaluate_dp(probs, scistree2_tree)
    # print('*** scistree2 same', likelihood_scistree2, acc_scistree2_tree)


    # ----------------- ScisTree 2 same input nni-------------------
    # scistree2 = popgen.utils.ScisTree2(nni=True)
    # sc_in = cp.exp(probs).get()[:, :, -1].T.astype(float)
    # sc_in[sc_in > 1-1e-10] = 1-1e-10
    # sc_in[sc_in < 1e-10] = 1e-10
    # # print(sc_in >= 1)
    # res = scistree2.infer(sc_in)
    # scistree2_tree = popgen.utils.from_newick(res[1])
    # scistree2_tree = popgen.utils.relabel(scistree2_tree, offset=-1)
    # print('00', scistree2_tree)
    # acc_scistree2_tree = tree_accuracy(tree, scistree2_tree)
    # likelihood_scistree2 = s.marginal_evaluate_dp(probs, scistree2_tree)
    # print('*** scistree2 same nni', likelihood_scistree2, acc_scistree2_tree)


    # ----------------- ScisTree 2 same input evaluate-------------------
    np.save('reads.npy', reads)
    print(nwk)
    scistree2 = popgen.utils.ScisTree2()
    sc_in = cp.exp(probs).get()[:, :, -1].T
    # sc_in[sc_in > 1-1e-10] = 1-1e-10
    # sc_in[sc_in < 1e-10] = 1e-10
    print((sc_in < 0.5).astype(int))
    imputed_geno, like = scistree2.evaluate(sc_in, nwk)
    print(like)



    # ----------------- ScisTree 2 -------------------
    scistree2_in = read_scistree2_input(f'{dirname}/{i}.prob')
    scistree2 = popgen.utils.ScisTree2()
    res = scistree2.infer(scistree2_in)
    scistree2_tree = popgen.utils.from_newick(res[1])
    scistree2_tree = popgen.utils.relabel(scistree2_tree, offset=-1)
    acc_scistree2_tree = tree_accuracy(tree, scistree2_tree)
    likelihood_scistree2 = s.marginal_evaluate_dp(probs, scistree2_tree)
    print('*** scistree2', likelihood_scistree2, acc_scistree2_tree)


    # ----------------- ScisTree 2 NNI-------------------
    scistree2_in = read_scistree2_input(f'{dirname}/{i}.prob')
    scistree2 = popgen.utils.ScisTree2(nni=True)
    res = scistree2.infer(scistree2_in)
    scistree2_tree = popgen.utils.from_newick(res[1])
    scistree2_tree = popgen.utils.relabel(scistree2_tree, offset=-1)
    acc_scistree2_tree = tree_accuracy(tree, scistree2_tree)
    likelihood_scistree2 = s.marginal_evaluate_dp(probs, scistree2_tree)
    print('*** scistree2 nni', likelihood_scistree2, acc_scistree2_tree)


    # ----------------- ScisTree 2 NJ -------------------
    scistree2_in = read_scistree2_input(f'{dirname}/{i}.prob')
    scistree2 = popgen.utils.ScisTree2(nj=True)
    res = scistree2.infer(scistree2_in)
    # print(res)
    scistree2_tree = popgen.utils.from_newick(res[0])
    scistree2_tree = popgen.utils.relabel(scistree2_tree, offset=-1)
    acc_scistree2_tree = tree_accuracy(tree, scistree2_tree)
    likelihood_scistree2 = s.marginal_evaluate_dp(probs, scistree2_tree)
    print('*** scistree2 nj', likelihood_scistree2, acc_scistree2_tree)




    mytree, myL = s.local_search(probs, ground_truth=tree)
    print('scistreec', myL, tree_accuracy(tree, mytree))