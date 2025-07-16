import os
import popgen
import numpy as np 
from scipy.special import logsumexp, log_softmax, comb
import popgen.utils
from transition_solver import TransitionProbability
from utils import *
from scipy.stats import binom
from find_deletion import *
from time import time
import warnings 
warnings.filterwarnings("ignore")
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

class ScisTreeC():
    def __init__(self, 
                CN_MAX = 8, 
                CN_MIN = 0,
                LAMBDA_C = 1,
                LAMBDA_S = 5,
                LAMBDA_T = 5,
                verbose = True):
        self.CN_MAX = CN_MAX
        self.CN_MIN = CN_MIN
        self.LAMBDA_C = LAMBDA_C
        self.LAMBDA_S = LAMBDA_S
        self.LAMBDA_T = LAMBDA_T
        self.N = int((CN_MAX-CN_MIN+1) * (CN_MAX+CN_MIN+2) / 2)
        self._indexing()
        self.traversor = popgen.utils.TraversalGenerator(order='post')
        transition = TransitionProbability(CN_MAX=CN_MAX,
                                            CN_MIN=CN_MIN,
                                            LAMBDA_C=LAMBDA_C,
                                            LAMBDA_S=LAMBDA_S,
                                            LAMBDA_T=LAMBDA_T)

        self.tran_prob_mutation_free = np.log(transition.solve_no_mutation(verbose=False))
        self.tran_prob_mutation = np.log(transition.solve_mutation(verbose=False))
        if verbose:
            print('with mutation')
            print(transition.format(self.tran_prob_mutation))
            print('mutation free')
            print(transition.format(self.tran_prob_mutation_free))

    def valid(self, g0, g1):
        return min(g0, g1) >= 0 and self.CN_MIN <= g0 + g1 <= self.CN_MAX

    def _indexing(self):
        state2index = {}
        index2state = {}
        for i in range(self.CN_MIN, self.CN_MAX+1):
            for j in range(self.CN_MIN, self.CN_MAX+1):
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
    


    def log_likelihood_with_ado_seqerr_gt(self, ref_counts, alt_counts, g0, g1, ado=0.1, seqerr=0.001):
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
    

    # def log_likelihood_with_ado_seqerr_gt(self, ref_counts, alt_counts, g0, g1, ado=0.1, seqerr=0.001):
    #     """
    #         L(G) = P(D|G), given G=(g0, g1)
    #     """
    #     if g0 == g1 == 0:
    #         return -np.inf
    #     p00, p01, p10, p11 = np.log(1-seqerr), np.log(seqerr), np.log(seqerr), np.log(1-seqerr)
    #     L = []
    #     w = []
    #     for g0_ in range(0, g0+1):
    #         for g1_ in range(0, g1+1):
    #             if g0_ == g1_ == 0:
    #                 continue
    #             q = g0_ / (g0_ + g1_)
    #             log_L = ref_counts*np.log(q*np.exp(p00)+(1-q)*np.exp(p10))+alt_counts*np.log(q*np.exp(p01)+(1-q)*np.exp(p11))
    #             L.append(log_L)
    #             w.append(np.exp((g0-g0_)*np.log(ado)+(g1-g1_)*np.log(ado)+g0_*np.log(1-ado)+g1_*np.log(1-ado)))
    #     likelihood = logsumexp(L, b=w)
    #     return likelihood

    # do not account for lower coverage when there is a deletion, maybe need a prior on total reads given coverage: P(ref+alt | cn)
    def log_posterior_with_gt(self, ref_counts, alt_counts, cn, af):
        """
            P(G|D)
            # prior of genotype is sampled in the binomial distribution
        """
        cn = min(cn, self.CN_MAX)   # scale cn into the reasonable range
        cn = max(cn, self.CN_MIN)
        posterior = np.empty(self.N, dtype=float)
        posterior.fill(-np.inf)
        for g0 in range(self.CN_MAX+1):
            for g1 in range(self.CN_MAX+1):
                if g0 + g1 == cn:
                    p = binom.logpmf(g0, cn, af)
                    l = self.log_likelihood_with_ado_seqerr_gt(ref_counts, alt_counts, g0, g1)
                    # print(p, l)
                    posterior[self.index_gt(g0, g1)] = p + l
        # print(posterior)
        posterior = log_softmax(posterior)
        return posterior


    def estimate_allele_frequency(self, reads):
        afs = []
        nsite, ncell = len(reads), len(reads[0])
        for i in range(nsite):
            af0, af1 = 0, 0
            for j in range(ncell):
                af0 += reads[i][j][0] / ((reads[i][j][2]) / 2)
                af1 += reads[i][j][1] / ((reads[i][j][2]) / 2)
            afs.append(af0 / (af0 + af1))
        return np.array(afs)
            

    def init_prob_leaves(self, reads):
        afs = self.estimate_allele_frequency(reads)
        probs = []
        nsite, ncell = len(reads), len(reads[0])
        for i in range(ncell):
            prob = []
            for j in range(nsite):
                prob.append(self.log_posterior_with_gt(reads[j][i][0], reads[j][i][1], reads[j][i][2], afs[j]))
            probs.append(prob)
        return np.array(probs)


    def marginal_evaluate(self, reads, nwk, offset=-1):  # offset=-1 (1-index)
        """
            The naive approach in O(n^2)
        """
        nsite, ncell = len(reads), len(reads[0])
        tree = popgen.utils.relabel(popgen.utils.from_newick(nwk), offset=offset)
        probs = self.init_prob_leaves(reads)
        # print(probs.shape)
        start = time()
        tran_prob_mut_broadcast = np.tile(self.tran_prob_mutation, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
        tran_prob_mut_free_broadcast = np.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
        log_likelihoods = []
        for node1 in self.traversor(tree):
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
                    
                    node2.state = np.sum(components, axis=0)
                # print(node1.name, node2.name, node2==node1.parent, node2.state[0].round(2).tolist(),)
                # print(node2.name, node2.state.tolist())
            log_likelihood = t.root.state[:, self.index_gt(2,0)]
            log_likelihoods.append(log_likelihood)
            # break
            
            # print('evaluate a single branch', end - start)
        log_likelihoods = np.array(log_likelihoods)
        max_L = log_likelihoods.max(axis=0).sum()
        print(log_likelihoods.argmax(axis=0))
        end = time()
        print('rr', end - start)
        return max_L
    

    def calcualte_U(self, tree, probs):
        """
            Bottom-up 
        """
        nsite = probs.shape[1]
        tran_prob_mut_free_broadcast = np.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, self.N, self.N)
        for node in self.traversor(tree):
            if node.is_leaf():
                node.U = probs[int(node.name)]
                continue
            components = []
            for child in node.get_children():
                components.append(log_mat_vec_mul(tran_prob_mut_free_broadcast, child.U.reshape(nsite, 1, self.N)))
            node.U = np.sum(components, axis=0)
            
    
    # def calculate_Q(self, node):
    #     """
    #         calcuate Q for each site recursively, have to do after calculation of U
    #     """
    #     assert 'U' in node.__dict__, "fatal: no U!"
    #     nsite = node.U.shape[0]
    #     tran_prob_mut_free_broadcast = np.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, self.N, self.N)
    #     if node.is_root():
    #         ident = np.log(np.identity(self.N))
    #         node.Q = np.tile(ident, (nsite, 1)).reshape(nsite, self.N, self.N)
    #     else:
    #         node.Q = log_matmul(node.parent.Q, tran_prob_mut_free_broadcast)
    #         for sib in node.get_siblings():
    #             s = log_mat_vec_mul(tran_prob_mut_free_broadcast, sib.U.reshape(nsite, 1, self.N))
    #             node.Q += s.reshape(nsite, self.N, 1)
    #     for child in node.get_children():
    #         self.calculate_Q(child)

    def calculate_Q(self, node):
        """
            calcuate Q for each site recursively, have to do after calculation of U
        """
        assert 'U' in node.__dict__, "fatal: calculate U first!"
        nsite = node.U.shape[0]
        tran_prob_mut_free_broadcast = np.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, self.N, self.N)
        if node.is_root():
            ident = np.log(np.identity(self.N))
            node.Q = np.tile(ident, (nsite, 1)).reshape(nsite, self.N, self.N)
        else:
            # node.Q = log_matmul(node.parent.Q, tran_prob_mut_free_broadcast)
            node.Q = node.parent.Q.copy()
            for sib in node.get_siblings():
                s = log_mat_vec_mul(tran_prob_mut_free_broadcast, sib.U.reshape(nsite, 1, self.N))
                node.Q += s.reshape(nsite, 1, self.N)
            node.Q = log_matmul(node.Q, tran_prob_mut_free_broadcast)
        for child in node.get_children():
            self.calculate_Q(child)

    def marginal_evaluate_dp(self, reads, nwk, offset=-1):
        """
            DP speedup
        """
        nsite, ncell = len(reads), len(reads[0])
        tree = popgen.utils.relabel(popgen.utils.from_newick(nwk), offset=offset)
        probs = self.init_prob_leaves(reads)
        print('init prob done.')
        start = time()
        tran_prob_mut_broadcast = np.tile(self.tran_prob_mutation, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
        tran_prob_mut_free_broadcast = np.tile(self.tran_prob_mutation_free, (nsite, 1)).reshape(nsite, probs.shape[-1], -1)
        self.calcualte_U(tree, probs)
        self.calculate_Q(tree.root)
        # loop all branches to place mutation
        likelihoods = []
        branches = []
        for node in self.traversor(tree):
            if node.is_root(): # currently do not consider mutation happens right above the root.
                continue
            res = log_mat_vec_mul(tran_prob_mut_broadcast, node.U.reshape(nsite, 1, self.N))
            for sib in node.get_siblings():
                res += log_mat_vec_mul(tran_prob_mut_free_broadcast, sib.U.reshape(nsite, 1, self.N))
            likelihood = log_mat_vec_mul(node.parent.Q, res.reshape(nsite, 1, self.N))
            likelihoods.append(likelihood[:, self.index_gt(2, 0)])
            branches.append(node.identifier)
        likelihoods = np.array(likelihoods)
        max_L = likelihoods.max(axis=0).sum()
        print(likelihoods.argmax(axis=0))
        end = time()
        print('dp', end - start)
        return max_L

        
        

    def maximal_evaluate(self, reads, nwk, offset=-1, return_trees=False):
        nsite, ncell = len(reads), len(reads[0])
        tree = popgen.utils.relabel(popgen.utils.from_newick(nwk), offset=offset)
        probs = self.init_prob_leaves(reads)
        print('init prob done.')
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
                    node2.reads = [r[int(node2.name)] for r in reads]
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
                    node2.state = np.sum(components, axis=0)
                    node2.arg = components_arg
                # print(node1.name, node2.name, node2==node1.parent, node2.state[0].round(2).tolist(),)
                # print(node2.name, node2.state.tolist())
            log_likelihood = t.root.state[:, self.index_gt(2,0)]
            log_likelihoods.append(log_likelihood)
            if return_trees:
                trees.append(t)
            # break
        log_likelihoods = np.array(log_likelihoods)
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

    def maximal_path_decoding(self, reads, nwk, site_index):
        max_L, trees = self.maximal_evaluate(reads, nwk, return_trees=True)
        max_tree = trees[np.argmax([tree.root.state[site_index, self.index_gt(2, 0)] for tree in trees])]
        self._bfs(max_tree.root, site_index, self.index_gt(2, 0))
        max_tree.draw(nameattr='cn')
        

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    s = ScisTreeC(verbose=False)
    # print(s.log_likelihood_with_ado_seqerr_gt(2, 0, 1, 0))
    # res = s.log_posterior_with_gt(2, 0, 2, 0.9)
    # print(res)
    # popgen.utils.ScisTree2
    # reads = [[(2,2,2), (3,2,2), (2,0,1), (1,0,1), (1,2,2), (2,1,2), (3,0,2), (2,0,2), (2,0,2), (1,0,2)], 
    #         [(2,0,2), (1,0,2), (2,0,2), (1,0,2), (1,0,2), (2,0,2), (3,0,2), (2,0,2), (1,2,2), (0,2,2)], 
    #         [(2,2,2), (3,2,2), (2,0,1), (1,0,1), (1,2,2), (2,1,2), (3,0,2), (2,0,2), (2,0,2), (1,0,2)]]
    
    # reads = [[(2,2,2), (3,2,2), (2,0,2), (1,0,2), (1,2,2), (2,1,2), (3,0,2), (2,0,2), (2,0,2), (1,0,2)], 
    #         [(2,0,2), (1,0,2), (2,0,2), (1,0,2), (1,0,2), (2,0,2), (3,0,2), (2,0,2), (1,2,2), (0,2,2)], 
    #         [(2,2,2), (3,2,2), (2,0,2), (1,0,2), (1,2,2), (2,1,2), (3,0,2), (2,0,2), (2,0,2), (1,0,2)]]


    # probs = s.init_prob_leaves(reads)
    # print(probs.shape)
    # nwk = popgen.utils.get_random_binary_tree(10, start_index=1).output()
    # print(nwk)
    # nwk = '((((1,2),(3,4)),(5,6)),(7,(8,(9,10))));'
    # # nwk = '(((((10,9),4),1),(((3,8),6),(2,5))),7);'
    # s.marginal_evaluate_dp(reads, nwk)

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
    from time import time
    dirname = 'simulation/test1'
    i = 1
    reads, tree, tg= get_scistreec_input_with_cn(dirname, i)
    n_site, n_cell, _ = reads.shape
    rename_dict = {f'cell{i+1:04}': str(i+1) for i in range(n_cell)}
    tree = popgen.utils.relabel(tree, name_map=rename_dict)
    nwk = tree.output()
    nwk = popgen.utils.get_random_binary_tree(100, start_index=1).output()
    # s.maximal_path_decoding(reads, nwk, site_index=626)
    likelihood = s.marginal_evaluate_dp(reads, nwk)
    print(likelihood)

    # likelihood = s.marginal_evaluate(reads, nwk)
    # print(likelihood)

    