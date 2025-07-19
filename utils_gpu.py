import popgen
import numpy as np
import cupy as cp
import scipy as sp
from cupyx.scipy.special import logsumexp, log_softmax
from time import time
import popgen.utils
import pandas as pd

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


def load_tree(dirname, i):
    tree_file = f'{dirname}/results/trees_dir/trees.{i:04}'
    with open(tree_file) as f:
        nwk = f.readline().strip()
    tree = popgen.utils.from_newick(nwk)
    tree = popgen.BaseTree(root=tree.root.get_children()[1]) # exclude outgroup
    n = len(tree.get_leaves())
    rename_dict = {f'cell{i+1:04}': str(i+1) for i in range(n)}
    tree = popgen.utils.relabel(tree, name_map=rename_dict)
    tree.root.set_parent(None)
    return tree


def load_true_genotype(dirname, i):
    true_hap_file = f'{dirname}/trees_dir/true_haplotypes_dir/true_hap.{i:04}'
    haps = []
    cell_names = []
    with open(true_hap_file) as f:
        line = f.readline().strip()
        num_cells, num_sites = int(line.split()[0]), int(line.split()[1])
        num_cells = num_cells // 2 - 1 # exclude outgroup
        for i in range(num_cells):
            cell_name, hap_parental = f.readline().strip().split()
            cell_name, hap_maternal = f.readline().strip().split()
            hap = [f'{hap_parental[j]}{hap_maternal[j]}' for j in range(num_sites)]
            haps.append(hap)
            cell_names.append(cell_name[:-1])
    df = pd.DataFrame(index=cell_names, data=haps)
    # heter_columns = []
    # for i in range(num_sites):
    #     unique_gts = df[i].unique()
    #     if len(unique_gts) == 1 and unique_gts[0][0] == unique_gts[0][1]:
    #         print(unique_gts)
    #         continue
    #     else:
    #         heter_columns.append(i)
    # df = df[heter_columns]
    return df


def scan_for_deletion(genotype):
    sites = []
    for i in range(genotype.shape[1]):
        for gt in genotype[i].unique():
            if '-' in gt and i not in sites:
                sites.append(i)
    return sites 


def on_branch(leaves, cells):
    # if leaves == cells
    if len(leaves) != len(cells):
        return False
    for c in cells:
        if c not in leaves:
            return False
    return True


def find_deletion_on_tree(tree, geno, reads, del_site, verbose=False):
    # print('Delete site', del_site)
    DEL = 'DEL'
    OK = 'OK'
    tree = tree.copy()
    site = geno[del_site]
    read = reads[del_site]
    del_cells = site[site.apply(lambda x: '-' in x)]
    del_cells = del_cells.index.tolist()
    traversor = popgen.utils.TraversalGenerator()
    for node in traversor(tree):
        node.event = ''
        if node.is_leaf():
            node.event = f'{node.name} {node.event} [{site.loc[node.name]}] [{read[int(node.name[-4:])-1]}]'
        leaves = [n.name for n in node.get_leaves()]
        # print(leaves, del_cells)
        if on_branch(leaves, del_cells):
            node.event = f'{node.event} [{DEL}]'
            for des in node.get_descendants():
                des.event =  f'{des.event} [{DEL}]'
        # else:
        #     node.event = f'{node.event} [{OK}]'
    if verbose:
        tree.draw(nameattr='event')
    return tree
        
def read_vcf(vcf):
    order = ['A', 'C', 'G', 'T']
    # order: A, C, G, T
    data = []
    tg = []
    count = 0
    with open(vcf, 'r') as f:
        for line in f.readlines():
            data_site = []
            tg_site = []
            if line.startswith('##'):
                continue
            if line.startswith('#'):
                line = line.strip().split()
                cell_names = line[9: -1]
                continue
            line = line.strip().split()
            ref = line[3]
            for cell in line[9: -1]:
                info = cell.split(':')
                true_gt = info[-1]
                cn = 1 if '-' in true_gt else 2
                ml_gt = info[0]
                reads = [int(_) for _ in info[2].split(',')]
                ref_count = reads[order.index(ref)]
                alt_count = sum(reads) - ref_count
                # if '-' in true_gt:
                #     print(count, (ref_count, alt_count, cn))
                data_site.append((ref_count, alt_count, cn))
                tg_site.append(true_gt)
            data.append(data_site)
            tg.append(tg_site)
            count += 1
    
    return np.array(data, dtype=np.object_), pd.DataFrame(index=cell_names, data=np.array(tg, dtype=np.object_).T)


def get_scistreec_input_with_cn(dirname, i):
    vcf = f'{dirname}/results/vcf_dir/vcf.{i:04}'
    data, tg = read_vcf(vcf)
    tree = load_tree(dirname, i)
    return data, tree, tg


## -------------------- test code ----------------------------

def pairwise_distance_matrix(probs):
    ncell, nsite, num_states = probs.shape
    exp_probs = cp.exp(probs)
    expected_distances = cp.einsum('ipq,jpq->ij', exp_probs, 1-exp_probs)
    # print(expected_distances)
    return expected_distances

if __name__ == '__main__':
    disMatrix = [
    [0,	23,	27,	20],
    [23,	0,	30,	28],
    [27,	30,	0,	30],
    [20,	28,	30,	0],
    ]
    probs = cp.load('./probs.cpy.npy')
    print(probs)
    disMatrix = pairwise_distance_matrix(probs).get()
    print(disMatrix)
    tree = neighbor_joining_np(disMatrix)
    print(tree)

    # a = neighbor_joining(disMatrix)
    # print(a)