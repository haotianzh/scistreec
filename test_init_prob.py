from scistreec_simple_gpu_modification import compute_genotype_log_probs, init_prob_leaves_gpu, get_scistreec_input_with_cn, ScisTreeC
import cupy as cp 
import numpy as np

CN_MAX = 2
CN_MIN = 0

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
    ref = cp.ascontiguousarray(ref)
    alt = cp.ascontiguousarray(alt)
    cn = cp.ascontiguousarray(cn)
    probs = cp.ascontiguousarray(probs)
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
    return cp.array(reads)



s = ScisTreeC(verbose=True)
reads = random_reads(n_leaves=500, n_sites=1000)
print(reads.shape)
## test gpu init
probs = init_prob_leaves_gpu(reads)
