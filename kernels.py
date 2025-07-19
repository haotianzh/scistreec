import cupy as cp 


kernel_log_probability = r'''
extern "C" __global__
void compute_genotype_log_probs(
    float* ref, float* alt, float* cn,
    float* afs, float* out,
    float ado, float seqerr,
    int ncell, int nsite,
    int CN_MAX, int CN_MIN, int N)
{
    float EPS = 1e-20f;
    float NEG_INF = -1.0f / 0.0f;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nsite * ncell;
    if (tid >= total) 
        return;

    int site = tid / ncell;
    float refc = ref[tid];
    float altc = alt[tid];
    int copy = cn[tid];
    float af = afs[site];
 
    af = fmaxf(af, EPS);
    if (copy < 0 || copy > 2 * CN_MAX) return;

    float logado = logf(fmaxf(ado, EPS));
    float log1mado = logf(fmaxf(1.0f - ado, EPS));

    float p00 = 1.0f - seqerr, p01 = seqerr;
    float p10 = seqerr, p11 = 1.0f - seqerr;
    float log_probs[210];
    for (int i = 0; i < N; ++i) log_probs[i] = NEG_INF;

    float maxval = NEG_INF;

    for (int g0 = 0; g0 <= CN_MAX; ++g0) {
        for (int g1 = 0; g1 <= CN_MAX; ++g1) {
            if (g0 + g1 != copy) continue;
            if (g0 == 0 && g1 == 0) continue;

            float log_af = logf(fmaxf(af, EPS));
            float log_1maf = logf(fmaxf(1.0f - af, EPS));
            float prior = lgammaf(copy + 1.0f) - lgammaf(g0 + 1.0f) - lgammaf(g1 + 1.0f)
                        + g0 * log_af + g1 * log_1maf;

            float acc_log = NEG_INF;

            for (int g0_ = 0; g0_ <= g0; ++g0_) {
                for (int g1_ = 0; g1_ <= g1; ++g1_) {
                    if (g0_ == 0 && g1_ == 0) continue;

                    float q = (float)g0_ / (g0_ + g1_);
                    float prob_ref = fmaxf(q * p00 + (1.0f - q) * p10, EPS);
                    float prob_alt = fmaxf(q * p01 + (1.0f - q) * p11, EPS);

                    float pread = refc * logf(prob_ref) + altc * logf(prob_alt);

                    float lw = lgammaf(g0 + 1.0f) - lgammaf(g0_ + 1.0f) - lgammaf(g0 - g0_ + 1.0f)
                             + lgammaf(g1 + 1.0f) - lgammaf(g1_ + 1.0f) - lgammaf(g1 - g1_ + 1.0f)
                             + g0_ * log1mado + (g0 - g0_) * logado
                             + g1_ * log1mado + (g1 - g1_) * logado;

                    float val = pread + lw;
                    acc_log = (val > acc_log)
                        ? val + log1pf(expf(acc_log - val))
                        : acc_log + log1pf(expf(val - acc_log));
                }
            }

            int index = ((g0 + g1) * (g0 + g1 + 1)) / 2 + g0 - (CN_MIN * (CN_MIN + 1)) / 2;
            log_probs[index] = prior + acc_log;

            if (log_probs[index] > maxval) maxval = log_probs[index];
        }
    }

    float sumexp = 0.0f;
    for (int i = 0; i < N; ++i) {
        if (log_probs[i] > NEG_INF) {
            sumexp += expf(log_probs[i] - maxval);
        }
    }

    float logZ = maxval + logf(fmaxf(sumexp, EPS));

    for (int i = 0; i < N; ++i) {
        out[tid * N + i] = log_probs[i] - logZ;
    }
}
'''

# kernel_log_probability = r'''
# extern "C" __global__
# void compute_genotype_log_probs(
#     const float* ref, const float* alt, const float* cn,
#     const float* afs, float* out,
#     const float ado, const float seqerr,
#     const int ncell, const int nsite,
#     const int CN_MAX, const int CN_MIN, const int N)
# {
#     const float EPS = 1e-20f;
#     const float NEG_INF = -1.0f / 0.0f;

#     int tid = blockIdx.x * blockDim.x + threadIdx.x;
#     int total = nsite * ncell;
#     if (tid >= total) 
#         return;
#     int site = tid / ncell;
#     int cell = tid % ncell;
#     int idx = site * ncell + cell;

#     float refc = ref[idx];
#     float altc = alt[idx];
#     int copy = cn[idx];
#     float af = afs[site];

#     af = fmaxf(af, EPS);
#     if (copy < 0 || copy > 2 * CN_MAX) return;

#     float logado = logf(fmaxf(ado, EPS));
#     float log1mado = logf(fmaxf(1.0f - ado, EPS));

#     float p00 = 1.0f - seqerr, p01 = seqerr;
#     float p10 = seqerr, p11 = 1.0f - seqerr;

#     const int MAX_N = 210; 
#     float log_probs[MAX_N];
#     for (int i = 0; i < N; ++i) log_probs[i] = NEG_INF;

#     float maxval = NEG_INF;

#     for (int g0 = 0; g0 <= CN_MAX; ++g0) {
#         for (int g1 = 0; g1 <= CN_MAX; ++g1) {
#             if (g0 + g1 != copy) continue;
#             if (g0 == 0 && g1 == 0) continue;

#             float log_af = logf(fmaxf(af, EPS));
#             float log_1maf = logf(fmaxf(1.0f - af, EPS));
#             float prior = lgammaf(copy + 1.0f) - lgammaf(g0 + 1.0f) - lgammaf(g1 + 1.0f)
#                         + g0 * log_af + g1 * log_1maf;

#             float acc_log = NEG_INF;

#             for (int g0_ = 0; g0_ <= g0; ++g0_) {
#                 for (int g1_ = 0; g1_ <= g1; ++g1_) {
#                     if (g0_ == 0 && g1_ == 0) continue;

#                     float q = (float)g0_ / (g0_ + g1_);
#                     float prob_ref = fmaxf(q * p00 + (1.0f - q) * p10, EPS);
#                     float prob_alt = fmaxf(q * p01 + (1.0f - q) * p11, EPS);

#                     float pread = refc * logf(prob_ref) + altc * logf(prob_alt);

#                     float lw = lgammaf(g0 + 1.0f) - lgammaf(g0_ + 1.0f) - lgammaf(g0 - g0_ + 1.0f)
#                              + lgammaf(g1 + 1.0f) - lgammaf(g1_ + 1.0f) - lgammaf(g1 - g1_ + 1.0f)
#                              + g0_ * log1mado + (g0 - g0_) * logado
#                              + g1_ * log1mado + (g1 - g1_) * logado;

#                     float val = pread + lw;
#                     acc_log = (val > acc_log)
#                         ? val + log1pf(expf(acc_log - val))
#                         : acc_log + log1pf(expf(val - acc_log));
#                 }
#             }

#             int index = ((g0 + g1) * (g0 + g1 + 1)) / 2 + g0 - (CN_MIN * (CN_MIN + 1)) / 2;
#             log_probs[index] = prior + acc_log;

#             if (log_probs[index] > maxval) maxval = log_probs[index];
#         }
#     }

#     float sumexp = 0.0f;
#     for (int i = 0; i < N; ++i) {
#         if (log_probs[i] > NEG_INF) {
#             sumexp += expf(log_probs[i] - maxval);
#         }
#     }

#     float logZ = maxval + logf(fmaxf(sumexp, EPS));

#     for (int i = 0; i < N; ++i) {
#         out[(cell * nsite + site) * N + i] = log_probs[i] - logZ;
#     }
# }
# '''

# mat1: (nsite, k) mat2: (k, k)
kernel_log_matmul = r'''
extern "C" __global__ void log_matmul(float* mat1, float* mat2, float* out, int n, int k){
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i < n && j < k) {
        out[i * k + j] = logf(0.0f); 
        for (int p=0; p<k; p++){
            float m = max(out[i*k+j], mat1[i*k+p] + mat2[j*k+p]);
            if (!__isinf(m))
            out[i*k+j] = m + logf(expf(out[i*k+j] - m) + expf(mat1[i*k+p] + mat2[j*k+p] - m));
        }   
    }
}
'''


def compute_genotype_log_probs():
    return cp.RawKernel(kernel_log_probability, 'compute_genotype_log_probs')


def log_matmul_cuda():
    return cp.RawKernel(kernel_log_matmul, 'log_matmul')


if __name__ == "__main__":
    cp.random.seed(42)

    a = cp.abs(cp.random.rand(50, 10))
    # a = cp.array([[cp.e, cp.e, cp.e]], dtype=cp.float32)
    b = cp.abs(cp.eye(10))
    # b = cp.array([[1e-3, 1e-3, 1-2e-3], [1e-3, 1e-3, 1-2e-3], [1e-3, 1e-3, 1-2e-3]], dtype=cp.float32)
    # b = cp.array([])

    a = cp.asarray(a, dtype=cp.float32) 
    a = a / cp.sum(a, axis=-1, keepdims=True)
    
    b = cp.asarray(b, dtype=cp.float32)
    # print(b)
    b = b / cp.sum(b, axis=-1)
    

    loga = cp.log(a)
    logb = cp.log(b)

    k = 1
    # ------------ test cuda -------------
    
    for i in range(k):
        c = cp.zeros([50, 10], dtype=cp.float32)
        # print(loga)
        log_matmul_cuda()((2, 1), (32, 32), (loga, logb, c, 50, 10))
        loga = c
        

    # ------------ standard --------------
    for i in range(k):
        a = cp.matmul(b, a.T).T



    print(c)
    print(cp.log(a))
    print(cp.isclose(c, cp.log(a)))