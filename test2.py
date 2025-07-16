import cupy as cp
import time

# Use smaller matrices that don't saturate the GPU
N = 2048
num_streams = 20

streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
A = [cp.random.rand(N, N) for _ in range(num_streams)]
B = [cp.random.rand(N, N) for _ in range(num_streams)]
C = [None] * num_streams

# Streamed version
cp.cuda.Device().synchronize()
start = time.time()

for i in range(num_streams):
    with streams[i]:
        C[i] = cp.add(A[i], B[i])

for s in streams:
    s.synchronize()
cp.cuda.Device().synchronize()
end = time.time()

print(f"Streamed time: {end - start:.3f} sec")

# Sequential version
cp.cuda.Device().synchronize()
start = time.time()

for i in range(num_streams):
    C[i] = cp.add(A[i], B[i])
    cp.cuda.Device().synchronize()

end = time.time()
print(f"Sequential time: {end - start:.3f} sec")
