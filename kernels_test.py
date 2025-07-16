import cupy as cp


zero_matrix_2d_kernel_code = r'''
extern "C" __global__ void zero_matrix_2d(float* out, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows && col < num_cols) {
        out[row * num_cols + col] = 0.0f; 
    }
}
'''

zero_matrix_2d_kernel = cp.RawKernel(zero_matrix_2d_kernel_code, 'zero_matrix_2d')


MATRIX_ROWS = 32
MATRIX_COLS = 3


my_matrix = cp.ones([MATRIX_ROWS, MATRIX_COLS], dtype=cp.float32)


TPB_X = 16
TPB_Y = 16 
block_dims = (TPB_X, TPB_Y) 
num_blocks_x = (MATRIX_COLS + TPB_X - 1) // TPB_X
num_blocks_y = (MATRIX_ROWS + TPB_Y - 1) // TPB_Y
grid_dims = (num_blocks_x, num_blocks_y) # This tuple defines the grid of blocks


zero_matrix_2d_kernel((2,2), (16,16), (my_matrix, MATRIX_ROWS, MATRIX_COLS))
print(my_matrix)