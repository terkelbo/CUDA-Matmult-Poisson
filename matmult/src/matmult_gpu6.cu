#include <cuda_runtime.h>

// Thread block size
#define BLOCK_SIZE 16

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    double* elements;
} Matrix;

// Get a matrix element
__device__ double GetElement1(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement1(Matrix A, int row, int col,
                           double value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix1(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

__global__ void matmult_gpu6Kernel(Matrix A, Matrix B, Matrix C);

extern "C" {
void matmult_gpu6(int m, int n, int k, double * A, double * B, double * C){


    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = k; d_A.height = m;
    size_t size = k * m * sizeof(double);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = n; d_B.height = k;
    size = n * k * sizeof(double);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = n; d_C.height = m;
    size = n * m * sizeof(double);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(n / dimBlock.x, m / dimBlock.y);
    matmult_gpu6Kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C, d_C.elements, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
}

// Matrix multiplication kernel called by MatMul()
 __global__ void matmult_gpu6Kernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int m = C.height;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix1(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    double C_reg[4] = {0};

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix1(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix1(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement1(Asub, row, col);
        Bs[row][col] = GetElement1(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e){
        	C_reg[0] += As[row][e] * Bs[e][col];
			for(int ei = 1; ei < 4; ei++){
				if(row + ei < m)
					C_reg[ei] += As[row+ei][e] * Bs[e][col];
			}
		}

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement1(Csub, row, col, C_reg[0]);
    for(int ei = 1; ei < 4; ei++){
		if(row + ei < m)
			SetElement1(Csub, row + ei, col, C_reg[ei]);
	}
}
