#include <cuda_runtime.h>

#define BLOCK_SIZE 8

__global__ void matmult_gpu5Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C);

extern "C" {
void matmult_gpu5(int m, int n, int k, double * A, double * B, double * C){
	double * d_A, * d_B, * d_C;

	cudaMalloc((void **)&d_A, m * k * sizeof(double *));
	cudaMalloc((void **)&d_B, k * n * sizeof(double *));
	cudaMalloc((void **)&d_C, m * n * sizeof(double *));

	cudaMemcpy(d_A, A, m * k * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(double *), cudaMemcpyHostToDevice);

	//kernel block and grid size
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 dimGrid(m/BLOCK_SIZE, n/BLOCK_SIZE);  

    matmult_gpu5Kernel<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, m * n * sizeof(double *), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
}

__global__ void matmult_gpu5Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C){

    double * Asub, * Bsub, * Csub; 
    double C = 0;
    int blockRow, blockCol, row, col;

    blockRow = blockIdx.x;
    blockCol = blockIdx.y;
    row = threadIdx.x;
    col = threadIdx.y;

    //Index the block of C and save adress in Csub
    Csub = &d_C[n * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];

    for(int b = 0; b < (k / BLOCK_SIZE); ++b){

    	//Get the submatrix of A and B and save adresses
    	Asub = &d_A[k * BLOCK_SIZE * blockRow + BLOCK_SIZE * b];
    	Bsub = &d_B[n * BLOCK_SIZE * b + BLOCK_SIZE * blockCol];

    	//Store Asub and Bsub in shared memory
    	__shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    	__shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

    	//Each thread loads one element from the Asub and Bsub
    	As[row][col] = Asub[k * row + col];
    	Bs[row][col] = Bsub[n * row + col];

    	//Make sure all threads have loaded
    	__syncthreads();

    	for(int e = 0; e < BLOCK_SIZE; ++e){
    		C += As[row][e] * Bs[e][col];
    	}

    	//Sync to make sure all computations are done before we overwrite in shared memory
    	//on the next iteration
    	__syncthreads();

    }	

    Csub[n * row + col] = C;
}

