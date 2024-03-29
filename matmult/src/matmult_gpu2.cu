#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>

#define BLOCK_SIZE 16

__global__ void matmult_gpu2Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C);

extern "C" {
void matmult_gpu2(int m, int n, int k, double * A, double * B, double * C){
	double * d_A, * d_B, * d_C;

	cudaMalloc((void **)&d_A, m * k * sizeof(double *));
	cudaMalloc((void **)&d_B, k * n * sizeof(double *));
	cudaMalloc((void **)&d_C, m * n * sizeof(double *));

	cudaMemcpy(d_A, A, m * k * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(double *), cudaMemcpyHostToDevice);

	//kernel block and grid size
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 dimGrid((int)ceil(((double)n)/BLOCK_SIZE), (int)ceil(((double)m)/BLOCK_SIZE));  

    matmult_gpu2Kernel<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, m * n * sizeof(double *), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
}

__global__ void matmult_gpu2Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C){

    int i, j, l;
    double C = 0;
    j = blockIdx.x * blockDim.x + threadIdx.x;
    i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(i < m && j < n){
		for(l=0;l < k;l++){
			C += d_A[i*k + l] * d_B[l*n + j];
		}
		d_C[i*n + j] = C;
	}

}
