#include <cuda_runtime.h>

#define REGISTER_BLOCKING 24

__global__ void matmult_gpu4Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C);

// REGISTER BLOCKING ALONG THE ROWS OF C
/*
extern "C" {
void matmult_gpu4(int m, int n, int k, double * A, double * B, double * C){
	double * d_A, * d_B, * d_C;

	cudaMalloc((void **)&d_A, m * k * sizeof(double *));
	cudaMalloc((void **)&d_B, k * n * sizeof(double *));
	cudaMalloc((void **)&d_C, m * n * sizeof(double *));

	cudaMemcpy(d_A, A, m * k * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(double *), cudaMemcpyHostToDevice);

	//kernel block and grid size
    dim3 dimBlock(16,16,1);
    dim3 dimGrid((int)ceil(((double)m)/16), (int)ceil(((double)n)/(16*REGISTER_BLOCKING)));  

    matmult_gpu4Kernel<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, m * n * sizeof(double *), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
}

__global__ void matmult_gpu4Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C){

    int i, j, l, e;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = REGISTER_BLOCKING*(blockIdx.y * blockDim.y + threadIdx.y);
	
	double C_reg[REGISTER_BLOCKING] = {0}; 

	if(i < m && j < n){
		for(l=0;l < k;l++){
			C_reg[0] += d_A[i*k + l] * d_B[l*n + j];
			for(e = 1; e < REGISTER_BLOCKING; e++){
				if(j + e < n)
					C_reg[e] += d_A[i*k + l] * d_B[l*n + j + e];
			}
		}
		d_C[i*n + j] = C_reg[0];
		for(e = 1; e < REGISTER_BLOCKING; e++){
			if(j + e < n)
				d_C[i*n + j + e] = C_reg[e];
		}
	}

}
*/


// REGISTER BLOCKING ALONG THE COLUMNS OF C
extern "C" {
void matmult_gpu4(int m, int n, int k, double * A, double * B, double * C){
	double * d_A, * d_B, * d_C;

	cudaMalloc((void **)&d_A, m * k * sizeof(double *));
	cudaMalloc((void **)&d_B, k * n * sizeof(double *));
	cudaMalloc((void **)&d_C, m * n * sizeof(double *));

	cudaMemcpy(d_A, A, m * k * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(double *), cudaMemcpyHostToDevice);

	//kernel block and grid size
    dim3 dimBlock(16,16,1);
    dim3 dimGrid((int)ceil(((double)n)/(16)), (int)ceil(((double)m)/(16*REGISTER_BLOCKING)));  

    matmult_gpu4Kernel<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, m * n * sizeof(double *), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
}

__global__ void matmult_gpu4Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C){

    int i, j, l, e;

    j = (blockIdx.x * blockDim.x + threadIdx.x);
    i = REGISTER_BLOCKING*(blockIdx.y * blockDim.y + threadIdx.y);
	
	double C_reg[REGISTER_BLOCKING] = {0}; 

	if(i < m && j < n){
		for(l=0;l < k;l++){
			C_reg[0] += d_A[i*k + l] * d_B[l*n + j];
			for(e = 1; e < REGISTER_BLOCKING; e++){
				if(i + e < m)
					C_reg[e] += d_A[(i+e)*k + l] * d_B[l*n + j];
			}
		}
		d_C[i*n + j] = C_reg[0];
		for(e = 1; e < REGISTER_BLOCKING; e++){
			if(i + e < m)
				d_C[(i+e)*n + j] = C_reg[e];
		}
	}

}

