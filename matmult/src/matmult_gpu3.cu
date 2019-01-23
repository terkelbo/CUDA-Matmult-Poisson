#include <cuda_runtime.h>

__global__ void matmult_gpu3Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C);

/*
extern "C" {
void matmult_gpu3(int m, int n, int k, double * A, double * B, double * C){
	double * d_A, * d_B, * d_C;

	cudaMalloc((void **)&d_A, m * k * sizeof(double *));
	cudaMalloc((void **)&d_B, k * n * sizeof(double *));
	cudaMalloc((void **)&d_C, m * n * sizeof(double *));

	cudaMemcpy(d_A, A, m * k * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(double *), cudaMemcpyHostToDevice);

	//kernel block and grid size
    dim3 dimBlock(16,16,1);
    dim3 dimGrid((int)ceil(((double)m)/32), (int)ceil(((double)n)/16));  

    matmult_gpu3Kernel<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, m * n * sizeof(double *), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
}


__global__ void matmult_gpu3Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C){

    int i, j, l;

    i = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    j = blockIdx.y * blockDim.y + threadIdx.y;
	
    double C_reg[2] = {0, 0}; 

	if(i < m && j < n){
		for(l=0;l < k;l++){
			C_reg[0] += d_A[i*k + l] * d_B[l*n + j];
			if(i + 1 < m)
				C_reg[1] += d_A[(i + 1)*k + l] * d_B[l*n + j];
		}
		d_C[i*k + j] = C_reg[0];
		if(i + 1 < m)
			d_C[(i+1)*k + j] = C_reg[1];

	}

}
*/

extern "C" {
void matmult_gpu3(int m, int n, int k, double * A, double * B, double * C){
	double * d_A, * d_B, * d_C;

	cudaMalloc((void **)&d_A, m * k * sizeof(double *));
	cudaMalloc((void **)&d_B, k * n * sizeof(double *));
	cudaMalloc((void **)&d_C, m * n * sizeof(double *));

	cudaMemcpy(d_A, A, m * k * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(double *), cudaMemcpyHostToDevice);

	//kernel block and grid size
    dim3 dimBlock(16,16,1);
    dim3 dimGrid((int)ceil(((double)m)/16), (int)ceil(((double)n)/32));  

    matmult_gpu3Kernel<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, m * n * sizeof(double *), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
}

__global__ void matmult_gpu3Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C){

    int i, j, l;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = 2*(blockIdx.y * blockDim.y + threadIdx.y);
	
    double C_reg[2] = {0, 0}; 

	if(i < m && j < n){
		for(l=0;l < k;l++){
			C_reg[0] += d_A[i*k + l] * d_B[l*n + j];
			if(j + 1 < n)
				C_reg[1] += d_A[i*k + l] * d_B[l*n + (j + 1)];
		}
		d_C[i*n + j] = C_reg[0];
		if(j + 1 < n)
			d_C[i*n + j + 1] = C_reg[1];
	}

}

