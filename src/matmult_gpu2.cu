#include <cuda_runtime.h>

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
    dim3 dimBlock(16,16,1);
    dim3 dimGrid((int)ceil(((double)m)/16), (int)ceil(((double)n)/16));  

    matmult_gpu2Kernel<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, m * n * sizeof(double *), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
}

__global__ void matmult_gpu2Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C){

    int i, j, l;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(i < n && j < m){
		d_C[i*m + j]=0;
		for(l=0;l < k;l++){
			d_C[i*m + j] += d_A[i*m + l] * d_B[l*k + j];
		}
	}

}
