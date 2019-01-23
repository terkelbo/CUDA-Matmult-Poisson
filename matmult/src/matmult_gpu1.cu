#include <cuda_runtime.h>

__global__ void matmult_gpu1Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C);

extern "C" {
void matmult_gpu1(int m, int n, int k, double * A, double * B, double * C){
	double * d_A, * d_B, * d_C;

	cudaMalloc((void **)&d_A, m * k * sizeof(double *));
	cudaMalloc((void **)&d_B, k * n * sizeof(double *));
	cudaMalloc((void **)&d_C, m * n * sizeof(double *));

	cudaMemcpy(d_A, A, m * k * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(double *), cudaMemcpyHostToDevice);

    matmult_gpu1Kernel<<<1,1>>>(m, n, k, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, m * n * sizeof(double *), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
}

__global__ void matmult_gpu1Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C){

    int i, j, l;
    double x;
	
	
	for(i=0;i < m; i++){
		for(j=0;j<n;j++){
			d_C[i*n + j]=0;
		}
		for(l=0;l < k;l++){
			x = d_A[i*k + l];
			for(j=0;j < n; j++){
				d_C[i*n + j] += x * d_B[l*n + j];
			}
		}
	}

}
