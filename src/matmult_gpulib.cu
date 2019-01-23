#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" {
void matmult_gpulib(int m, int n, int k, double * A, double * B, double * C){
   	cublasHandle_t handle;
  	cublasCreate(&handle);

  	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	double * d_A, * d_B, * d_C;

	cudaMalloc((void **)&d_A, m * k * sizeof(double *));
	cudaMalloc((void **)&d_B, k * n * sizeof(double *));
	cudaMalloc((void **)&d_C, m * n * sizeof(double *));

	cudaMemcpy(d_A, A, m * k * sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(double *), cudaMemcpyHostToDevice);

	cublasDgemm(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               m, n, k,
               alpha,
               d_A, m,
               d_B, k,
               beta,
               d_C, m);

    cudaMemcpy(C, d_C, m * n * sizeof(double *), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy(handle);
}
}
