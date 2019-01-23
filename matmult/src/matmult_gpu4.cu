#include <cuda_runtime.h>

__global__ void matmult_gpu4Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C);


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
    dim3 dimGrid((int)ceil(((double)m)/16), (int)ceil(((double)n)/64));  

    matmult_gpu4Kernel<<<dimGrid,dimBlock>>>(m, n, k, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, m * n * sizeof(double *), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
}

__global__ void matmult_gpu4Kernel(int m, int n, int k, double * d_A, double * d_B, double * d_C){

    int i, j, l;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = 4*(blockIdx.y * blockDim.y + threadIdx.y);
	
    double C_reg[4] = {0, 0, 0, 0}; 

	if(i < m && j < n){
		for(l=0;l < k;l++){
			C_reg[0] += d_A[i*k + l] * d_B[l*n + j];
			if(j + 1 < n)
				C_reg[1] += d_A[i*k + l] * d_B[l*n + j + 1];
			if(j + 2 < n)
				C_reg[2] += d_A[i*k + l] * d_B[l*n + j + 2];
			if(j + 3 < n)
				C_reg[3] += d_A[i*k + l] * d_B[l*n + (j + 3)];
		}
		d_C[i*n + j] = C_reg[0];
		if(j + 1 < n)
			d_C[i*n + j + 1] = C_reg[1];
		if(j + 2 < n)
			d_C[i*n + j + 2] = C_reg[2];
		if(j + 3 < n)
			d_C[i*n + j + 3] = C_reg[3];

	}

}

