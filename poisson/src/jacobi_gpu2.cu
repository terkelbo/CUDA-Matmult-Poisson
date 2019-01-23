#include "jacobi_gpu2.h"

__global__ void jacobi(int n, double h, double * u_old, double * u_new, double * f){


	int i = blockIdx.x * blockDim.x + threadIdx.x;  
	int j = blockIdx.y * blockDim.y + threadIdx.y;  

	if(i > 0 && i < (n+1) && j > 0 && j < (n+1)){
		u_new[i*(n + 2) + j] = 0.25*(u_old[(i-1)*(n + 2) + j] + u_old[(i+1)*(n + 2) + j] + u_old[i*(n + 2) + j-1] + u_old[i*(n + 2) + j+1] + h*h*f[i*(n + 2) + j]);
	}
	
		
	
}
