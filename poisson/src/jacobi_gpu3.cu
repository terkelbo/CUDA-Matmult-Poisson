#include "jacobi_gpu3.h"

__global__ void jacobi_kernel1(int n, double h, double * u0_old, double * u0_new, double * f, double * u1_old, double * u1_new){


	int i = blockIdx.x * blockDim.x + threadIdx.x;  
	int j = blockIdx.y * blockDim.y + threadIdx.y;  

	if(i > 0 && i < (n+1)/2 && j > 0 && j < (n+1)/2){
	
		if (i==(n+2)/2){
			u0_new[i*(n + 2) + j] = 0.25*(u0_old[(i-1)*(n + 2) + j] + u1_old[j] + u0_old[i*(n + 2) + j-1] + u0_old[i*(n + 2) + j+1] + h*h*f[i*(n + 2) + j]);
		}
		else{
			u0_new[i*(n + 2) + j] = 0.25*(u0_old[(i-1)*(n + 2) + j] + u0_old[(i+1)*(n + 2) + j] + u0_old[i*(n + 2) + j-1] + u0_old[i*(n + 2) + j+1] + h*h*f[i*(n + 2) + j]);
		}
	
	}
		
	
}

__global__ void jacobi_kernel2(int n, double h, double * u1_old, double * u1_new, double * f, double * u0_old, double * u0_new){


	int i = blockIdx.x * blockDim.x + threadIdx.x;  
	int j = blockIdx.y * blockDim.y + threadIdx.y;  

	if(i > 0 && i < (n+1)/2 && j > 0 && j < (n+1)/2){
	
		if (i == 0){
			u1_new[i*(n + 2) + j] = 0.25*(u0_old[(n + 2)*(n + 2)/2 + j] + u1_old[(i+1)*(n + 2) + j] + u1_old[i*(n + 2) + j-1] + u1_old[i*(n + 2) + j+1] + h*h*f[i*(n + 2) + j]);
		}
		else{
			u1_new[i*(n + 2) + j] = 0.25*(u1_old[(i-1)*(n + 2) + j] + u1_old[(i+1)*(n + 2) + j] + u1_old[i*(n + 2) + j-1] + u1_old[i*(n + 2) + j+1] + h*h*f[i*(n + 2) + j]);
		}
	
	}
}
