#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "inittools_gpu3.h"
#include "jacobi_gpu3.h"
#include <omp.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CACHE_LINE_SIZE 64

int
main( int argc, char *argv[] ){

	int i, j, n, k, max_it = 5000;
	char * algo, * test;
	double u_start = 0.0, d = 100000.0, threshold = 0.001;
	double memory, te, mflops, bandwidth;
	double * h_u_old, * h_u_new, * h_f, * h_temp;
	double * sol;
	double * d0_u_old, * d0_u_new, * d0_f, * d0_temp;
	double * d1_u_old, * d1_u_new, * d1_f, * d1_temp;
	
	algo = "jacobi";
	if(argc >= 2){
		n = atoi(argv[1]);
	}
	else{
		n = 100;
	}
	double h = 2.0/(n + 1);
	if(argc >= 3){
		algo = argv[2];
	}
	else{
		algo = "jacobi";
	}
	if(argc >= 4){
		test = argv[3];
	}
	else{
		test = "notest";
	}
	

	/* Allocate memory for all arrays */
	if(strcmp(algo,"jacobi")==0){
		cudaMallocHost((void **)&h_u_old,(n + 2)*(n + 2)* sizeof(double *));
	}
	
	cudaMallocHost((void **)&h_f,(n + 2)*(n + 2)* sizeof(double *));
	cudaMallocHost((void **)&h_u_new,(n + 2)*(n + 2)* sizeof(double *));
	
	if(strcmp(algo,"jacobi")==0){
		if (h_u_old == NULL  || h_u_new == NULL | h_f == NULL) {
		    fprintf(stderr, "Memory allocation error...\n");
		    exit(EXIT_FAILURE);
		}
	}
	else{
		if (h_u_new == NULL | h_f == NULL) {
		    fprintf(stderr, "Memory allocation error...\n");
		    exit(EXIT_FAILURE);
		}
	}

	/* Initialize arrays */
	if(strcmp(test,"test")==0){
		sol = (double *)malloc((n + 2)*(n + 2)* sizeof(double *));
		init_u_test(n, algo, h_u_old, h_u_new);
		init_f_test(n, h, h_f);
		init_sol(n, h, sol);
	}
	else{
		init_u(n, algo, u_start, h_u_old, h_u_new);
		init_f(n, h, h_f);
	}
	
	double * dummy;
	cudaMalloc((void **)&dummy,0);
	     
	te = omp_get_wtime();
		
	cudaSetDevice(0);
	cudaDeviceEnablePeerAccess(1, 0);
	cudaMalloc((void **)&d0_u_old,  (n + 2)*(n + 2)* sizeof(double *)/2);
	cudaMalloc((void **)&d0_u_new,  (n + 2)*(n + 2)* sizeof(double *)/2);
	cudaMalloc((void **)&d0_f,  (n + 2)*(n + 2)* sizeof(double *)/2);
	cudaMalloc((void **)&d0_temp,  (n + 2)*(n + 2)* sizeof(double *)/2);
	cudaMemcpy(d0_u_new, h_u_new, (n + 2)*(n + 2)* sizeof(double *)/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d0_u_old, h_u_old, (n + 2)*(n + 2)* sizeof(double *)/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d0_f, h_f, (n + 2)*(n + 2)* sizeof(double *)/2, cudaMemcpyHostToDevice);
	
	cudaSetDevice(1);
	cudaDeviceEnablePeerAccess(0, 0);
	cudaMalloc((void **)&d1_u_old,  (n + 2)*(n + 2)* sizeof(double *)/2);
	cudaMalloc((void **)&d1_u_new,  (n + 2)*(n + 2)* sizeof(double *)/2);
	cudaMalloc((void **)&d1_f,  (n + 2)*(n + 2)* sizeof(double *)/2);
	cudaMalloc((void **)&d1_temp,  (n + 2)*(n + 2)* sizeof(double *)/2);
	cudaMemcpy(d1_u_new, h_u_new+(n+2)/2, (n + 2)*(n + 2)* sizeof(double *)/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d1_u_old, h_u_old+(n+2)/2, (n + 2)*(n + 2)* sizeof(double *)/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d1_f, h_f+(n+2)/2, (n + 2)*(n + 2)* sizeof(double *)/2, cudaMemcpyHostToDevice);
	
	// Kernel launch
	dim3 dimGrid((int)ceil(((double)(n+2))/(16*2)),(int)ceil(((double)(n+2))/(16*2)),1);
	dim3 dimBlock(16,16,1);
	
	for(k = 0; k < max_it; k++){
		
		cudaSetDevice(0);
		jacobi_kernel1<<<dimGrid,dimBlock>>>(n, h, d0_u_old, d0_u_new, d0_f, d1_u_old, d1_u_new);		
		
		cudaSetDevice(1);
		jacobi_kernel2<<<dimGrid,dimBlock>>>(n, h, d1_u_old, d1_u_new, d1_f, d0_u_old, d0_u_new);
		cudaDeviceSynchronize();		

		//now that the values are updated we copy new values into the old array
		if(strcmp(algo,"jacobi")==0){
			d0_temp = d0_u_old;
			d0_u_old = d0_u_new;
			d0_u_new = d0_temp;	
			d1_temp = d1_u_old;
			d1_u_old = d1_u_new;
			d1_u_new = d1_temp;	
			}		
	}
	
	cudaMemcpy(h_u_new, d0_u_new, (n+2)*(n+2)*sizeof(double *)/2, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_u_new+(n+2)/2, d1_u_new, (n+2)*(n+2)*sizeof(double *)/2, cudaMemcpyDeviceToHost);
	
	
	te = omp_get_wtime() - te;
	mflops   = 1.0e-06*CHECK_FLOP*n*n*max_it/te;
	memory = (double)(8.0*n*n)/1000; // in Kbytes
	bandwidth = 1.0e-06*n*n*8*3*max_it/te;

	if(strcmp(test,"test")==0){
		d = euclidian_norm(n, sol, h_u_new);
		printf("Mean euclidian norm between sol and approximation is %f \n", d/(n*n));
	}

	printf("%10.2li %10.2lf %.3f %.3f %.3f\n", 
	   max_it, memory, mflops, te, bandwidth);

	if(strcmp(algo,"jacobi")==0){
		cudaFree(d0_u_old);
		cudaFree(d0_f);
		cudaFree(d0_u_new);
		cudaFree(d1_u_old);
		cudaFree(d1_f);
		cudaFree(d1_u_new);
	}
	cudaFreeHost(h_f);
	cudaFreeHost(h_u_new);
	cudaFreeHost(h_u_old);
	if(strcmp(test,"test")==0){
		free(sol);
	}
}
