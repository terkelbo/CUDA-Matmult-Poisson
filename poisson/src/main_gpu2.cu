#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "inittools_gpu2.h"
#include "jacobi_gpu2.h"
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
	double * d_u_old, * d_u_new, * d_f, * d_temp;
	
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
	cudaMalloc((void **)&d_u_old,  (n + 2)*(n + 2)* sizeof(double *));
	cudaMalloc((void **)&d_u_new,  (n + 2)*(n + 2)* sizeof(double *));
	cudaMalloc((void **)&d_f,  (n + 2)*(n + 2)* sizeof(double *));
	cudaMalloc((void **)&d_temp,  (n + 2)*(n + 2)* sizeof(double *));
	
	/* Start the time loop */
	cudaMemcpy(d_u_new, h_u_new, (n + 2)*(n + 2)* sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_old, h_u_old, (n + 2)*(n + 2)* sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f, h_f, (n + 2)*(n + 2)* sizeof(double *), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_temp, h_temp, (n + 2)*(n + 2)* sizeof(double *), cudaMemcpyHostToDevice);

	// Kernel launch
	dim3 dimGrid((int)ceil(((double)(n+2))/16),(int)ceil(((double)(n+2))/16),1);
	dim3 dimBlock(16,16,1);
	
	for(k = 0; k < max_it; k++){
		
		
		jacobi<<<dimGrid,dimBlock>>>(n, h, d_u_old, d_u_new, d_f);
		cudaDeviceSynchronize();		
	
		
		//now that the values are updated we copy new values into the old array
		if(strcmp(algo,"jacobi")==0){
			d_temp = d_u_old;
			d_u_old = d_u_new;
			d_u_new = d_temp;	
			}		
	}
	cudaMemcpy(h_u_new, d_u_new, (n+2)*(n+2)*sizeof(double *), cudaMemcpyDeviceToHost);
	
	
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
		cudaFree(d_u_old);
		cudaFree(d_f);
		cudaFree(d_u_new);
	}
	cudaFreeHost(h_f);
	cudaFreeHost(h_u_new);
	cudaFreeHost(h_u_old);
	if(strcmp(test,"test")==0){
		free(sol);
	}
}
