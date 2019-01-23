#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include "inittools.h"
#include "jacobi.h"
#include <omp.h>

#define CACHE_LINE_SIZE 64

int
main( int argc, char *argv[] ){

	int i, j, n, k, max_it = 5000;
	char * algo, * test;
	double u_start = 0.0, d = 100000.0, threshold = 0.001;
	double memory, te, mflops;
	double * u_old, * u_new, * f, * temp;
	double * sol;

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
		//u_old = memalign(CACHE_LINE_SIZE, (n + 2)*(n + 2)* sizeof(double *));
		u_old = (double *)malloc((n + 2)*(n + 2)* sizeof(double *));
	}
	//u_new = memalign(CACHE_LINE_SIZE, (n + 2)*(n + 2)* sizeof(double *)); 
	//f = memalign(CACHE_LINE_SIZE, (n + 2)*(n + 2)* sizeof(double *));
	u_new = (double *)malloc((n + 2)*(n + 2)* sizeof(double *));
	f = (double *)malloc((n + 2)*(n + 2)* sizeof(double *));
	if(strcmp(algo,"jacobi")==0){
		if (u_old == NULL  || u_new == NULL | f == NULL) {
		    fprintf(stderr, "Memory allocation error...\n");
		    exit(EXIT_FAILURE);
		}
	}
	else{
		if (u_new == NULL | f == NULL) {
		    fprintf(stderr, "Memory allocation error...\n");
		    exit(EXIT_FAILURE);
		}
	}

	/* Initialize arrays */
	if(strcmp(test,"test")==0){
		//sol = memalign(CACHE_LINE_SIZE, (n + 2)*(n + 2)* sizeof(double *));
		sol = (double *)malloc((n + 2)*(n + 2)* sizeof(double *));
		init_u_test(n, algo, u_old, u_new);
		init_f_test(n, h, f);
		init_sol(n, h, sol);
	}
	else{
		init_u(n, algo, u_start, u_old, u_new);
		init_f(n, h, f);
	}
	
	te = omp_get_wtime();	
	/* Start the time loop */
	#pragma omp parallel default(none) shared(n,h,f,max_it,algo,temp,u_old,u_new) private(k,i,j)
	{
	//while(d > threshold && k < max_it){
	for(k = 0; k < max_it; k++){
		//solve one timestep using jacobi
		if(strcmp(algo,"jacobi")==0){
			jacobi(n, h, u_old, u_new, f); 	
		}

		
		//calculate 2-norm between old and new
		//d = euclidian_norm(n, u_old, u_new);

		//now that the values are updated we copy new values into the old array
		#pragma omp single
		{
		if(strcmp(algo,"jacobi")==0){
			temp = u_old;
			u_old = u_new;
			u_new = temp;	
			}
		}
	}
	}
	te = omp_get_wtime() - te;
	mflops   = 1.0e-06*n*n*max_it*CHECK_FLOP/te;
	memory = (double)(8.0*n*n)/1000; // in Kbytes

	if(strcmp(test,"test")==0){
		d = euclidian_norm(n, sol, u_new);
		printf("Mean euclidian norm between sol and approximation is %f \n", d/(n*n));
	}

	printf("%10.2li %10.2lf %le %le\n", 
	   max_it, memory, mflops, te);

	if(strcmp(algo,"jacobi")==0){
		free(u_old);
	}
	free(u_new);
	free(f);
	if(strcmp(test,"test")==0){
		free(sol);
	}
}
