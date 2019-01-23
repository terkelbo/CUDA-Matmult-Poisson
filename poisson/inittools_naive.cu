#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "inittools_naive.h"

#define M_PI 3.14159265358979323846

void
init_u(int n, char * algo, double u_start, double * u_old, double * u_new){

	int i, j;

	for(i = 0; i < (n + 2); i++){
		for(j = 0; j < (n + 2); j++){
			if(j == (n + 1) || i == 0 || i == (n + 1)){
				if(strcmp(algo,"jacobi")==0){
					u_old[i*(n + 2) + j] = 20.0;
				}
				u_new[i*(n + 2) + j] = 20.0;
			} 
			else if(j == 0){
				if(strcmp(algo,"jacobi")==0){
					u_old[i*(n + 2) + j] = 0.0;
				}
				u_new[i*(n + 2) + j] = 0.0;
			}
			else{
				if(strcmp(algo,"jacobi")==0){
					u_old[i*(n + 2) + j] = u_start;
				}
				u_new[i*(n + 2) + j] = u_start;
			}
		}
	}
}

void
init_u_test(int n, char * algo, double * u_old, double * u_new){

	int i, j;

	for(i = 0; i < (n + 2); i++){
		for(j = 0; j < (n + 2); j++){
			if(strcmp(algo,"jacobi")==0){
				u_old[i*(n + 2) + j] = 0.0;
			}
			u_new[i*(n + 2) + j] = 0.0;
		}
	}
}

void
init_f(int n, double h, double * f){

	int i, j;

	for(i = 0; i < (n + 2); i++){
		for(j = 0; j < (n + 2); j++){
			if((double)(-1 + i*h) >= 0.0 && (double)(-1 + i*h) <= 0.33 && (double)(-1 + j*h) >= -0.66 && (double)(-1 + j*h) <= -0.33){
				f[i*(n + 2) + j] = 200;
			}
			else{
				f[i*(n + 2) + j] = 0;
			}
		}
	}
}

void
init_f_test(int n, double h, double * f){

	int i, j;
	double x, y;

	for(i = 0; i < (n + 2); i++){
		for(j = 0; j < (n + 2); j++){
			x = -1 + h*i;
			y = -1 + h*j;
			f[i*(n + 2) + j] = 2*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y);
		}
	}
}

void
init_sol(int n, double h, double * sol){

	int i, j;
	double x, y;

	for(i = 0; i < (n + 2); i++){
		for(j = 0; j < (n + 2); j++){
			x = -1 + h*i;
			y = -1 + h*j;
			if(x == -1 || y == -1 || x == 1 || y == 1){
				sol[i*(n + 2) + j] = 0.0;
			}
			else{
				sol[i*(n + 2) + j] = sin(M_PI*x)*sin(M_PI*y);				
			}
		}
	}
}

/* Routine for calculating two norm differences between two arrays */
double
euclidian_norm(int n, double * u_old, double * u_new){
	int i, j;
	double sum = 0, diff;
	
	for(i = 0; i < (n + 2); i++){
		for(j = 0; j < (n + 2); j++){
			diff = (u_new[i*(n + 2) + j] - u_old[i*(n + 2) + j]);
			sum += diff*diff;
		}
	}
	sum = sqrt(sum);
	return(sum);
} 


