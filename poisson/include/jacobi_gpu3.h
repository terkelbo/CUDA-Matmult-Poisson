#ifndef __JACOBI_GPU3_H
#define __JACOBI_GPU3_H

__global__ void jacobi_kernel1(int n, double h, double * u0_old, double * u0_new, double * f, double * u1_old, double * u1_new);

__global__ void jacobi_kernel2(int n, double h, double * u1_old, double * u1_new, double * f, double * u0_old, double * u0_new);

#define CHECK_FLOP 7
#endif
