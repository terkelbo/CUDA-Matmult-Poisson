#ifndef __JACOBI_GPU1_H
#define __JACOBI_GPU1_H

__global__ void jacobi(int n, double h, double * u_old, double * u_new, double * f);

#define CHECK_FLOP 7
#endif
