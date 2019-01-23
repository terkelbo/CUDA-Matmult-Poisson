#ifndef __INITTOOLS_H
#define __INITTOOLS_H

void init_u(int n, char * algo, double u_start, double *  u_old, double *  u_new);

void init_u_test(int n, char * algo, double *  u_old, double *  u_new);

void init_f(int n, double h, double *  f);

void init_f_test(int n, double h, double *  f);

void init_sol(int n, double h, double *  sol);

double euclidian_norm(int n, double *  u_old, double *  u_new);

#endif
