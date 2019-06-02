/*
 * pgqp.h
 *
 *  Created on: Sep 23, 2008
 *      Author: noah
 */

#ifndef PGQP_H_
#define PGQP_H_

#include <stdio.h>
#ifndef INF
#define INF 1.E10
#endif

#ifndef MAX_DBL
#define MAX_DBL INF
#endif

#ifndef max
#define max(a,b) ((a)>(b)?(a):(b))
#endif

#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#endif

/**
 * Quadratic defination :
 * min 1/2 \sum_{i,j} K(i,j) <x_i, x_j> + \sum_{i} <x_i, g_i>
 * 		x_i \leq e_{y_i}
 *  	\sum_{r} x_{ir} = 0;
 */
typedef struct _qpdata {
   double *K;
   int *y;

   double *g;
   double *grad;
   double *x;

   int n;
   int m;
   int k;
} QPDATA;

/* Parameters for Spectral Projected Gradient */
typedef struct _spg_param {

	double alpha_min;
	double alpha_max;

	double sigma1;
	double sigma2;

	int M;

	double epsilon;
	double epsilon_kkt;
	double epsilon_2norm;
	double epsilon_alga;
	
	int max_iteration;
	int verbose;

	int init;
	int proj; /* 0: Sort 1: Dai-Fletcher 2: Pardalos */
	FILE* fp;
} SPG_PARAM;

typedef struct _Int2Double {
	int id;
	double val;
	bool operator()(const _Int2Double& i1, const _Int2Double& i2)
	{
		return i1.val < i2.val;
	}
} INT2DOUBLE;

void set_default_param(SPG_PARAM* param);
void allocate_qp(QPDATA* qp, int nx, int k);
void resize_qp(QPDATA* qp, int n);
void free_qp(QPDATA* qp);
int  feasible(QPDATA* qp, double epsilon);
void dump(QPDATA* qp);

/* projection */
int project(QPDATA* qp, double* d, double alpha, double* tx /*OUT*/);

int Sort_project(QPDATA* qp, int ind, double* tx /*IN/OUT*/);
int Pardalos_project(QPDATA* qp, int ind, double* tx /*IN/OUT*/);
int DF_project(QPDATA* qp, int ind, double* tx /*IN/OUT*/);
double quick_select(double *arr, int n);

/* just boundary projection, used by Dai-Fletcher projection method */
void bproject(QPDATA* qp, double lambda, double* tx, int ind, double* lmdx /*OUT*/);
/* calculate the residual of current x_lambda, used by Dai-Fletcher projection method */
double residual(QPDATA* qp, double* lmdx);

void init_grad(QPDATA* qp, double* g, int zero);
void calc_k_multi_d(QPDATA* qp, SPG_PARAM* param, double* d, double* auxd, double* kmd);
void update_grad(QPDATA* qp, double* g_prev, double alpha, double* kmd, double* g);
int  spg_solve( QPDATA* Q, SPG_PARAM* param );

double ddot(double* a, double* b, int n);
double inf_norm(double* a, int n);
double square_norm(double* a, int n);

#endif /* PGQP_H_ */
