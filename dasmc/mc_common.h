/*
 * common.h
 *
 *  Created on: Sep 23, 2008
 *      Author: noah
 */

#ifndef COMMON_H_
#define COMMON_H_
#include <stdio.h>
#include "../mc_sample.h"
#define cache_type float
#define MAX_CLASS 1024
enum { LINEAR=0, POLYNOMIAL, GAUSS };
#define CHECK_EQ(a, b)  \
                  if ((a) != (b)) { printf("Check failed: " #a " == " #b " in file %s at line %d.",  __FILE__, __LINE__) ; exit(1);}
#define CHECK_LE(a, b)  \
                  if ((a) > (b)) { printf("Check failed: " #a " <= " #b " in file %s at line %d",  __FILE__, __LINE__) ; exit(1);}

#define _SEQ
typedef struct _data {
	int *y;
	int **ix;
	float **x;
	int *lx;
	double *sqnorm;

	int m;
	int n;
	int k;
} DATA;

typedef struct _node
{
	_node *prev, *next;   // a circular double-linked list
	cache_type *data;
	int col;
} NODE;

typedef struct _cache {
	NODE** nodes; // the same length as #sample
	NODE head;

	int m;
	int cache_size;
	int max_cols;
	int free_cols;
} CACHE;

typedef struct _kernel {
	int type;
	/* RBF: exp(-\gamma\|x-y\|) */
	double gamma;
	/* POLYNOMIAL: (s(x^Ty)+r)^d */
	double s;
	double r;
	int d;
	/* data set used in kernel */
	McSample* data;
	/* for speed up */
	int row;
	float* x;
	/* cache used in kernel */
	CACHE* cache;
	/* kernel function */
	typedef double (*kernel_func)(_kernel*, int , int);
	kernel_func func;
} KERNEL;

typedef struct _learn_param {
	double C;
	int nsp;
	int nc;
	double epsilon;
	double epsilonuse;
	/*epsilon for sub-solver*/
	double epsilon0;
	double epsilon00;
	double epsilon0use;
	double epsilon_alga;
	double ratio;
	int projector;
	/*epsilon for support vector (machine accurate)*/
	double epsilon_sv;

	double init_threshold;
	int cache_size; // MB
	int tail_one;
	int check_point;
	int max_iteration;
	int time_limit;
	int use_init;
	int subsolver;
	int steps_to_alter;
	int ws_selector;
} LEARN_PARAM;

typedef struct _model {

	/* kernel related */
	int type;
	double gamma;
	double s;
	double r;
	int d;

	/* data info */
	int m;
	int n;
	int k;
	int tail_one;
	/* supported vectors */
	int nsv;
	double* support_alpha; /* k*nsv */
	cache_type** sv_x; /* nsv */
	int** sv_ix; /* nsv */
	int* sv_lx; /* nsv */
	
	/* sparse structure */
	int* nnz_alpha; /* k */
	int** nz_alpha_ind; /* k*x */
	double** nz_alpha_val; /* k*x */
	
} MODEL;

void dump_param(LEARN_PARAM* param, KERNEL* knl);
double dot(KERNEL* kernel, int i, int j);
double linear(KERNEL* kernel, int i, int j);
double rbf(KERNEL* kernel, int i, int j);
double polynomial(KERNEL* kernel, int i, int j);
void create_flag_file(const char* file_name);
#endif /* COMMON_H_ */
