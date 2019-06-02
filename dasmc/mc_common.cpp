/*
 * common.cpp
 *
 *  Created on: Sep 23, 2008
 *      Author: noah
 */

#include <stdio.h>
#include <fstream>
#include <string.h>
#include <sstream>
#include <math.h>
#include <assert.h>
#include "mem_util.h"
#include "mc_common.h"
using namespace std;

void create_flag_file(const char* file_name)
{
	std::ofstream flag_strm;
	flag_strm.open(file_name);
	flag_strm << "End flag.\n";
	flag_strm.close();
}

void dump_param(LEARN_PARAM* param, KERNEL* knl)
{
	printf("#### parameters used ####\n");
	printf("Kernel parameters:\n");
	printf("    t: %d\n", knl->type);
	if(knl->type==GAUSS) {
		printf("    g: %lf\n", knl->gamma);
	}else if(knl->type==POLYNOMIAL) {
		printf("    s: %lf\n", knl->s);
		printf("    r: %lf\n", knl->r);
		printf("    d: %d\n", knl->d);
	}
	printf("Learning parameters:\n");
	printf("    C: %lf\n", param->C);
	printf("  nsp: %d\n", param->nsp);
	printf("   nc: %d\n", param->nc);
	printf("    e: %lf\n", param->epsilon);
	printf("    x: %lf\n", param->epsilon0);
	printf("    y: %lf\n", param->epsilon00);
	printf("    o: %lf\n", param->ratio);
	printf("    m: %d\n", param->cache_size);
	printf("    z: %d\n", param->check_point);
	printf("    p: %d\n", param->projector);
	printf("#########################\n");
}

#ifdef _SEQ
double dot(KERNEL* kernel, int i, int j) {
	const Example* a = kernel->data->get(i);
	const Example* b = kernel->data->get(j);
	int ia, ib;
	ia = ib = 0;
	float val=0;
	while(ia < a->n && ib < b->n) {
		if(a->ix[ia] < b->ix[ib]) {
			ia += 1;
		} else if(a->ix[ia] > b->ix[ib]) {
			ib += 1;
		} else {
			val += a->x[ia] * b->x[ib];
			ia += 1; ib += 1;
		}
	}
	return val;
}
#else
double dot(KERNEL* kernel, int i, int j)
{
	// store in registers for better performance
	register int    k;
	register double acc;

	int n     = kernel->data->lx[j];
	int *ip   = kernel->data->ix[j];
	float *xp = kernel->data->x[j];
	// use the characteristic that we always compute Q_{i,j} for j=1,...m
	if (i != kernel->row)
	{
		for (k = 0; k < kernel->data->lx[kernel->row]; k++)
			kernel->x[kernel->data->ix[kernel->row][k]] = 0.0;
		kernel->row = i;

		for (k = 0; k < kernel->data->lx[i]; k++)
			kernel->x[kernel->data->ix[kernel->row][k]] = kernel->data->x[i][k];
	}

	acc = 0.0;
	for (k = 0; k < n; k++)
		acc += (double)(xp[k] * kernel->x[ip[k]]);
	
	return acc;
}
#endif

double linear(KERNEL* kernel, int i, int j)
{
	return dot(kernel, i, j);
}

double rbf(KERNEL* kernel, int i, int j)
{
	return exp(-kernel->gamma * ( kernel->data->sqnorm[i] + kernel->data->sqnorm[j] 
                            - 2*dot(kernel, i,j)));
}

double polynomial(KERNEL* kernel, int i, int j)
{
	register double tmp, val;
	register int p = kernel->d;
	tmp = dot(kernel, i, j) * kernel->s + kernel->r;
	val = 1.0;
	while (p) {
		while (!(p&1)) {
			p>>=1;
			tmp = tmp*tmp;
		}
		p>>=1;
		val=val*tmp;
		tmp=tmp*tmp;
	}
	return val;
}
