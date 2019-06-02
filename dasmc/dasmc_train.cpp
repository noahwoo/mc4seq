/*
 * mcsvm_train.cpp
 *
 *  Created on: Sep 23, 2008
 *      Author: noah
 */
#include "dasmc_train.h"
#include <math.h>
#include <assert.h>
#include <time.h>
#include "mem_util.h"
#include <algorithm>

#ifndef _SEQ
#include "ooqp/QpGenSparseMa27.h"
#include "ooqp/GondzioSolver.h"
#include "ooqp/QpGenVars.h"
#include "ooqp/QpGenResiduals.h"
#include "ooqp/SimpleVector.h"
#include "ooqp/QpGenData.h"
#endif

using namespace std;

/* add the global variable for performance profile */
/* kernel and cache related */
static double lknl_eval = 0;
static long lknl_hit  = 0;
static long lknl_miss = 0;
static double lknl_req  = 0;
/* timing profile */
static double fcons_subqp = 0;
static double fsolv_subqp = 0;
static double fsele_workingset = 0;
static double fupdate_grad = 0;
static double fcheck_opt  = 0;
static double fknl_eval   = 0;
static double ftotal      = 0;
static double fupdate_grad_knl   = 0;
static double fupdate_grad_calcd = 0;
static double fupdate_grad_updat = 0;

static int * ws_from = NULL;

#define CHKPT "mcpsvm.ckp"
#define INTV 100
static FILE* ckpf = NULL;
/* end of global variable*/
static sub_qp_solver optimize_qp = NULL;
static int current_sub_solver = 0;
static int steps_use_ipm = 0;

int cache_init(CACHE* cache, int msize, int m)
{
	cache->m = m;
	cache->cache_size  = msize * (1<<20);
	cache->cache_size -= sizeof(NODE*)*m;
	cache->cache_size /= sizeof(cache_type);
	cache->max_cols    = cache->cache_size/m;
	if(cache->max_cols < 2) {
		cache->max_cols = 2;
		cache->cache_size = 2*m;
	}
	cache->free_cols = cache->max_cols;
	// calloc to initialize to zero
	cache->nodes     = (NODE**)mu_calloc(sizeof(NODE*), m);
	// init the double-link circle
	cache->head.next = &cache->head;
	cache->head.prev = &cache->head;
	return (cache->max_cols);
}

NODE* cache_lru_delete(CACHE* cache, NODE* node)
{
	/* delete the node right before head */
	node->next->prev = node->prev;
	node->prev->next = node->next;

	return node;
}

NODE* cache_lru_insert(CACHE* cache, NODE* node)
{
	/* insert the node right after head */
	node->next = cache->head.next;
	node->prev = &cache->head;
	node->next->prev = node;
	node->prev->next = node;

	return node;
}

/* get the cached data, return 1 if hit, 0 if miss */
int cache_get_data(CACHE* cache, int col, cache_type** data)
{
	NODE* node;
	/* if cached */
	if(cache->nodes[col] != NULL) {
		*data = cache->nodes[col]->data;
		/* move to the head of the lru double-linked queue */
		node = cache_lru_delete(cache, cache->nodes[col]);
		cache_lru_insert(cache, node);
		
		return (1);
	}

	/* if not cached, allocate space and cache it */
	if(cache->free_cols > 0) {
		node = (NODE*)mu_alloc(sizeof(NODE),1);
		node->data = (cache_type*)mu_alloc(sizeof(cache_type), cache->m);
		node->col  = col;
		cache_lru_insert(cache, node);
		*data = node->data;
		cache->nodes[col] = node;

		cache->free_cols -= 1;
	} else {
		node = cache_lru_delete(cache, cache->head.prev);
		cache->nodes[node->col] = NULL;
		/* printf("move column %d out and %d in cache.\n", node->col, col); */
		node->col  = col;
		cache_lru_insert(cache, node);
		*data = node->data;
		cache->nodes[col] = node;
	}
	return (0);
}

int cache_touch_data(CACHE* cache, int col, cache_type** data)
{
	/* touch if col was cached */
	if(cache->nodes[col] != NULL) {
		(*data) = cache->nodes[col]->data;
		return (1);
	}
	(*data) = NULL;
	return (0);
}

int cache_uninit(CACHE* cache)
{
	for(NODE* node = cache->head.next;
			node != &cache->head; node = node->next) {
		mu_free((void**)(&node->data));
	}
	mu_free((void**)(&cache->nodes));
	return 0;
}

int kernel_init(KERNEL* kernel, McSample* data, CACHE* cache)
{
	kernel->cache = cache;
	kernel->data  = data;
	switch(kernel->type) {
	case LINEAR:
		kernel->func = &linear;
		break;
	case GAUSS:
		kernel->func = &rbf;
		break;
	case POLYNOMIAL:
		kernel->func = &polynomial;
		break;
	default:
		kernel->func = &rbf;
		break;
	}
	
#ifndef _SEQ
	// init for speedup
	int i;
	kernel->x  = (float*)mu_calloc(sizeof(float), data->n);
	kernel->row = 0;
	for(i=0;i<data->lx[0];++i) {
		kernel->x[data->ix[0][i]]  = data->x[0][i];
	}
#endif
	return 0;
}

cache_type* kernel_get_col(KERNEL* kernel, int col)
{
	int i, m;
	cache_type* data;
	m = kernel->data->m;

	if(!cache_get_data(kernel->cache, col, &data)) {
		// fill up the data column

		clock_t t = clock();
		for(i=0;i<m;++i) {
			data[i] = kernel_eval(kernel, col, i);

		}
		fknl_eval += (double(clock()-t))/CLOCKS_PER_SEC;
		lknl_miss += 1;
		lknl_eval += m;

	}else{
		lknl_hit += 1;
	}
	lknl_req += m;
	return data;
}

double kernel_eval(KERNEL* kernel, int i, int j)
{
	return (*(kernel->func))(kernel, i, j);
}

void kernel_add(KERNEL* kernel, double* weight, int i, double multi)
{
	register int k;
#ifdef _SEQ
	const Example* exa = kernel->data->get(i);
	float*  x = exa->x;
	int* ix = exa->ix;
	int n   = exa->n;
#else
	float*  x = kernel->data->x[i];
	int* ix = kernel->data->ix[i];
	int n   = kernel->data->lx[i];
#endif
	for(k=0;k<n;++k){
		weight[ix[k]] += x[k]*multi;
	}
	return;
}

double kernel_product(KERNEL* kernel, double* weight, int j)
{
	register int k;
	double prod;
#ifdef _SEQ
	const Example* exa = kernel->data->get(j);
	float*  x = exa->x;
	int* ix = exa->ix;
	int n   = exa->n;
#else
	float*  x = kernel->data->x[j];
	int* ix = kernel->data->ix[j];
	int n   = kernel->data->lx[j];
#endif
	
	prod = 0.0;
	for(k=0;k<n;++k){
		prod += weight[ix[k]]*x[k];
	}
	return prod;
}

int write_checkpoint(FILE* fp, int steps, int m, int k, double* alpha, double* grad, 
		double eps_use, int q, int* ind_in, int* ind_out, int old_q, 
		int* ind_in_old, int* ind_out_old, int* ws_hist)
{
	/* int sz;
	sz = fwrite(&steps, sizeof(steps), 1, fp);
	printf("sz=%d,sizeof(steps)==%d\n",sz,sizeof(steps)); */
	
	CHECK_EQ(fwrite(&steps, sizeof(steps), 1, fp), 1);
	CHECK_EQ(fwrite(&m, sizeof(m), 1, fp), 1);
	CHECK_EQ(fwrite(&k, sizeof(k), 1, fp), 1);
	CHECK_EQ(fwrite(alpha, sizeof(alpha[0]), m*k, fp), (size_t)m*k);
	CHECK_EQ(fwrite(grad, sizeof(grad[0]), m*k, fp), (size_t)m*k);
	CHECK_EQ(fwrite(&eps_use, sizeof(eps_use), 1, fp), 1);
	CHECK_EQ(fwrite(&q, sizeof(q), 1, fp), 1);
	CHECK_EQ(fwrite(ind_in, sizeof(ind_in[0]), (q+1), fp), (size_t)(q+1));
	CHECK_EQ(fwrite(ind_out, sizeof(ind_out[0]), ((m-q)+1), fp), (size_t)((m-q)+1));
	CHECK_EQ(fwrite(&old_q, sizeof(old_q), 1, fp), 1);
	CHECK_EQ(fwrite(ind_in_old, sizeof(ind_in_old[0]), (old_q+1), fp), (size_t)(old_q+1));
	CHECK_EQ(fwrite(ind_out_old, sizeof(ind_out_old[0]), ((m-old_q)+1), fp), (size_t)((m-old_q)+1));
	CHECK_EQ(fwrite(ws_hist, sizeof(ws_hist[0]), (m+1), fp), (size_t)(m+1));
	return 0;
}

int read_checkpoint(FILE* fp, int* steps, int m_new, int* m, int k_new, int*k, 
		double* alpha, double* grad, double* eps_use,
		int q_new, int* q, int* ind_in, int* ind_out, 
		int* old_q, int* ind_in_old, int* ind_out_old, int* ws_hist)
{
	CHECK_EQ(fread(steps, sizeof(*steps), 1, fp), 1);
	CHECK_EQ(fread(m, sizeof(*m), 1, fp), 1);
	CHECK_EQ(*m,m_new);
	CHECK_EQ(fread(k, sizeof(*k), 1, fp), 1);
	CHECK_EQ(*k,k_new);
	CHECK_EQ(fread(alpha, sizeof(alpha[0]), (*m)*(*k), fp), (size_t)(*m)*(*k));
	CHECK_EQ(fread(grad, sizeof(grad[0]), (*m)*(*k), fp), (size_t)(*m)*(*k));
	CHECK_EQ(fread(eps_use, sizeof(*eps_use), 1, fp), 1);
	CHECK_EQ(fread(q, sizeof(*q), 1, fp), 1);
	CHECK_LE(*q,q_new);
	CHECK_EQ(fread(ind_in, sizeof(ind_in[0]), (*q)+1, fp), (size_t)(*q+1));
	CHECK_EQ(fread(ind_out, sizeof(ind_out[0]), (*m-*q+1), fp), (size_t)(*m-*q+1));
	CHECK_EQ(fread(old_q, sizeof(*old_q), 1, fp), 1);
	CHECK_EQ(fread(ind_in_old, sizeof(ind_in_old[0]), (*old_q+1), fp), (size_t)(*old_q+1));
	CHECK_EQ(fread(ind_out_old, sizeof(ind_out_old[0]), (*m-*old_q+1), fp), (size_t)(*m-*old_q+1));
	CHECK_EQ(fread(ws_hist, sizeof(ws_hist[0]), (*m+1), fp), (size_t)(*m+1));
	return 0;
}
void svm_learn(McSample* data, KERNEL* kernel, LEARN_PARAM* param,
		MODEL* model, double* _alpha) {
	/*This function implement the main logic for svm training*/
	int i, ii, r;
	int nsp, new_nsp, old_nsp, nc;
	int mk, m, new_m, k, new_k;
	int steps, stop, substeps;
#ifdef CHECK_SV
	int nsv, nbsv;
#endif

// #define CHECK_OBJ
	int misclassified;
	double obj;
	double violation;
	/* 0. allocate space */
	m = data->m; new_m = data->m;
	k = data->k; new_k = data->k;
	mk = data->m * data->k;
	
	double* alpha    = (double*)mu_alloc(sizeof(double), mk);
	double* grad     = (double*)mu_alloc(sizeof(double), mk);
	double* part_grad = (double*)mu_alloc(sizeof(double), mk);
	double* dgrad    = (double*)mu_alloc(sizeof(double), mk);
	
	double* ol       = (double*)mu_alloc(sizeof(double), data->m);
	
	int* ind_in      = (int*)mu_alloc(sizeof(int), param->nsp+1);
	int* ind_in_nz   = (int*)mu_alloc(sizeof(int), param->nsp+1);
	int* mind_in     = (int*)mu_alloc(sizeof(int), param->nsp+1);
	int* ind_in_old  = (int*)mu_alloc(sizeof(int), param->nsp+1);
	
	int* ws_hist     = (int*)mu_alloc(sizeof(int), data->m+1);
	int* ind_out     = (int*)mu_alloc(sizeof(int), data->m+1);
	int* ind_out_old = (int*)mu_alloc(sizeof(int), data->m+1);
	int* selected    = (int*)mu_alloc(sizeof(int), data->m+1);
	INT2DOUBLE *ind2ol  = (INT2DOUBLE*)mu_alloc(sizeof(INT2DOUBLE), data->m);
	INT2DOUBLE *ind2ol2 = (INT2DOUBLE*)mu_alloc(sizeof(INT2DOUBLE), data->m);
	int* alpha_status = (int*)mu_alloc(sizeof(int), data->m);
	int* alpha_status_old = (int*)mu_alloc(sizeof(int), param->nsp);
	double *weight   = (double*)mu_alloc(sizeof(double), data->n);
	double *diag     = (double*)mu_alloc(sizeof(double), data->m);
	
	ws_from = (int*)mu_alloc(sizeof(int), data->m);
	int init;
	double *rowK = NULL;
	double *diff = NULL;
	/*0: non-sv, 1: sv -1: bsv*/
	
	/* set subsolver */
	if( param->subsolver == 0) {
		optimize_qp = pg_optimize_qp; 
	}else if(param->subsolver == 1) {
		optimize_qp = ipm_optimize_qp;
	} else if(param->subsolver == 2){
		optimize_qp = comb_optimize_qp;
	} else {
		assert(0 && "Specified subsolver is not supported.\n");
	}
	
	QPDATA qp;
	allocate_qp(&qp, param->nsp, data->k);
	rowK = (double*)mu_alloc(sizeof(double), qp.n*qp.n);
	diff = (double*)mu_alloc(sizeof(double), qp.n*qp.k);
	nsp = param->nsp; new_nsp = nsp;
	nc  = param->nc;
	param->epsilonuse = param->epsilon * param->C;
	
	
	{
		printf("allocate space complete for #sample=%d, #dim=%d, #tag=%d, nsp=%d.\n",
				data->m, data->n, data->k, nsp);
	}
	/* 1. initialization */
	memset(alpha, 0, sizeof(double)*mk);
	memset(ws_hist, 0, sizeof(int)*(data->m+1));
	memset(selected, 0, sizeof(int)*(data->m+1));
	memset(alpha_status, 0, sizeof(int)*data->m);
	memset(grad,  0, sizeof(double)*mk);
	
	for(i=0;i<data->m;++i) {
		grad[i*data->k+data->y[i]] = -param->C;
		diag[i]=kernel_eval(kernel,i,i);
	}

	nsp = select_workingset_rand(data, param, ind_in_old, ind_in, ind_out_old, ind_out);
	resize_qp(&qp, nsp);
	{
		printf("initialization complete.\n");
	}
	clock_t t1, t2, t3, t4, t5, t6;
	stop = 0; steps = 0;
	init = 0; ii = 0;
	violation = 0.0;
// #define DUMP_DETAILS
	/*load status*/
	if(param->check_point != 0) {
		ckpf = fopen(CHKPT,"rb");
		if(ckpf) {
			read_checkpoint(ckpf, &steps, new_m, &m, new_k, &k, alpha, grad, 
					&param->epsilon0use, new_nsp, &nsp, ind_in, ind_out, 
					&old_nsp, ind_in_old, ind_out_old, ws_hist);
			fclose(ckpf);
			steps += 1;
			for(i=0;i<m;++i) {
				alpha_status[i] = get_alpha_status(data, param, alpha+i*k, i);
			}
			printf("Resume from steps %d.\n", steps);

		}
	}
	fflush(stdin);
	while(!stop) {
		{
			printf("===== Iteration %d =====\n", steps);
		}

		/* 2. construct QP problem */
		if(steps==0) {
			param->epsilon0use = 10*param->epsilon00;
		}else if(steps==1){
			param->epsilon0use = param->epsilon00;
		}
		t1 = clock();
		init = ((violation < param->init_threshold * param->epsilon * param->C)
				|| param->use_init);
		// printf("init=%d \n", init);
		construct_subqp(kernel, grad, alpha, ind_in, ind_out, init, rowK, &qp);

		/* 3. solve QP problem */
		t2 = clock();
		substeps = (*optimize_qp)(&qp, init, param);
		{
			printf("subqp scale %d, inner iteration %d.\n", qp.n, substeps);
		}
		/* 4. update gradient and alpha*/
		t3 = clock();
		update_gradient(kernel, param, alpha, weight, &qp, ind_in, ind_in_nz,
				mind_in, diff, ind_out, part_grad, dgrad, grad);
		for(i=0;((ii=ind_in[i])>=0);++i) {
			for(r=0;r<qp.k;++r) {
				alpha[ii*qp.k+r] = qp.x[i*qp.k+r];
			}
			alpha_status_old[i] = alpha_status[ii];
			alpha_status[ii]    = get_alpha_status(data, param, alpha+ii*qp.k, ii);
		}
		t4 = clock();
		
		/* update working set history */
		for(i=0;((ii=ind_in[i])>=0);++i) {
			if(alpha_status_old[i] == alpha_status[ii]) {
				ws_hist[ii] += 1;
			}else{
				ws_hist[ii] = 1;
			}
		}
		
		for(i=0;((ii=ind_out[i])>=0);++i) {
			ws_hist[ii] = 0;
		}
		
		/* count current sv */
#ifdef CHECK_SV
		nsv = 0; nbsv = 0;
		for(i=0;i<data->m;++i){
			if(alpha_status[i]!=0){
				nsv += 1;
				if(alpha_status[i]==-1){
					nbsv += 1;
				}
			}
		}
		{
			printf("nsv=%d(%d bsv)\n", nsv, nbsv);
		}
#endif
		
		/* 5. check optimality */
		if(param->ws_selector==0){
			if(optimal(grad, alpha, ol, &violation, data, param)){
				stop=1;
				break;
			}
		}else if(param->ws_selector==1){
			if(optimal_quad(grad, alpha, diag, ol, &violation, data, param)){
				stop=1;
				break;
			}
		}else if(param->ws_selector==2){
			if(optimal_quad_ss(grad, alpha, diag, ol, &violation, data, param)){
				stop=1;
				break;
			}
		}
		
		if(steps >= param->max_iteration) {
			stop = 1;
			break;
		}
		
		/* 5' calc objective */
#ifdef CHECK_OBJ
		obj = 0.0;
		for(i=0;i<m;++i){
			for(r=0;r<k;++r){
				obj += alpha[i*k+r]*(grad[i*k+r]-param->C*(r==data->y[i]));
			}
		}
		obj /= 2;
#endif
		t5 = clock();
		{
			printf("violation: %.8lf\n", violation);
#ifdef CHECK_OBJ
			printf("objective: %.8lf\n", obj);
#endif
		}
		/* 6. update working set */
		printf("update working set with nc=%d, nsp=%d\n", nc, nsp);
		nsp = update_workingset(ol, alpha_status, ws_hist, ind2ol, 
				ind2ol2, selected, ind_in_old, ind_in,
				ind_out_old, ind_out, &nc, param);  
		if(nsp != qp.n) {
			resize_qp(&qp, nsp);
		}
		/* record the status for check point */
		{
			if((param->check_point > 0)&&
					(steps && (steps%param->check_point==0))) {

				ckpf = fopen(CHKPT,"wb");
				write_checkpoint(ckpf, steps, m, k, alpha, grad, 
						param->epsilon0use, nsp, ind_in, ind_out, 
						qp.n, ind_in_old, ind_out_old, ws_hist);
				fclose(ckpf);
				printf("Save at steps = %d.\n", steps);
			}
		}

		t6 = clock();
		
		if(param->time_limit>0 && ftotal > param->time_limit) {

			ckpf = fopen(CHKPT,"wb");
			write_checkpoint(ckpf, steps, m, k, alpha, grad, 
					param->epsilon0use, nsp, ind_in, ind_out, 
					qp.n, ind_in_old, ind_out_old, ws_hist);
			fclose(ckpf);
			printf("Save at steps = %d.\n", steps);

			stop = 1;
			break;
		}
		
		steps += 1;

		/* record the time profile*/
		fcons_subqp += (double(t2-t1))/CLOCKS_PER_SEC;
		fsolv_subqp += (double(t3-t2))/CLOCKS_PER_SEC;
		fupdate_grad += (double(t4-t3))/CLOCKS_PER_SEC;
		fcheck_opt += (double(t5-t4))/CLOCKS_PER_SEC;
		fsele_workingset += (double(t6-t5))/CLOCKS_PER_SEC;
		ftotal += (double(t6-t1))/CLOCKS_PER_SEC;
	}
	
	{
		printf("complete in %d steps with violation %.6lf.\n", steps, violation);
	}
	
	obj = 0.0;
	misclassified=0;
	for(i=0;i<m;++i){
		for(r=0;r<k;++r){
			obj += alpha[i*k+r]*(grad[i*k+r]-param->C*(r==data->y[i]));
		}
		for(r=0;r<k;++r){
			if(r!=data->y[i] && (grad[i*k+r]>=(grad[i*k+data->y[i]]+param->C))) {
				break;
			}
		}
		
		if(r<k){
			misclassified+=1;
		}
	}
	obj /= 2;
	printf("objective: %.8lf, misclassified=%d\n", obj, misclassified);
	
	/* calculate the model */
#ifndef _SEQ
	if(model) {
		compute_model(data, kernel, alpha, model, param);
	}
#endif
	/* print the profile statistics */
	{
		printf("Kernel caching: \n");
		printf("  Required:    %.0lf \n", lknl_req);
		printf("  Evaluated:   %.0lf \n", lknl_eval);
		printf("  Kernel hit:  %ld \n", lknl_hit);
		printf("  Kernel miss: %ld \n", lknl_miss);

		printf("Profile total: %.4lf, in which kernel %.4lf(%.2lf%%)\n", ftotal, fknl_eval, fknl_eval/ftotal*100);
		printf("  Construct subproblem: %.4lf(%.2lf%%)\n", fcons_subqp, fcons_subqp/ftotal*100);
		printf("  Solve subproblem:     %.4lf(%.2lf%%)\n", fsolv_subqp, fsolv_subqp/ftotal*100);
		printf("  Update gradient:      %.4lf(%.2lf%%)\n", fupdate_grad, fupdate_grad/ftotal*100);
		printf("    Eval kernel:        %.4lf(%.2lf%%)\n", fupdate_grad_knl, fupdate_grad_knl/ftotal*100);
		printf("    Calc d:             %.4lf(%.2lf%%)\n", fupdate_grad_calcd, fupdate_grad_calcd/ftotal*100);
		printf("    Update:             %.4lf(%.2lf%%)\n", fupdate_grad_updat, fupdate_grad_updat/ftotal*100);
		printf("  Check optimality:     %.4lf(%.2lf%%)\n", fcheck_opt, fcheck_opt/ftotal*100);
		printf("  Select working set:   %.4lf(%.2lf%%)\n", fsele_workingset, fsele_workingset/ftotal*100);
	}
	free_qp(&qp);
	mu_free((void**)(&rowK));
	mu_free((void**)(&diff));
	mu_free((void**)(&alpha_status_old));
	mu_free((void**)(&alpha_status));
	mu_free((void**)(&ind2ol));
	mu_free((void**)(&ind2ol2));
	mu_free((void**)(&weight));
	mu_free((void**)(&alpha));
	mu_free((void**)(&diag));
	mu_free((void**)(&part_grad));
	mu_free((void**)(&grad));
	mu_free((void**)(&ol));
	mu_free((void**)(&ws_hist));
	mu_free((void**)(&mind_in));
	mu_free((void**)(&selected));
	mu_free((void**)(&ind_in_nz));
	mu_free((void**)(&ind_in));
	mu_free((void**)(&ind_out));
	mu_free((void**)(&ws_from));
}

int get_alpha_status(McSample* data, LEARN_PARAM* param, double* alpha, int i)
{
	int s = 0;
	if(alpha[data->y[i]]> param->epsilon_alga) {
		if((1-alpha[data->y[i]]) < param->epsilon_alga){
			s = -1;
			/* int r;
			for(r=0;r<data->k;++r){
				if(alpha[r] < -1+1.0e-6){
					s = -2;
					break;
				}
			} */
		}else{
			s = 1;
		}
	}
	return s;
}

void construct_subqp(KERNEL* kernel, double* grad, double* alpha, int *ind_in,
		int *ind_out, int init, double* rowK, QPDATA* qp/*OUT*/) {

	int i,j,ii,jj,r,ik,iik;
	int in;
	cache_type* kcol;
	
	/* if(pid==0&&init) {
		printf("construct sub-qp with initial point.\n");
	} */

	/* the initial point and the upper bound of constraints */
	for(i=0; ((ii=ind_in[i])>=0); ++i){
		ik  = i*qp->k;
		iik = ii*qp->k;
		for(r=0;r<qp->k;++r){
			if(!init){ /* initialize x or not */
				qp->x[ik+r] = 0.0;
			}else{
				qp->x[ik+r] = alpha[iik+r];
			}
		}
		qp->y[i] = kernel->data->y[ii];
	}
	/* the hession part */
	for(i=0;((ii=ind_in[i])>=0 && i<qp->n);++i) {
		/* construct the parallel part */
		kcol = NULL;
		if(kernel->type!=0) {
			cache_touch_data(kernel->cache, ii, &kcol);
		}
		if(kcol == NULL) {
			for(j=0;(j <= i && (jj=ind_in[j])>=0); ++j){
				rowK[i*qp->n+j] = kernel_eval(kernel, ii, jj);
			}
		}else{
			for(j=0;(j <= i && (jj=ind_in[j])>=0); ++j){
				rowK[i*qp->n+j] = kcol[jj];
			}
		}
	}
	/* fill up the triangle hole */
	for(i=0;((ii=ind_in[i])>=0 && i<qp->n);++i) {
		for(j=(i+1); ((j<qp->n) && (ind_in[j]>=0)); ++j){
			rowK[i*qp->n+j] =rowK[j*qp->n+i]; 
		}
	}
	/* transpose to column based */
	for(i=0;((ii=ind_in[i])>=0 && i < qp->n );++i) {
		for(j=0; j<qp->n; ++j) {
			qp->K[j*qp->n+i] = rowK[i*qp->n+j];
		}
	}
	/* the linear coefficient */
	for(i=0;((ii=ind_in[i])>=0 && i<qp->n);++i) {
		ik  = i*qp->k;
		iik = ii*qp->k;
		in  = i*qp->n;
		for(r=0;r<qp->k;++r){
			qp->g[ik+r]    = grad[iik+r];
			qp->grad[ik+r] = grad[iik+r];
		}
		for(j=0;((jj=ind_in[j])>=0); ++j){
			for(r=0;r<qp->k;++r){
				qp->g[ik+r] -= rowK[in+j] * alpha[jj*qp->k+r];
			}
		}
	}
}

int comb_optimize_qp(QPDATA* qp /*IN/OUT*/, int init, LEARN_PARAM* param)
{
	int alter;
	/* return ipm_optimize_qp(qp, init, param); */
	if(current_sub_solver == 0) {
		alter = pg_optimize_qp(qp, init, param);
		if(alter) {
			ipm_optimize_qp(qp, init, param);
			current_sub_solver = 1;
			steps_use_ipm = 0;
		}
	} else if(current_sub_solver == 1) {
		ipm_optimize_qp(qp, init, param);
		steps_use_ipm += 1;
		if(steps_use_ipm >= param->steps_to_alter) {
			current_sub_solver = 0;
			steps_use_ipm = 0;
		}
	} else {
		assert( 0 && "Specified subsolver is not supported.\n");
	}
	return current_sub_solver;
}
#ifdef _SEQ
int ipm_optimize_qp(QPDATA* qp /*IN/OUT*/, int init, LEARN_PARAM* param) {
	printf("Error: No IPM method implemented for sequence classification.");
	return -1;
}
#else
int ipm_optimize_qp(QPDATA* qp /*IN/OUT*/, int init, LEARN_PARAM* param)
{
	int n1, m1, m2, nnzQ, nnzA, nnzC;
	int nnz;
	int i, j, r, k, ik;
	int *krowQ, *jcolQ, *krowA, *jcolA;
	char *ixlow, *ixupp;
	double *dQ, *dA, *b, *xupp, *xlow;
	double *c;
	double epsilon_opt;
	n1 = qp->n * qp->k; m1 = qp->n; m2 = 0;
	nnzQ  = qp->n * (qp->n+1) * qp->k / 2; 
	nnzA  = qp->n * qp->k;
	nnzC  = 0;
	// printf("n=%d, k=%d\n", qp->n, qp->k);
	krowQ = (int*) mu_alloc(sizeof(int), n1+1);
	jcolQ = (int*) mu_alloc(sizeof(int), nnzQ);
	dQ    = (double*) mu_alloc(sizeof(double), nnzQ);
	
	krowA = (int*) mu_alloc(sizeof(int), m1+1);
	jcolA = (int*) mu_alloc(sizeof(int), nnzA);
	dA    = (double*) mu_alloc(sizeof(double), nnzA);
	b     = (double*) mu_alloc(sizeof(double), m1);
	
	ixlow = (char*) mu_alloc(sizeof(char), n1);
	ixupp = (char*) mu_alloc(sizeof(char), n1);
	xupp  = (double*) mu_alloc(sizeof(double), n1);
	xlow  = (double*) mu_alloc(sizeof(double), n1);
	c     = (double*) mu_alloc(sizeof(double), n1);
	// construct the sparse Q
	nnz = 0;
	r   = 0;
	for(k=0; k<qp->k; ++k) {
		for(i=0; i<qp->n; ++i) {
			r = k*qp->n+i; 
			// krowQ[r] = qp->n * r;
			krowQ[r] = k*qp->n*(qp->n+1)/2 + i*(i+1)/2;
			for(j=0; j<=i; ++j) {
				jcolQ[nnz] = k*qp->n+j;
				dQ[nnz]    = qp->K[i*qp->n+j];
				nnz += 1;
			}
		}
	} 
	krowQ[n1] = nnzQ;
	
	nnz = 0;
	for(i=0; i<qp->n; ++i) {
		krowA[i] = i*qp->k;
		for(j=0; j<qp->k; ++j) {
			jcolA[nnz] = j*qp->n+i;
			dA[nnz]    = 1.0;
			nnz += 1;
		}
	}
	krowA[i] = nnzA;
	
	for(i=0; i<m1; ++i) {
		b[i] = 0.0;
	}
	
	for(i=0; i<n1; ++i) {
		ixlow[i] = 0;
		xlow[i]  = 0.0;
		ixupp[i] = 1;
		xupp[i]  = ((i/qp->n)==qp->y[i%qp->n])?1:0;
		
		// assert(((i/qp->k)+(i%qp->k)*qp->n) < n1);
		c[(i/qp->k)+(i%qp->k)*qp->n] = qp->g[i];
	} 
	QpGenSparseMa27 * qps 
	      = new QpGenSparseMa27( n1, m1, m2, nnzQ, nnzA, nnzC );

	QpGenData * prob = (QpGenData * ) qps->makeData(
				c,
				krowQ, jcolQ, dQ,
				xlow, ixlow,
				xupp, ixupp,
				krowA, jcolA, dA,
				b,
				NULL, NULL, NULL,
				NULL, NULL,
				NULL, NULL
			);
	// dump(qp);
	// prob->print();
	QpGenVars     * vars  = (QpGenVars * ) qps->makeVariables( prob );
	Residuals     * resid = qps->makeResiduals( prob );
	GondzioSolver * s     = new GondzioSolver( qps, prob );
	
	// s->setMuTol(s->getMuTol() * param->epsilon0use/param->epsilonuse);
	// s->setArTol(s->getArTol() * param->epsilon0use/param->epsilonuse * param->C);
	printf("subsolver tolerance: %.20lf\n", param->epsilon0use);
	s->setMuTol(param->epsilon0use * 1.0e-9);
	s->setArTol(param->epsilon0use * 1.0e-9 * param->C);
	
	// s->monitorSelf();
	epsilon_opt = param->epsilon * 0.001;
	// epsilon_opt = 0.0;
	// epsilon_opt = s->getArTol() * param->epsilon0use/param->epsilonuse * param->C;
	
	int result = s->solve(prob, vars, resid);
	vars->x->copyIntoArray(c);
	for(k=0;k<n1;++k) {
		// assert((i/qp->n+(i%qp->n)*qp->k) < n1);
		i = k%qp->n;
		r = k/qp->n;
		qp->x[r+i*qp->k] = c[k];
		/*
		if(fabs(c[k])<epsilon_opt) {
			qp->x[r+i*qp->k] = 0.0;
		}else if(r==qp->y[i] && qp->x[r+i*qp->k] > 1.0-epsilon_opt) {
			qp->x[r+i*qp->k] = 1.0;
		} else {
			
		} */
	}
	/* do rounding here */
	for(i=0;i<qp->n;++i) {
		ik = i*qp->k;
		if(qp->x[qp->y[i]+ik] < epsilon_opt) {
			for(r=0;r<qp->k;++r) {
				qp->x[r+ik] = 0.0;
			}
		} else if(qp->x[qp->y[i]+ik] > 1-epsilon_opt) {
			for(r=0;r<qp->k;++r) {
				if(qp->x[r+ik] < -1+epsilon_opt) {
					break;
				}
			}
			if(r < qp->k) { /* contains a +1 -1 pair*/
				for(j=0;j<qp->k;++j) {
					if(j==r) {
						qp->x[j+ik] = -1;
					} else if(j==qp->y[i]) {
						qp->x[j+ik] = 1;
					} else {
						qp->x[j+ik] = 0.0;
					}
				}
			}
		}
	}

	mu_free((void**)(&krowQ));
	mu_free((void**)(&jcolQ));
	mu_free((void**)(&dQ)); 
	mu_free((void**)(&krowA));
	mu_free((void**)(&jcolA));
	mu_free((void**)(&dA));
	mu_free((void**)(&ixlow));
	mu_free((void**)(&ixupp)); 
	mu_free((void**)(&xupp));
	mu_free((void**)(&xlow));
	mu_free((void**)(&c)); 
	
	delete qps;
	delete s;
	
	return (result);
}
#endif
int pg_optimize_qp(QPDATA* qp /*IN/OUT*/, int init, LEARN_PARAM* param) {
	SPG_PARAM p;
	int steps;

	set_default_param(&p);
	p.verbose = 0;
	p.epsilon = param->epsilon0use * param->C ;
	p.init = init;
	p.proj = param->projector;
	/* p.fp   = fp; */
	p.epsilon_alga = param->epsilon_alga;
	
	if( qp->m <= 20 ) {
		p.max_iteration = 30*qp->m;
	} else if( qp->m <= 200 ) {
		p.max_iteration = 20*qp->m;
	} else {
		p.max_iteration = 10*qp->m;
	}
	
	// dump(qp);
	
	steps = spg_solve(qp, &p);
	return (steps >= p.max_iteration);
}


int update_gradient(KERNEL* kernel, LEARN_PARAM* param, double* alpha, double* weight, 
		QPDATA* qp,
		int *ind_in, int* ind_in_nz, int* mind_in, double* diff,
		int *ind_out, double* part_grad, double* dgrad, double* grad/*OUT*/) {

	int i, ik, j, jk, jj, jjk, r, nnz, mk, m, n;
	cache_type *kcol;
	clock_t t1, t2, t3;
// #define PASS_GRAD	
	n = kernel->data->n;
	m = kernel->data->m;
#ifndef PASS_GRAD 
	mk = kernel->data->m * kernel->data->k;
#else
	int ii, iik, ik;
	mk= (m-qp->n)*(kernel->data->k);
#endif
	
	/* count the nonzeros */
	nnz = 0;
	for(j=0;((jj=ind_in[j])>=0);++j) {
		for(r=0; r<qp->k;++r){
			if(fabs(qp->x[j*qp->k+r]-alpha[jj*qp->k+r]) > param->epsilon_alga){
				break;
			}
		}
		if(r < qp->k) {
			for(r=0;r<qp->k;++r){
				diff[nnz*qp->k+r] = (qp->x[j*qp->k+r]-alpha[jj*qp->k+r]);
			}
			ind_in_nz[nnz] = jj;
			nnz += 1;
		}
	}
	ind_in_nz[nnz] = -1;
	// printf("nnz=%d\n",nnz);
	// fflush(stdout);

	if( kernel->type != 0) {
	    for(i=0;i<nnz;++i) {
	    	mind_in[i] = i;
	    }
	    mind_in[i] = -1;
		memset(dgrad, 0, sizeof(double)*mk);
		for(j=0;((jj=mind_in[j])>=0);++j) {
			
			t1 = clock();
			jk  = j*qp->k;
			jjk = jj*qp->k;
			kcol = kernel_get_col(kernel, ind_in_nz[jj]);

			t2 = clock();

			for(i=0;i<m;++i) {
#ifdef PASS_GRAD
				/* for(i=0;((ii=ind_in[i])>=0);++i) {
					part_grad[ii*qp->k+r] += kcol[ii]*diff[jjk+r];
				} */
				for(i=0;((ii=ind_out[i])>=0);++i) {
					dgrad[i*qp->k+r] += kcol[ii]*diff[jjk+r];
				}
#else
				ik = i*qp->k;
				for(r=0;r<qp->k;++r) {
					dgrad[ik+r] += kcol[i]*diff[jjk+r];
				}
#endif
			}
			t3 = clock();
			fupdate_grad_knl   += ((double)(t2-t1))/CLOCKS_PER_SEC;
			fupdate_grad_calcd += ((double)(t3-t2))/CLOCKS_PER_SEC;
		}
	} else {
		/* special case for linear kernel */
		t2 = clock();
		for(r=0;r<qp->k;++r) {
			for(i=0;i<n;++i){ weight[i] = 0.0; }
			for(j=0; j<nnz; ++j) {
				kernel_add(kernel, weight, ind_in_nz[j], diff[j*qp->k+r]);
			}
#ifdef PASS_GRAD			
			/* for(i=0;((ii=ind_in[i])>=0);++i) {
				part_grad[ii*qp->k+r] = kernel_product(kernel, weight, ii);
			} */
			for(i=0;((ii=ind_out[i])>=0);++i) {
				dgrad[i*qp->k+r] = kernel_product(kernel, weight, ii);
			}
#else
			for(i=0;i<m;++i){
				dgrad[i*qp->k+r] = kernel_product(kernel, weight, i);
			}
#endif
		}
		t3 = clock();
		fupdate_grad_calcd += ((double)(t3-t2))/CLOCKS_PER_SEC;
	}
#ifndef PASS_GRAD
	t1 = clock();
	for(i=0;i<mk;++i) {
		grad[i] += dgrad[i];
	}
	t2 = clock();
	fupdate_grad_updat += ((double)(t2-t1))/CLOCKS_PER_SEC;
#else
	// double norm=0.0;
	for(i=0;((ii=ind_in[i])>=0);++i) {
		iik = ii*qp->k;
		ik = i*qp->k;
		for(r=0;r<qp->k;++r) {
			// norm += fabs(grad[ii*qp->k+r]+dgrad[ii*qp->k+r] - qp->grad[i*qp->k+r]);
			grad[iik+r] = qp->grad[ik+r];
			// grad[ii*qp->k+r] += dgrad[ii*qp->k+r];
		}
	}
	for(i=0;((ii=ind_out[i])>=0);++i) {
		iik = ii*qp->k;
		ik = i*qp->k;
		for(r=0;r<qp->k;++r) {
			grad[iik+r] += dgrad[ik+r];
		}
	}
	// printf("difference norm of grad: %.16lf.\n", norm);
#endif

	return (0);
}

int optimal_quad(double* grad, double* alpha, double* diag, double* ol /*OUT*/, double* violation /*OUT*/,
		McSample* data, LEARN_PARAM* param)
{
	/* check the optimality */
	int i, r, ik ,p;
	double maxf, a;
	(*violation) = INF;

	for(i=0;i<data->m;++i){
		ik   = i*data->k;
		maxf = grad[ik];
		p=0;
		for(r=1;r<data->k;++r) {
			if(grad[ik+r] > maxf) {
				maxf = grad[ik+r];
				p=r;
			}
		}
		ol[i] = 0.0;
		a=0.0;
		for(r=0;r<data->k;++r){
			if(grad[ik+r] < maxf){
				ol[i] += (grad[ik+r] - maxf)*((r==data->y[i])-alpha[ik+r]);
			}
			a+=alpha[ik+r]*alpha[ik+r];
		}
		
		/*if(ol[i] < (*violation)){
			(*violation) = ol[i];
		}*/
		
		// printf("Lin l(%d)=%.6lf.\n",i,ol[i]);
		a+=2*(1-(data->y[i]==p)-alpha[ik+data->y[i]]+alpha[ik+p]);
		a*=diag[i];
		// printf("a=%.6lf.\n",a);
		if(a+ol[i]<=0) {
			ol[i]+=0.5*a;
		} else {
			ol[i]=-ol[i]*ol[i]/(2*a);
		}
		// printf("Quad l(%d)=%.6lf.\n",i,ol[i]);
		assert(ol[i] <= param->epsilon_alga);
		if(ol[i] < (*violation)){
			(*violation) = ol[i];
		}
	}

	(*violation) *= -1;
	if(*violation < param->epsilonuse){
		return 1;
	}
	return 0;	
}

int optimal_quad_ss(double* grad, double* alpha, double* diag, double* ol /*OUT*/, double* violation /*OUT*/,
		McSample* data, LEARN_PARAM* param)
{
	/* check the optimality */
	int i, r, ik,rr;
	double b, a, d, hd, *coef, v;
	(*violation) = INF;
	coef=(double*)mu_alloc(sizeof(double), data->k);
	
	for(i=0;i<data->m;++i){
		ik   = i*data->k;
		b=0.0;
		for(r=0;r<data->k;++r) {
			b+=grad[ik+r];
			coef[r]=diag[i]*((r==data->y[i])-alpha[ik+r])+grad[ik+r];
		}
		// calculate v - Lagrangian of equation constraints
		std::sort(coef, coef+data->k);
		
		/* printf("b=%.6lf, ", b);
		for(r=0;r<data->k;++r) {
			printf("coef[%d]=%.6lf,", r, coef[r]);
		}
		printf("\n"); */
		
		a=0.0;
		rr=data->k;
		
		if(b/rr<=coef[0]){
			v=b/rr;
		}else{
			for(r=0;r<data->k;++r) {
				a+=coef[r];
				rr-=1;
				if(rr==0) break;
				v=(b-a)/rr;
				// printf("find v=%.6lf at %d.\n",v,r);
				if(v>coef[r] && v<=coef[r+1]){
					// printf("find v=%.6lf at %d.\n",v,r);
					break;
				}
			}
		}
		assert(rr>0 && "Fail to get find v.");
		// calculate the optimal objective value
		ol[i]=0.0;
		hd=0.5*diag[i];
		for(r=0;r<data->k;++r){
			d=min((v-grad[ik+r])/diag[i], (data->y[i]==r)-alpha[ik+r]);
			ol[i]+=d*(grad[ik+r]+hd*d);
		}
		// printf("Quad l(%d)=%.6lf.\n",i,ol[i]);
		assert(ol[i] <= param->epsilon_alga);
		if(ol[i] < (*violation)){
			(*violation) = ol[i];
		}
	}

	(*violation) *= -1;
	if(*violation < param->epsilonuse){
		return 1;
	}
	
	mu_free((void**)(&coef));
	return 0;	
}

int optimal(double* grad, double* alpha, double* ol /*OUT*/, double* violation /*OUT*/,
		McSample* data, LEARN_PARAM* param) {
	/* check the optimality */
	int i, r, ik ;
	double maxf;
	(*violation) = INF;

	for(i=0;i<data->m;++i){
		ik   = i*data->k;
		maxf = grad[ik];
		for(r=1;r<data->k;++r){
			if(grad[ik+r] > maxf){
				maxf = grad[ik+r];
			}
		}
		ol[i] = 0.0;
		for(r=0;r<data->k;++r){
			if(grad[ik+r] < maxf){
				ol[i] += (grad[ik+r] - maxf)*((r==data->y[i])-alpha[ik+r]);
			}
		}
		if(ol[i] < (*violation)){
			(*violation) = ol[i];
		}
		assert(ol[i] <= param->epsilon_alga);
	}

	(*violation) *= -1;
	if(*violation < param->epsilonuse){
		return 1;
	}
	return 0;
}

int update_workingset(double* ol, int* alpha_status, int* ws_hist,
		INT2DOUBLE* ind2col_in, INT2DOUBLE* ind2col,
		int* selected, int* ind_in_old, /*IN/OUT*/ int *ind_in /*OUT*/,
		int* ind_out_old,/*IN/OUT*/ int *ind_out /*OUT*/,
		int* newnc, LEARN_PARAM* param) {

	int i, ii, j;
	int ini, outi, nin, nout;
	int outin, nnew;
	int nremained;
	int ri, rn;
	int roi, ron;
	
	/* select nc new from index out */
	nout = 0;
	for(i=0;((ii=ind_out_old[i])>=0);++i){
		ind2col[nout].val = ol[ii];
		ind2col[nout].id = ii;

		nout += 1;
	}
	for(i=0;((ii=ind_in_old[i])>=0);++i){
		ind2col[nout].val = ol[ii];
		ind2col[nout].id = ii;
		
		selected[ii] = 0;
		nout += 1;
	}
	
	if(nout>0) {
		std::partial_sort(ind2col, ind2col+(*newnc), ind2col+nout, INT2DOUBLE());
	}

	/* fill up the working set with previous */
	nin = 0;
	for(i=0;((ii=ind_in_old[i])>=0);++i){
		ind2col_in[nin].val = ol[ii];
		ind2col_in[nin].id = ii;
		nin += 1;
	}

	ini = 0; outi = 0; outin = 0;
	i = 0; j = 0;
	roi=0; ron=nout;
	while(ini < (*newnc)) {
		if(-ind2col[j].val <= param->epsilonuse) {
			break;
		}
		if(j >= nout) {
			break;
		}
		{
			ind_in[ini] = ind2col[j].id;

			if(!ws_hist[ind_in[ini]]) {
				outin += 1;
			}
			selected[ind_in[ini]]=1;
			
			ini += 1;
			j += 1;
		}
	}
	roi = j;
	/* (*newnc) = ini; */

	if(outin > 0) {
		rn=0; ri = 0;
		/* collect the sample to prev part */
		for(i=0; i<nin; ++i) {
			if(!selected[ind2col_in[i].id]) {
				if(rn<i) {
					ind2col_in[rn].id=ind2col_in[i].id;
					ind2col_in[rn].val=ind2col_in[i].val;
				}
				rn += 1;
			}
		}
		i=0;
// #define SELECT_PREV
#ifdef SELECT_PREV
		std::sort(ind2col_in, ind2col_in+rn, INT2DOUBLE());
		for(i=0;i<rn;++i) {
			if(-ind2col_in[i].val < param->ratio* param->epsilonuse /* param->epsilon0use*/ 
					|| ini >= param->nsp) {
				break;
			}
			ind_in[ini] = ind2col_in[i].id;

			selected[ind_in[ini]]= 1;
			ini += 1;
		}
#endif
		ri = i;

		/* sort according to the history in working set */
		for(i=ri; i<rn; ++i) {
			ind2col_in[i].val = ws_hist[ind2col_in[i].id];
		}
		sort(ind2col_in+ri, ind2col_in+rn, INT2DOUBLE());
	
		if(ini < param->nsp) {
			nremained = 0;
			for(i=ri; i<rn;++i) {
				if(ini < param->nsp && alpha_status[ind2col_in[i].id] == 1){ /* sv in previous working set */
					ind_in[ini] = ind2col_in[i].id;

					selected[ind_in[ini]]=1;
					ini += 1;
				} else {
					ind2col_in[nremained].id = ind2col_in[i].id;
					ind2col_in[nremained].val = ind2col_in[i].val;
					nremained += 1;
				}
			}

			ri = 0; rn = nremained;
			if(ini < param->nsp) { /* add the previous free sample if not enough */
				nremained = 0;

				for(i=ri; i<rn; ++i) { /* non-sv in previous working set */
					if(ini<param->nsp && alpha_status[ind2col_in[i].id] == 0) {
						/* printf("status(%d,%d) ", ind2col[i].id, alpha_status[ind2col[i].id]); */
						ind_in[ini] = ind2col_in[i].id;

						selected[ind_in[ini]]=1;
						ini += 1;
					} else {
						ind2col_in[nremained].id = ind2col_in[i].id;
						ind2col_in[nremained].val = ind2col_in[i].val;
						nremained += 1;
					}
				}

				ri = 0; rn = nremained;
				if(ini < param->nsp) {
					for(i=ri; ini<param->nsp && i<rn; ++i){
						ind_in[ini] = ind2col_in[i].id;

						selected[ind_in[ini]]=1;
						ini += 1;
					}

					ri = i; rn = nremained;
				}
			}
		}

		outi = 0;
		for(j=roi;j<ron;++j) {
			if(!selected[ind2col[j].id]) {
				ind_out[outi] = ind2col[j].id;
				outi += 1;
			}
		}

		ind_in[ini]   = -1;
		ind_out[outi] = -1;

		std::sort(ind_in, ind_in+ini);
		/* copy ind_in and ind_out to old */
		memcpy(ind_in_old, ind_in, sizeof(int)*(ini+1));
		memcpy(ind_out_old, ind_out, sizeof(int)*(outi+1));
	} else {
		ini = nin;
		/* copy the old ind_in and ind_out directly */
		memcpy(ind_in, ind_in_old, sizeof(int)*(nin+1));
		memcpy(ind_out, ind_out_old, sizeof(int)*(nout-nin+1));
	}
	
	assert((ini>=nin) && "working set can not be decreased.\n");
	/* adjust the tolerance of inner solver */
	if(outin==0) {
		param->epsilon0use *= 0.1;
		param->use_init = 1;
		if(param->epsilon0use < 1.0e-6*param->epsilon0) {
			printf("Fail to improve the inner QP solution, please change the kernel or parameter.\n");
			exit(-1);
		}
	} else {
		if(param->epsilon0use < param->epsilon0) {
			param->epsilon0use = param->epsilon0;
		}
	}
	
	/* heuristic to adjust the nc */
	nnew = outin;
	nnew = nnew & (~1);
	if (nnew < 10)
		nnew = 10;
	if (nnew < param->nsp/10)
		nnew = param->nsp/10;

	if (nnew < param->nc) {
		printf("Set nc from %d to %d. \n", param->nc, nnew);
		param->nc = (nnew & (~1));
	}
	
	return ini;
}

int select_workingset_rand(McSample* data, LEARN_PARAM* param,
		int* ind_in_old, int* ind_in/*OUT*/,
		int* ind_out_old, int* ind_out /*OUT*/)
{
	int i,mc ;
	int ii, oi;
	int *accum = (int*)mu_alloc(sizeof(int),data->k);
	int *nspc  = (int*)mu_alloc(sizeof(int),data->k);
	
	for(i=0;i<data->k;++i){
		nspc[i] = param->nsp/data->k;
	}

	mc = param->nsp % data->k;
	for(i=0;i<mc;++i){
		nspc[i] += 1;
	}

	memset(accum, 0, sizeof(int)*data->k);
	ii = oi = 0;
	for(i=0;i<data->m;++i){
		if(accum[data->y[i]] < nspc[data->y[i]]){
			ind_in[ii] = i;
			ii += 1;
			accum[data->y[i]] += 1;
		}else{
			ind_out[oi] = i;
			oi += 1;
		}
	}
	ind_in[ii]  = -1;
	ind_out[oi] = -1;

	/* copy ind_in and ind_out to old */
	memcpy(ind_in_old, ind_in, sizeof(int)*(ii+1));
	memcpy(ind_out_old, ind_out, sizeof(int)*(oi+1));

	mu_free((void**)(&accum));
	mu_free((void**)(&nspc));
	{
		printf("init with nin = %d, nout = %d. \n", ii, oi);
	}
	return ii;
}

