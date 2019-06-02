/*
 * mcsvm_train.h
 *
 *  Created on: Sep 23, 2008
 *      Author: noah
 */

#ifndef MCSVM_TRAIN_H_
#define MCSVM_TRAIN_H_
#include "pgqp.h"
#include "mc_common.h"

typedef int (*sub_qp_solver)(QPDATA* qp /*IN/OUT*/, int init, LEARN_PARAM* param);
/* cache */
int cache_init(CACHE* cache, int msize, int m);
NODE* cache_lru_delete(CACHE* cache);
NODE* cache_lru_insert(CACHE* cache, NODE* node);
int cache_get_data(CACHE* cache, int col, cache_type** data);
int cache_touch_data(CACHE* cache, int col, cache_type** data);
int cache_uninit(CACHE* cache);

/* kernel */
int kernel_init(KERNEL* kernel, McSample* data, CACHE* cache);
cache_type* kernel_get_col(KERNEL* kernel, int col);
double kernel_eval(KERNEL* kernel, int i, int j);
void kernel_add(KERNEL* kernel, double* weight, int i, double multi);
double kernel_product(KERNEL* kernel, double* weight, int j);

/* svm solver */
void svm_learn(McSample* data, KERNEL* kernel, LEARN_PARAM* param,
		MODEL* model, double* alpha);
void construct_subqp(KERNEL* kernel, double* grad, double* alpha, int *ind_in,
		int *ind_out, int init, double* rowK, QPDATA* qp/*OUT*/);
int comb_optimize_qp(QPDATA* qp /*IN/OUT*/, int init, LEARN_PARAM* param);
int pg_optimize_qp(QPDATA* qp /*IN/OUT*/, int init, LEARN_PARAM* param);
int ipm_optimize_qp(QPDATA* qp /*IN/OUT*/, int init, LEARN_PARAM* param);
int update_gradient(KERNEL* kernel, LEARN_PARAM* param, double* alpha, double* weight, 
		QPDATA* qp, int *ind_in,int *ind_in_nz, int *mind_in, double* diff,
		int *ind_out, double* part_grad, double* dgrad, double* grad/*OUT*/);

int optimal_quad(double* grad, double* alpha, double* diag, double* ol /*OUT*/, double* violation /*OUT*/,
		McSample* data, LEARN_PARAM* param);
int optimal_quad_ss(double* grad, double* alpha, double* diag, double* ol /*OUT*/, double* violation /*OUT*/,
		McSample* data, LEARN_PARAM* param);
int optimal(double* grad, double* alpha, double* ol /*OUT*/, double* violation /*OUT*/,
		McSample* data, LEARN_PARAM* param);

int update_workingset(double* ol, int* alpha_status, int* ws_hist,
		INT2DOUBLE* ind2col_in, INT2DOUBLE* ind2col,
		int* selected, int* ind_in_old, /*IN/OUT*/ int *ind_in /*OUT*/,
		int* ind_out_old,/*IN/OUT*/ int *ind_out /*OUT*/,
		int* newnc, LEARN_PARAM* param);
int select_workingset_rand(McSample* data, LEARN_PARAM* param,
		int *ind_in_old, int* ind_in/*OUT*/,
		int *ind_out_old, int* ind_out /*OUT*/);
int get_alpha_status(McSample* data, LEARN_PARAM* param, double* alpha, int i);
int write_checkpoint(FILE* fp, int m, int k, double* alpha, double* grad, 
		int q, int* ind_in, int* ind_out, int old_q, 
		int* ind_in_old, int* ind_out_old, int* ws_hist);
int read_checkpoint(FILE* fp, int m_new, int* m, int k_new, int*k, double* alpha, double* grad,
		int q_new, int* q, int* ind_in, int* ind_out, int* old_q, 
		int* ind_in_old, int* ind_out_old, int* ws_hist);
#endif /* MCSVM_TRAIN_H_ */
