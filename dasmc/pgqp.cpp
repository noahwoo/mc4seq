/*
 * pgqp.cpp
 *
 *  Created on: Sep 23, 2008
 *      Author: noah
 */
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <algorithm>
#include "mem_util.h"
#include "pgqp.h"

// #define PROFILE_IT
#ifdef PROFILE_IT
/* profile */
static double fproject = 0;
static double fmatrixbyvec = 0;
static double fupdate_grad = 0;
static double fbacktracking = 0;
static double fupdate_steplen = 0;
static double ftotal = 0;
#endif

// #define CHECK_SPARSE
#ifdef CHECK_SPARSE
static double fsparsity = 0;
static double fsparsity_block = 0;
#endif

#define SWAP(a,b) { register double t=(a);(a)=(b);(b)=t; }

unsigned int Randnext = 1;
#define ThRand    (Randnext = Randnext * 1103515245L + 12345L)
#define ThRandPos ((Randnext = Randnext * 1103515245L + 12345L) & 0x7fffffff)

static int (*proj_func)(QPDATA* qp, int ind, double* tx /*IN/OUT*/);

/* dense dot calculation*/
double ddot(double* a, double* b, int n)
{
	int i = 0;
	double d = 0.0;
	for(i = 0; i < n; ++i) {
		d += a[i] * b[i];
	}
	return d;
}

/* infinity norm of given vector */
double inf_norm(double* a, int n)
{
	double inf = fabs(a[0]);
	int i = 1;
	for(; i < n; ++i) {
		if(fabs(a[i]) > inf) { /*if multi maximum, select the first one*/
			inf = fabs(a[i]);
		}
	}
	return inf;
}

/* square of two norm */
double square_norm(double* a, int n)
{
	double sum = 0.0;
	int i = 0;
	for(; i < n; ++i) {
		sum += a[i] * a[i];
	}
	return sum;
}

int less_than_int2double(const void * x, const void* y)
{
	const INT2DOUBLE* px = (const INT2DOUBLE*)x;
	const INT2DOUBLE* py = (const INT2DOUBLE*)y;

	if( px->val == py->val ) return 0;

	if(px->val > py->val) {
		return 1;
	} else {
		return -1;
	}
}

int less_than_double(const void* x, const void* y)
{
	const double *px = (const double*)x;
	const double *py = (const double*)y;

	if( *px == *py ) return 0;
	if( *px > *py ) {
		return 1;
	} else {
		return -1;
	}
}

void set_default_param(SPG_PARAM* param)
{
	/* set the default parameters */

	param->alpha_min = 1.0e-10;
	param->alpha_max = 1.0e+10;

	param->sigma1 = 0.1;
	param->sigma2 = 0.9;

	param->M = 2;

	param->epsilon = 1.0e-8;
	param->epsilon_kkt = 1.0e-3;
	param->epsilon_2norm = 1.0e-6;
	param->epsilon_alga = 1.0e-10;
	param->max_iteration = 100000;
	param->verbose = 1;
	param->init = 0;
	param->proj = 0;
}

void allocate_qp(QPDATA* qp, int n, int k)
{
	qp->n  = n;
	qp->k  = k;
	qp->m  = n*k;
	
	qp->K  = (double*)mu_alloc(sizeof(double), n*qp->n);
	qp->y  = (int*)mu_alloc(sizeof(int), n);

	qp->g  = (double*)mu_alloc(sizeof(double), qp->m);
	qp->grad  = (double*)mu_alloc(sizeof(double), qp->m);
	qp->x  = (double*)mu_alloc(sizeof(double), qp->m);
}

void resize_qp(QPDATA* qp, int n)
{
	qp->n  = n;
	qp->m  = n*qp->k;
}

void free_qp(QPDATA* qp)
{
	mu_free((void**)&qp->K);
	mu_free((void**)&qp->y);

	mu_free((void**)&qp->g);
	mu_free((void**)&qp->grad);
	mu_free((void**)&qp->x);
}

int feasible(QPDATA* qp, double epsilon)
{
	int feasibility = 1, i, r;
	double sum = 0.0;

	for(i = 0; i < qp->n; ++i) {
		sum = 0.0;
		for(r = 0; r < qp->k; ++r) {
			if(qp->x[i*qp->k+r] > (qp->y[i] == r)) {
				feasibility = 0;
				break;
			}
			sum += qp->x[i*qp->k+r];
		}
		if( fabs(sum) > epsilon ) {
			feasibility = 0;
			break;
		}
	}
	return feasibility;
}

void dump(QPDATA* qp)
{
	printf("K=\n");
	int i,j,r;
	for(i=0;i<qp->n;++i){
		for(j=0;j<qp->n;++j){
			printf("%.4lf\t", qp->K[i*qp->n+j]);
		}
		printf("\n");
	}
	printf("y=");
	for(i=0;i<qp->n;++i){
		printf(" %d", qp->y[i]);
	}
	printf("\n");
	printf("g=\n");
	for(i=0;i<qp->n;++i){
		printf("%d:",i);
		for(r=0;r<qp->k;++r){
			printf(" %.4lf", qp->g[i*qp->k+r]);
		}
		printf("\n");
	}
}

void bproject(QPDATA* qp, double lambda, double *tx, int ind, double *lmdx /*OUT*/ )
{
	int r;
	for(r=0;r<qp->k;++r) {
		lmdx[r] = (tx[r] + lambda * 1.0 )/1.0;
		if( lmdx[r] > (r==qp->y[ind])) {
			lmdx[r] = (r==qp->y[ind]);
		}
	}
}

double residual(QPDATA* qp, double* lmdx)
{
	int r;
	double resd = 0.0;
	for(r=0;r<qp->k;++r) {
		resd += lmdx[r];
	}
	return resd;
}

int project(QPDATA* qp, double* g, double alpha, double* tx /*OUT*/ )
{
	int i, r, ik;
	int steps = 0;
	for(i=0;i<qp->n;++i) {
		ik = i*qp->k;
		for(r=0;r<qp->k;++r) {
			tx[ik+r] = qp->x[ik+r] - alpha * g[ik+r];
		}
		steps += (*proj_func)(qp, i, tx+ik /*IN/OUT*/);
	}
	return steps;
}
/* three different projection methods implemented here */
/* Dai-Flether Method, Pardalos Method, Sort-Based method */
int Sort_project(QPDATA* qp, int ind, double* tx /*IN/OUT*/) 
{

	int i, s, n;
	double sum;

	double lambda;
	
	int label = qp->y[ind];
	assert( qp->k >= 2 );
	n = qp->k;
	
	double* lmdx = (double*) mu_alloc(sizeof(double), n);

	sum = 0.0;
	for(i = 0; i < n; ++i) {
		lmdx[i] = tx[i];
		sum += lmdx[i];
	}
	lmdx[label] -= 1.0;
	sum -= 1.0;

	std::sort(lmdx, lmdx+n);

	if( (1.0 + sum)/n >= lmdx[n-1]) { // check if tx[i] <= -lambda for all i=1,...,n
		lambda = -(sum + 1.0)/n;
	} else { // search for the s
		double tmp;
		for(i = n-1; i > 0; --i) {
			s = i;
			sum -= lmdx[i];
			tmp = (sum+1.0)/s;
			if( lmdx[i-1] <= tmp ) {
				lambda = -tmp;
				break;
			}
		}
		if(i==0) {
			printf("y[%d]=%d\n", ind, label);
			printf("hat \\alpha =\n"); 
			for(s = 0; s < n; ++s) {
				printf(" %lf", tx[i]);
			}
			printf("\n");
		}
		assert( i > 0 );
	}

	/* */
	for(i = 0; i < n; ++i) {
		tx[i] = tx[i] + lambda;
		if(tx[i] > (i==label)) {
			tx[i] = (i==label);
		}
	}
	mu_free((void**)(&lmdx));
	return (0);
}

double quick_select(double *arr, int n)
{
  int low, high ;
  int median;
  int middle, l, h;

  low    = 0; 
  high   = n-1;
  median = (low + high) / 2;
  
  for (;;)
  {
    if (high <= low)
        return arr[median];

    if (high == low + 1)
    {
        if (arr[low] > arr[high])
            SWAP(arr[low], arr[high]);
        return arr[median];
    }

    middle = (low + high) / 2;
    if (arr[middle] > arr[high]) SWAP(arr[middle], arr[high]);
    if (arr[low]    > arr[high]) SWAP(arr[low],    arr[high]);
    if (arr[middle] > arr[low])  SWAP(arr[middle], arr[low]);

    SWAP(arr[middle], arr[low+1]);

    l = low + 1;
    h = high;
    for (;;)
    {
      do l++; while (arr[low] > arr[l]);
      do h--; while (arr[h]   > arr[low]);
      if (h < l)
          break;
      SWAP(arr[l], arr[h]);
    }

    SWAP(arr[low], arr[h]);
    if (h <= median)
        low = l;
    if (h >= median)
        high = h - 1;
  }
}

int Pardalos_project(QPDATA* qp, int ind, double* tx /*IN/OUT*/)
{
	/* implement the Pardalos projection method */
	int r, n, nitv, nitv2;
	int nusv, nusv2;
	
	double d;
	n = qp->k;
	
	double* membuff  = (double*) mu_alloc(sizeof(double), 3*n+2);
	double* itvpts   = membuff;
	double* usV = membuff + (n+2);
	double* b   = membuff + 2*(n+1);
	
	double mmin, mmax, tightsum, slackweight, slackweight2, testsum;
	double median, median_old, y;
	d = 0.0;

	itvpts[0] = -INF;
	for(r=0;r<n;++r){
		itvpts[r+1] = (qp->y[ind]==r)-tx[r];
		usV[r] = itvpts[r+1];
		b[r]   = usV[r];
		d     -= tx[r];
	}
	itvpts[n+1] = INF;
	
	tightsum = 0.0; slackweight = 0.0;
	mmin = -INF; mmax = INF;
	median = mmin;
	
	nusv = n;
	nitv = n+2;

	/* begin iteration */
	while(nusv > 0) {
		
		/* get median in random */
		median_old = median;
		median = quick_select(itvpts, nitv);
		if(median == median_old) {
			median = itvpts[(int)ThRandPos%nitv];
		}
		
		testsum = tightsum;
		slackweight2 = slackweight;
		for(r=0;r<nusv;++r){
			if(usV[r]<median){
				testsum += usV[r];
			}else{
				slackweight2 += 1.0;
			}
		}
		testsum += slackweight2*median;
		/* update */
		if(testsum <= d) {
			mmin = median;
		}else{
			mmax = median;
		}

		nitv2 = 0;
		for(r=0;r<nitv;++r){
			if(itvpts[r]<=mmax && itvpts[r]>=mmin){
				itvpts[nitv2] = itvpts[r];
				nitv2+=1;
			}
		}
		nitv = nitv2;
		
		nusv2 = 0;
		for(r=0;r<nusv;++r){
			if(usV[r] <= mmin) {
				tightsum += usV[r];
			} else if(usV[r] >= mmax) {
				slackweight += 1.0;
			} else {
				usV[nusv2] = usV[r];
				nusv2 += 1;
			}
		}
		nusv = nusv2;
	}

	/* calculate the projection point */
	testsum = 0.0;
	for(r=0;r<n;++r){
		if(b[r] <= mmin){ /* outside of the [mmin,mmax] */
			y = b[r];
		}else if(b[r] >= mmax){
		/* }else{ */ /* inside of [mmin,mmax] */	
			y = (d-tightsum)/slackweight;
		}else{
			printf("\nWARN: projection error, b[%d]=%.16lf still in (%.16lf,%.16lf).\n", 
					r, b[r], mmin, mmax);
			exit(-1);
		}
		testsum += y;
		tx[r] = tx[r] + y;
	}

	mu_free((void**)(&membuff));
	return (0);
}

int DF_project(QPDATA* qp, int ind, double* tx /*IN/OUT*/)
{
	// Persistent variable to record the trace
	static double lambda = 0.0;
	static double lambda1 = 1.0;

	double dlambda = 1.0 + fabs(lambda1 - lambda);

	// CuteTimer::alloc_timer.Start();
	double* lmdx = (double*) mu_alloc(sizeof(double), qp->k);
	// CuteTimer::alloc_timer.Stop();

	double lambda_l = 0.0, lambda_u = 0.0, resd = 0.0;
	double resdl = 0.0, resdu = 0.0;
	double s = 0.0;

	double lambda_new = 0.0, lambda_old = 0.0;

	double tol_lam = 1.0e-11;
	double tol_r = 1.0e-10;

	int stop = 0;
	int steps = 0;
	// Bracketing Phrase
	bproject(qp, lambda, tx , ind, lmdx);
	resd = residual(qp, lmdx );
	if(fabs(resd) < tol_r){
		goto FREE;
	}

	// record the last step lambda
	lambda1 = lambda;

	if( resd < 0.0 ) {
		lambda_l = lambda;
		resdl = resd;
		lambda += dlambda;

		bproject(qp, lambda, tx, ind, lmdx);
		resd = residual(qp, lmdx );

		while( resd < 0 ) { // tight the lower bound
			lambda_l = lambda;
			s = max( resdl/resd - 1, 0.1 );
			dlambda += dlambda/s;
			lambda += dlambda;

			bproject( qp, lambda, tx, ind, lmdx );
			resd = residual( qp, lmdx );

		}
		// tight the upper bound with lambda
		lambda_u = lambda;
		resdu    = resd;
	} else { // tight the upper bound
		lambda_u = lambda;
		resdu = resd;
		lambda -= dlambda;

		bproject(qp, lambda, tx, ind, lmdx );
		resd = residual(qp, lmdx );

		while( resd > 0.0 ) {
			lambda_u = lambda;
			s = max( resdu/resd - 1, 0.1 );
			dlambda += dlambda/s;
			lambda -= dlambda;

			bproject( qp, lambda, tx, ind, lmdx );
			resd = residual( qp, lmdx );

		}
		// tight the lower bound with lambda
		lambda_l = lambda;
		resdl    = resd;
	}

	if(resdu == 0.0) {
		goto FREE;
	}
	// Scant Phrase
	s = 1 - resdl / resdu;
	dlambda = dlambda/s;
	lambda = lambda_u - dlambda;
	bproject(qp, lambda, tx, ind, lmdx );
	resd = residual(qp, lmdx );

	while( !stop ) {

		if( fabs(resd) < tol_r ) {
			stop = 1;
			break;
		}

		if( fabs( lambda - lambda_old ) < tol_lam * (1+lambda) ) {
			stop = 1;
			break;
		}

		lambda_old = lambda;

		if( resd > 0 ) {
			if( s <= 2 ) {
				lambda_u = lambda;
				resdu = resd;
				s = 1 - resdl/resdu;
				dlambda = (lambda_u - lambda_l)/s;
				lambda = lambda_u - dlambda;
			} else {
				s = max( resdu/resd-1, 0.1 );
				dlambda = (lambda_u - lambda)/s;
				lambda_new = max( lambda-dlambda, 0.75*lambda_l+0.25*lambda );
				lambda_u = lambda;
				resdu = resd;
				lambda = lambda_new;
				s = (lambda_u-lambda_l)/(lambda_u-lambda);
			}
		} else {
			if( s >= 2 ) {
				lambda_l = lambda;
				resdl = resd;
				s = 1 - resdl/resdu;
				dlambda = (lambda_u-lambda_l)/s;
				lambda = lambda_u - dlambda;
			} else {
				s = max(resdl/resd-1, 0.1);
				dlambda = (lambda - lambda_l)/s;
				lambda_new = min( lambda+dlambda, 0.75*lambda_u+0.25*lambda );
				lambda_l = lambda;
				resdl = resd;
				lambda = lambda_new;
				s = (lambda_u - lambda_l)/(lambda_u - lambda);
			}
		}

		bproject(qp,lambda, tx, ind, lmdx);
		resd = residual(qp, lmdx );

		steps += 1;
	}


FREE:
	memcpy(tx, lmdx, sizeof(tx[0])*qp->k);


	mu_free((void**)(&lmdx));

	return steps;
}

void calc_k_multi_d(QPDATA* qp, SPG_PARAM* param, double* d, double* kmd)
{
	register int i,j,r;
	register int jn, ik, jk /*, sr*/;

	memset(kmd, 0, sizeof(kmd[0])*(qp->n*qp->k));
#ifdef IJ
	for(i=0;i<qp->n;++i) {
		in = i*qp->n;
		ik = i*qp->k;
		for(j=0;j<qp->n;++j) {
			jk = j*qp->k;
			for(r=0;r<qp->k;++r) {
				kmd[ik+r] += qp->K[in+j] * d[jk+r];
			}
		}
	}
#else
	for(j=0;j<qp->n;++j) {
		jk = j*qp->k;
		jn = j*qp->n;

		/* check the first level sparsity */
		for(r=0;r<qp->k;++r){
			/* if(d[jk+r]!=0.0) { */
			if(fabs(d[jk+r])>param->epsilon_alga){
				for(i=0;i<qp->n;++i) {
					ik = i*qp->k;
					kmd[ik+r] += qp->K[jn+i] * d[jk+r];
				}
			}else{
				d[jk+r] = 0.0;
			}
		}
	}
#endif
	return;
}

void init_grad(QPDATA* qp, double* g, int zero)
{
	int i,r;
	int ik;
// #define PASS_GRAD
	
#ifndef PASS_GRAD
	int j, jk, jn;
#endif
	/* init g to linear coefficient */
#ifdef PASS_GRAD
	if(zero)
#endif
	{
		for(i=0;i<qp->n;++i){
			ik = i*qp->k;
			for(r=0;r<qp->k;++r){
				g[ik+r] = qp->g[ik+r];
			}
		}
	} 
#ifdef PASS_GRAD
	else {
		for(i=0;i<qp->n;++i){
			ik = i*qp->k;
			for(r=0;r<qp->k;++r){
				g[ik+r] = qp->grad[ik+r];
			}
		}
	}
#endif
	
#ifndef PASS_GRAD
	if(!zero) {
		/* then add the Hession part */
		for(j=0;j<qp->n;++j) {
			jk = j*qp->k;
			jn = j*qp->n;
			/* check the first level sparsity */
			for(r=0;r<qp->k;++r){
				for(i=0;i<qp->n;++i) {
					ik = i*qp->k;
					g[ik+r] += qp->K[jn+i] * qp->x[jk+r];
				}
			}
		}
	}
#endif
}

void update_grad(QPDATA* qp, double* g_prev, double alpha, double* kmd, double* g)
{
	int i, r;
	for(i=0;i<qp->n;++i) {
		for(r=0;r<qp->k;++r){
			g[i*qp->k+r] = g_prev[i*qp->k+r] + alpha * kmd[i*qp->k+r];
		}
	}
	return;
}

int spg_solve(QPDATA* Q, SPG_PARAM* param)
{
	// allocate local variable
	// double* x_prev = (double*) mu_alloc(sizeof(double), Q->n);
	double* tx = (double*) mu_alloc(sizeof(double), Q->m);

	double* d    = (double*) mu_alloc(sizeof(double), Q->m);
	double* kmd  = (double*) mu_alloc(sizeof(double),Q->m);

	double* g  = (double*) mu_alloc(sizeof(double), Q->m);
	double* g_prev = (double*) mu_alloc(sizeof(double), Q->m);
	double* aol = (double*) mu_alloc(sizeof(double), Q->n);
	// double f_history = (double*) mu_alloc(sizeof(double), param->M);
	// variable for updating line search parameter
	double f_best = MAX_DBL, f_c = 0.0;
	double f_ref = MAX_DBL;

	double xnorm = 0.0;
	int t = 0;
	//
	double alpha = 0.0, lambda = 0.0, lambda2 = 0.0;
	double obj = 0.0, obj_prev = 0.0;
	double llamb, rlamb;
	double inf_pg, maxf, ol, minol;
	double obj0;
	
	double eps = 1.0e-16;
	int stop = 0, steps = 0, i = 0, r, ik ;

	/* set projector */
	if(param->proj == 0) {
		proj_func = &Sort_project;
	}else if(param->proj == 1) {
		proj_func = &DF_project;
	}else if(param->proj == 2) {
		proj_func = &Pardalos_project;
	}else{
		proj_func = &Sort_project;
	}
	
	if(!param->init) {
		// project x to be feasible
		for(i=0;i<Q->n;++i) {
			(*proj_func)(Q, i, Q->x+i*Q->k);
		}
	}

	// initialize x
	double max_abs = 0.0;
	for(i = 0; i < Q->m; ++i) {
		if( max_abs < fabs(Q->x[i]) ) {
			max_abs = fabs(Q->x[i]);
		}
	}

	obj = 0.0;
	if(max_abs > 0.0) { // non-zero initial point
		// assert( pfeasible(Q, param->epsilon) == 1 );
		init_grad(Q, g, 0);
		for(i = 0; i < Q->m; ++i) {
			obj += (g[i]+Q->g[i])*Q->x[i];
		}
		obj /= 2;
	} else { // zero initial point
		init_grad(Q, g, 1);
		obj = 0.0;
	}

	obj0 = obj;
	f_best = obj0, f_c = obj0;

	minol = INF;
	for(i=0;i<Q->n;++i) {
		ik = i*Q->k;
		maxf = g[ik];
		for(r=1;r<Q->k;++r) {
			if(g[ik+r] > maxf) {
				maxf = g[ik+r];
			}
		}
		ol = 0.0;
		for(r=0;r<Q->k;++r) {
			if(g[ik+r] < maxf) {
				ol += (g[ik+r]-maxf)*((Q->y[i]==r)-Q->x[ik+r]);
			}
		}
		if(ol < minol) {
			minol = ol;
		}
		aol[i] = ol;
	}
	
	if(-minol < param->epsilon) {
		printf("Inner solver no solve since %.10lf < %.10lf\n",
				-minol, param->epsilon);
		goto CLEAN;
	}

#define DINF_INIT 1	
#define LSUM_INIT 0
#if DINF_INIT
	/* project for initial alpha(BB-Step) */
	project(Q, g, 1.0, tx);
	for( i = 0; i < Q->m; ++i ) {
		d[i] = tx[i] - Q->x[i];
		if(fabs(d[i]) <= param->epsilon_alga){
			d[i] = 0.0;
		}
	}
	inf_pg  = inf_norm(d, Q->m);
	
	if(inf_pg < 1.0/param->alpha_max) {
		printf("Inner solver no solve since %.10lf < %.10lf\n",
				inf_pg, param->epsilon);
		goto CLEAN;
	}
	
	alpha = 1.0/inf_pg;
#elif LSUM_INIT
	alpha = 0.0;
	for(i=0;i<Q->n;++i) {
		alpha += aol[i];
	}
	alpha = -alpha;
#else
	alpha = aol[0];
	for(i=1;i<Q->n;++i) {
		if(aol[i] < alpha){
			alpha = aol[i];
		}
	}
	alpha = -alpha;
#endif
	
	double ddkd /*d^TKd*/, ddg /*d^Tg*/, ddd /*d^Td*/;

	double ddkd_old, ddd_old;
	ddkd_old = ddd_old = 0.0;

	clock_t t1, t2, t3, t4, t5, t6;

#ifdef CHECK_SPARSE
	double ffull = 0.0;
	fsparsity = 0;
	fsparsity_block = 0;
#endif
	
	while( !stop ) {

		if(param->verbose >= 2) {
			printf("  ====== Iteration %d ====== \n", steps);
			fflush(stdout);
		}
		
		// Step 1. Project with BB step
		t1 = clock();
		project(Q, g, alpha, tx);
		t2 = clock();
		// Calculate each local d
		for( i = 0; i < Q->m; ++i ) {
			d[i] = tx[i] - Q->x[i];
			/* if(fabs(d[i]) <= 1.0e-10){
				d[i] = 0.0;
#ifdef CHECK_SPARSE
				fsparsity += 1;
#endif
			} */
		}
#ifdef CHECK_SPARSE
		ffull += Q->m;
#endif
		// Compute Kd in parallel
		calc_k_multi_d(Q, param, d, kmd);
		ddd = square_norm(d, Q->m);
		// Compute d^Tg in parallel
		ddg = ddot(d, g, Q->m);
		// Compute d^TKd
		ddkd = ddot(d, kmd, Q->m);
		t3 = clock();
		obj_prev = obj;
		// printf("(ddd, ddg, ddkd) = (%.lf %lf, %lf)\n", ddd, ddg, ddkd);
		if(ddkd > eps * ddd && ddg < 0) {
			lambda = -ddg/ddkd;
		} else {
			lambda =  1.0;
		}
		
		// calculate the feasible interval
		llamb = -INF; rlamb = INF;
		for(i=0;i<Q->n;++i) {
			for(r=0;r<Q->k;++r) {
				if(d[i*Q->k+r] > param->epsilon_alga) {
					if(((r==Q->y[i])-Q->x[i*Q->k+r])/d[i*Q->k+r] < rlamb) {
						rlamb = ((r==Q->y[i])-Q->x[i*Q->k+r])/d[i*Q->k+r];
					}
				}else if(d[i*Q->k+r] < -param->epsilon_alga) {
					if(((r==Q->y[i])-Q->x[i*Q->k+r])/d[i*Q->k+r] > llamb) {
						llamb = ((r==Q->y[i])-Q->x[i*Q->k+r])/d[i*Q->k+r];
					}
				}
			}
		}

		if(ddkd < param->epsilon_alga){
			if(ddg > 0){
				lambda = llamb;
			}else if(ddg < 0) {
				lambda = rlamb;
			}
		}

		if( lambda > rlamb ) {
			// printf("upper bounded.\n");
			lambda = rlamb;
		}else if(lambda < llamb) {
			// printf("lower bounded.\n");
			lambda = llamb;
		}
		
		// Calculate objective of BB step
		obj = obj_prev + 0.5 * ddkd + ddg;
		// update x according to the non-decreasing strategy
		if((steps == 0 && obj >= obj0)||(steps > 0 && obj >= f_ref)) {
			lambda2 = lambda * lambda;
			for(i = 0; i < Q->m; ++i) {
				Q->x[i] += lambda * d[i];
			}
			ddd  *= lambda2;
			ddkd *= lambda2;
			obj = obj_prev + 0.5 * ddkd + lambda * ddg;
		} else {
			lambda = 1.0;
			for(i = 0; i < Q->m; ++i) {
				Q->x[i] += d[i];
			}
		}
		t4 = clock();

		// printf("(prev_obj->objective, lambda)=(%lf->%lf, %lf) \n", obj_prev, obj, lambda);
		// Step a2. Update x
		for(i = 0; i < Q->m; ++i) {
			g_prev[i] = g[i];
		}
		// Step ... Calculate gradient for next step
		update_grad(Q, g_prev, lambda, kmd, g);
		t5 = clock();
		// Step ... check if stop
		xnorm = square_norm(Q->x, Q->m);
		xnorm = sqrt(xnorm);

// #define PG_TERMINATE
#ifdef PG_TERMINATE
		inf_pg = inf_norm(d, Q->m);
		// check projection gradient if terminate
		{
			if( inf_pg < param->epsilon ||
					steps > param->max_iteration ) {
				stop = 1;
				break;
			}
		}
		if(param->verbose >= 2) {
			printf("inf_pg = %.8lf \n", inf_pg);
		}
#else
		// check kkt if terminate
		minol = INF;
		for(i=0;i<Q->n;++i) {
			ik = i*Q->k;
			maxf = g[ik];
			for(r=1;r<Q->k;++r){
				if(g[ik+r] > maxf){
					maxf = g[ik+r];
				}
			}
			ol = 0.0;
			for(r=0;r<Q->k;++r){
				if(g[ik+r] < maxf) {
					ol += (g[ik+r]-maxf)*((Q->y[i]==r)-Q->x[ik+r]);
				}
			}
			if(ol < minol) {
				minol = ol;
			}
			aol[i] = ol;
		}
		if(-minol < param->epsilon || steps >= param->max_iteration) {
			stop = 1;
			break;
		}
#endif
		// Step 3. Update alpha for next step
		if(ddkd <= eps * ddd) {
			alpha = param->alpha_max;
		} else {
			if(ddkd_old <= eps * ddd_old) {
				alpha = ddd/ddkd;
			} else {
				alpha = (ddd+ddd_old) / (ddkd + ddkd_old);
			}
			if( alpha > param->alpha_max ) {
				alpha = param->alpha_max;
			}else if( alpha < param->alpha_min ) {
				alpha = param->alpha_min;
			}
		}
		ddkd_old = ddkd;
		ddd_old  = ddd;
		t6 = clock();
		// printf("ddd = %.12lf, ddkd = %.12lf, alpha = %.12lf.\n", ddd, ddkd, alpha);

		// Step 4. Update the f_ref
		if( obj < f_best ) {
			f_best = obj;
			f_c = obj;
			t = 0;
		} else {
			f_c = max(f_c, obj); // record the largest objective
			t += 1;
			if( t == param->M ) {
				f_ref = f_c;
				f_c   = obj;
				t     = 0;
			}
		}
		steps += 1;
#ifdef PROFILE_IT
		/* profile */
		fproject += (double(t2-t1))/CLOCKS_PER_SEC;
		fmatrixbyvec += (double(t3-t2))/CLOCKS_PER_SEC;
		fbacktracking += (double(t4-t3))/CLOCKS_PER_SEC;
		fupdate_grad += (double(t5-t4))/CLOCKS_PER_SEC;
		fupdate_steplen += (double(t6-t5))/CLOCKS_PER_SEC;
		ftotal += (double(t6-t1))/CLOCKS_PER_SEC;
#endif
	}
	
	if(param->verbose >= 1) {
		printf("spg complete with %d steps and violation=%.6lf < %.6lf.\n", steps, -minol, param->epsilon);
	}
	/* copy back the gradient \hat{F}*/
	for(i=0;i<Q->m;++i) {
		Q->grad[i] = g[i];
	}
#ifdef PROFILE_IT
	printf("Profile: \n");
	printf("  Project:      %.2lf%%\n", fproject/ftotal*100);
	printf("  Matrix*Vec:   %.2lf%%\n", fmatrixbyvec/ftotal*100);
	printf("  Backtracking: %.2lf%%\n", fbacktracking/ftotal*100);
	printf("  Update grad:  %.2lf%%\n", fupdate_grad/ftotal*100);
	printf("  Update BB:    %.2lf%%\n", fupdate_steplen/ftotal*100);
#endif

	
#ifdef CHECK_SPARSE
	printf("  Utilized sparsity: %.2lf%%\n", fsparsity_block/ffull*100);
	printf("  Real sparsity: %.2lf%%\n", fsparsity/ffull*100);
#endif
CLEAN:
	// mu_free((void**)(&x_prev));
	mu_free((void**)(&aol));
	mu_free((void**)(&tx));
	mu_free((void**)(&d));
	mu_free((void**)(&kmd));
	mu_free((void**)(&g));
	mu_free((void**)(&g_prev));

	// mu_free(&f_history);
	return steps;
}
