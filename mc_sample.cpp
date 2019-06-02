/*
 * McSample.cpp
 *
 *  Created on: 2010-6-19
 *      Author: jianmin
 */

#include "mc_sample.h"

bool McSample::buildSampleIndex() {
	// build the sample index
	// i) count the number of sample
	std::vector<TaggerImpl*>::const_iterator it;
	
	int ysize = findex->ysize();
	int mctag = 0, nseq = 0;
	m=0;
	indmap.clear();
	std::map<int,int>::const_iterator pos;
	for(it = sample.begin(); it != sample.end(); ++it) {
		TaggerImpl* seq = *it;
		findex->rebuildFeatures(seq);
		for(size_t i=1; i < seq->size(); ++i) {
			SeqIndex idx;
			idx.n = nseq;
			idx.p = i;
			indmap.push_back(idx);
			
			int pair = seq->answer(i-1)*ysize + seq->answer(i);
			int tag;
			pos = tag2tag.find(pair); 
			if(pos == tag2tag.end()) {
				tag2tag[pair] = mctag;
				tag = mctag;
				
				mctag += 1;
			}else{
				tag = pos->second;
			}
			_y.push_back(tag);
			m += 1;
		}
		nseq += 1;
	}
	k = mctag;
	n = findex->size();
	
	n2cmap = new int[m];
	std::fill(n2cmap, n2cmap + m, -1);
	
	sqnorm = new float[m];
	std::fill(sqnorm, sqnorm + m, 0);
	
	fval   = new int[n];
	std::fill(fval, fval + n, 0);
	
	y      = new int[m];
	std::copy(_y.begin(), _y.end(), y);
	
	// ii) allocate the buffer
	for(int i=0; i<nc; ++i) {
		Example* exa = cache[i];
		exa->n  = 0;
		exa->ix = new int[n];
		exa->x  = new float[n];
	}
	std::cout << "Number of sample: " << m << std::endl;
	std::cout << "Size of dimension: " << n << std::endl;
	std::cout << "Size of tags: " << k << std::endl;
	return true;
}

const Example* McSample::get(size_t i) {
	
	if(i > indmap.size()) return NULL;
	if(n2cmap[i] >= 0) {
		return cache[n2cmap[i]];
	}
	
	SeqIndex idx = indmap[i];
	TaggerImpl* seq = sample[idx.n];
	
	Example* exa = cache[ic];
	// build the example
	size_t ysize = findex->ysize();
	
	size_t ptag = seq->answer(idx.p-1);
	size_t tag  = seq->answer(idx.p);
	size_t pair = ptag*ysize+tag;
	exa->y   = tag2tag[pair];
	y[i] = exa->y;
	
#ifdef _DEBUG
	std::cout << "McSample::get("<<i<<":(" << idx.n << "," << idx.p << "))" << std::endl;
	std::cout << "ptag="<< ptag <<",tag=" << tag << std::endl;
	std::cout << "size=" << seq->size() << ", ysize=" << seq->ysize() << std::endl;
#endif
	
	buildFeatures(seq->emission_vector(idx.p-1, ptag), ptag); 
	buildFeatures(seq->emission_vector(idx.p,   tag),  tag);
	buildFeatures(seq->prev_transition_vector(idx.p, tag, ptag), pair);
	
	// gen the feature vector
	size_t p=0;
	sqnorm[i] = 0;
	
	for(int j = 0; j < n; ++j) {
		if(fval[j] != 0) {
			exa->ix[p] = j;
			exa->x[p]  = fval[j];
			sqnorm[i] += fval[j]*fval[j];	
			fval[j] = 0;
			p+=1;
		}
	}
	
	exa->n = p;
	// update the index mapping
	if(c2nmap[ic] >= 0) {
		n2cmap[c2nmap[ic]] = -1;
	}
	
	c2nmap[ic] = i;
	n2cmap[i]  = ic;
	
	ic += 1;
	if(ic == nc) ic = 0;
	
	return exa;
}

void McSample::buildFeatures(const int* features, size_t tag) {
	size_t k = 0;
	while(features[k] >= 0) {
		fval[features[k]+tag] += 1;
		k+=1;
	}
}
McSample::~McSample() {
	delete n2cmap;
	delete c2nmap;
	delete fval;
	delete sqnorm;
	delete []cache;
}
