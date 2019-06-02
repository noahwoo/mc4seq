/*
 * McSample.h
 *
 *  Created on: 2010-6-19
 *      Author: jianmin
 */

#ifndef MCSAMPLE_H_
#define MCSAMPLE_H_

#include "tagger.h"
using namespace CRFPP;

struct Example {
	int n;
	int* ix;
	float* x;
	
	int y;
};

struct SeqIndex {
	size_t n;
	size_t p;
};

class McSample {

	FeatureIndex* findex;
	const std::vector<CRFPP::TaggerImpl* >& sample;
	
	// tag map
	std::map<int, int> tag2tag;
	
	// example to sequence
	std::vector<SeqIndex> indmap;
	
	// example cache
	int* n2cmap;
	int ic, nc;
	int* c2nmap;
	Example** cache;
	
	// label
	std::vector<int> _y;
	// feature value vector
	int* fval;
	
	bool buildSampleIndex();
	inline void buildFeatures(const int* features, size_t tag);
public:
	// interface for DASMC
	int m;
	int n;
	int k;
	int *y;
	float* sqnorm;
public:
	McSample(FeatureIndex* index, const std::vector<TaggerImpl* > &x): findex(index), sample(x), 
										ic(0), nc(4), m(0), n(0) {
		cache = new Example*[nc];
		c2nmap = new int[nc];
		for(int i=0; i<nc; ++i) {
			c2nmap[i] = -1;
			cache[i] = new Example;
		}
		buildSampleIndex();
	}
    const Example* get(size_t);
	virtual ~McSample();
};

#endif /* MCSAMPLE_H_ */
