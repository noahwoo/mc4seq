#include "mem_util.h"
#include <errno.h>
#include <stdlib.h>

extern "C" {
void* mu_alloc(int size, int len)
{
	void* p = malloc( len * size );
	if( NULL == p ) {
		perror("Cannot malloc required memory");
		exit(1);
	}
	return p;
}

void* mu_calloc(int size, int len)
{
	void* p = calloc(len, size);
	if( NULL == p ) {
		perror("Cannot calloc required memory");
		exit(1);
	}
	return p;
}

void mu_free(void** p) {
	free(*p);
	*p = NULL;
}
}

