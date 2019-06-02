#ifndef MEM_UTIL_H_
#define MEM_UTIL_H_
#include <stdlib.h>
#include <stdio.h>
extern "C" {
void* mu_alloc(int size, int len);
void* mu_calloc(int size, int len);
void mu_free(void** p);
}
#endif /*MEM_UTIL_H_*/
