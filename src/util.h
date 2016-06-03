/**
copyright by:
George Washington University
Hao Howie Huang
howie@gwu.edu
**/
#ifndef __UTIL_H__
#define __UTIL_H__
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>

#define LOCK(vert, lock) while(!__sync_bool_compare_and_swap(lock+vert,0,-1))
#define UNLOCK(vert, lock) lock[vert]=0

///change to int for SCC
//typedef long index_t;
//typedef long vertex_t;
//typedef double path_t;
//typedef long depth_t;

typedef int64_t index_t;
typedef int64_t vertex_t;
typedef double path_t;
typedef int depth_t;

#define INFTY (float)10000000 
#define ORPHAN	(unsigned char)254
#define UNVIS		(long)-1

inline off_t fsize(const char *filename) {
	struct stat st; 
	if (stat(filename, &st) == 0)
		return st.st_size;
	return -1; 
}

#endif
