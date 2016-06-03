/**
copyright by:
George Washington University
Hao Howie Huang
howie@gwu.edu
**/
#ifndef __GRAPH_H__
#define __GRAPH_H__
#include "util.h"
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "wtime.h"
class graph
{
	public:
		index_t *beg_pos;
		index_t vert_count;
		vertex_t *csr;
		index_t edge_count;
		path_t *weight;
		vertex_t *src_list;
		index_t src_count;

	public:
		graph(){};
		~graph(){};
		graph(index_t *mbeg_pos,
              const index_t mvert_count,
              vertex_t *mcsr,
              const index_t medge_count) : beg_pos(mbeg_pos), vert_count(mvert_count), csr(mcsr), edge_count(medge_count) {};

		graph(const char *beg_file,
				const char *csr_file);

		void gen_src(){};
		void groupby(){};
};
#endif
