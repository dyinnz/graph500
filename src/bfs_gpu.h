#pragma once

#include "utility.h"

// struct

struct CudaInfo {
  int64_t blocks_number;
  int64_t threads_per_block;
};


struct HostInfo {
  // pass in
  int64_t root;
  int64_t *adja_arrays;
  int64_t local_v_num;
  int64_t global_v_num;
  int64_t local_v_beg;
  int64_t local_v_end;
  int64_t *csr;
  int64_t csr_edge_num;
  int64_t *bfs_tree;

  // calc
  int64_t average;
  bool change;

  // tmp
  bit_type *local_bitmap;
  bit_type *global_bitmap;
};


struct CudaGraphMemory {
  int64_t *adja_arrays;
  int64_t *csr;
  int64_t *bfs_tree;
  bit_type *local_bitmap;
  bit_type *global_bitmap;
  bool *p_change;
};


/*----------------------------------------------------------------------------*/


void CudaBFS(int64_t root,
             int64_t *adja_arrays,
             int64_t local_v_num,
             int64_t global_v_num,
             int64_t local_v_beg,
             int64_t local_v_end,
             int64_t *csr,
             int64_t csr_edge_num,
             int64_t *bfs_tree);
