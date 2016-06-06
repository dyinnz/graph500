#pragma once

#include "util.h"

void bfs_cu(index_t root,
            index_t *beg_pos,
            const index_t vert_count,
            vertex_t *csr,
            const index_t edge_count,
            const index_t fist_alpha=2);

void CudaBFS(int64_t root, 
             int64_t *adja_arrays, 
             int64_t local_v_num, 
             int64_t global_v_num,
             int64_t *csr,
             int64_t csr_edge_num,
             int64_t *bfs_tree);

