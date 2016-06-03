#pragma once

#include "util.h"

void bfs_cu(index_t root,
            index_t *beg_pos,
            const index_t vert_count,
            vertex_t *csr,
            const index_t edge_count,
            const index_t fist_alpha=2);
