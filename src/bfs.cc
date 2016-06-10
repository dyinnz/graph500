/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-28
 ******************************************************************************/

#include <cstring>
#include <queue>
#include "utility.h"
#include "construct.h"
#include "bfs_gpu.h"

int64_t * __restrict__ g_local_adja_arrays {nullptr};
int64_t * __restrict__ g_local_csr_mem {nullptr};

static inline int64_t adja_beg(int64_t u) {
  return g_local_adja_arrays[u * 2];
}

static inline int64_t adja_end(int64_t u) {
  return g_local_adja_arrays[u * 2 + 1];
}

static inline int64_t next_vertex(int64_t offset) {
  return g_local_csr_mem[offset];
}

void SettingCSRGraph(LocalCSRGraph &local_csr) {
  g_local_adja_arrays = (int64_t*)local_csr.adja_arrays();
  g_local_csr_mem = local_csr.csr_mem();
}

#if 0


static int64_t *
NaiveBFS(CSRGraph &csr, int64_t root) {
  auto bfs_tree = new int64_t[csr.vertex_num()];
  memset(bfs_tree, -1, sizeof(int64_t) * csr.vertex_num());
  bfs_tree[root] = root;

  std::queue<int64_t> q;
  q.push(root);
  while (!q.empty()) {
    int64_t u = q.front();
    q.pop();

    for (auto iter = csr.adja_beg(u); iter != csr.adja_end(u); ++iter) {
      int64_t v = csr.next_vertex(iter);
      if (-1 == bfs_tree[v]) {
        bfs_tree[v] = u;
        q.push(v);
      }
    }
  }

  /*
  for (int64_t v = 0; v < csr.vertex_num(); ++v) {
    printf("u[%ld] -> v[%ld]\n", bfs_tree[v], v);
  }
  */

  return bfs_tree;
}

static int64_t*
MPIBFS(CSRGraph &csr, int64_t root) {
  auto bfs_tree = new int64_t[csr.vertex_num()];

  return bfs_tree;
}

#endif

int64_t *
BuildBFSTree(LocalCSRGraph &local_csr, int64_t root) {
  mpi_log_barrier();
  logger.log("begin bfs, root: %d...\n", root);
  SettingCSRGraph(local_csr);

  int64_t *bfs_tree = new int64_t[local_csr.local_v_num()];

  //bfs_cu(root, g_local_adja_arrays, local_csr.local_v_num(), g_local_csr_mem, local_csr.csr_mem_size());
  CudaBFS(root,
      g_local_adja_arrays,
      local_csr.local_v_num(),
      local_csr.global_v_num(),
      local_csr.local_v_beg(),
      local_csr.local_v_end(),
      g_local_csr_mem,
      local_csr.csr_edge_num(),
      bfs_tree);

  MPI_Barrier(MPI_COMM_WORLD);
  logger.log("end bfs.\n");
  return bfs_tree;
}

