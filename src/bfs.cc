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

int64_t * __restrict__ g_adja_arrays {nullptr};
int64_t * __restrict__ g_csr_mem {nullptr};

static inline int64_t adja_beg(int64_t u) {
  return g_adja_arrays[u * 2];
}

static inline int64_t adja_end(int64_t u) {
  return g_adja_arrays[u * 2 + 1];
}

static inline int64_t next_vertex(int64_t offset) {
  return g_csr_mem[offset];
}

#if 0
void SettingCSRGraph(CSRGraph &csr) {
  g_adja_arrays = (int64_t*)csr.adja_arrays();
  g_csr_head = csr.csr_head();
}

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

int64_t *
BuildBFSTree(CSRGraph &csr, int64_t root) {
  SettingCSRGraph(csr);

  // logger.log("begin bfs, root: %d...\n", root);
  int64_t *bfs_tree = NaiveBFS(csr, root);

  // logger.log("end bfs.\n");
  return bfs_tree;
}

#endif
