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
    _Pragma("omp parallel for")
    for (auto iter = csr.adja_beg(u); iter != csr.adja_end(u); ++iter) {
      int64_t v = csr.next_vertex(iter);
      if (-1 == bfs_tree[v]) {
        _Pragma("omp atomic update") bfs_tree[v] = u;
        _Pragma("omp atomic update") q.push(v);
      }
    }
  }

  for (int64_t v = 0; v < csr.vertex_num(); ++v) {
    printf("u[%ld] -> v[%ld]\n", bfs_tree[v], v);
  }

  return bfs_tree;
}

int64_t *
BuildBFSTree(CSRGraph &csr, int64_t root) {
  // logger.log("begin bfs, root: %d...\n", root);
  int64_t *bfs_tree = NaiveBFS(csr, root);

  // logger.log("end bfs.\n");
  return bfs_tree;
}


