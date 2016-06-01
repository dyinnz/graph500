/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-28
 ******************************************************************************/

#include <cstring>
#include <vector>
#include <queue>
#include <list>
#include "utility.h"

using std::vector;

/**
 * @param[out]    levels
 */
bool ComputeLevels(int64_t *bfs_tree,
                   int64_t vertex_num,
                   int64_t root,
                   vector<int64_t> &levels) {
  int64_t *&parent = bfs_tree;
  bool ret {true};

  levels[root] = 0;

  // rebuild a real tree
  vector<std::list<int64_t>> level_tree(vertex_num);
  for (int64_t v = 0; v < vertex_num; ++v) {
    int64_t u = parent[v];
    if (-1 != u && v != root) {
      level_tree[u].push_back(v);
    }
  }

  std::queue<int64_t> q;
  q.push(root);
  while (!q.empty() && ret) {
    int64_t u = q.front(); q.pop();
    for (int64_t v : level_tree[u]) {
      if (levels[v] > 0) {
        logger.error("compute level error: u[%d] -> v[%d]\n"
                     "may contain cycle!\n" , u, v);
        ret = false;
        break;
      } else {
        levels[v] = levels[u] + 1;
        q.push(v);
      }
    }
  }

  /*
  for (int64_t v = 0; v < vertex_num; ++v) {
    printf("v[%ld] level: %ld\n", v, levels[v]);
  }
  */

  if (ret) {
    logger.log("compute level: PASS\n", ret);
  } else {
    logger.log("compute level: FAILED\n", ret);
  }
  return ret;
}

bool VerifyBFSTree(int64_t *bfs_tree,
                   int64_t vertex_num,
                   int64_t root,
                   Edge *edges,
                   int64_t edge_desired_num) {
  bool ret {false};

  if (root >= vertex_num) {
    return false;
  }

  vector<int64_t> levels(vertex_num, -1);
  if (!ComputeLevels(bfs_tree, vertex_num, root, levels)) {
    return false;
  }

  for (int64_t v = 0; v < vertex_num; ++v) {
    int64_t u = bfs_tree[v];
    if (-1 == u || root == v) continue;
    if (1 != abs(levels[u] - levels[v])) {
      logger.error("the levels of u[%d] and v[%d] of edge in bfs tree do not differ by one\n", u, v);
      return false;
    }
  }
  logger.log("each tree edge connects vertices whose BFS levels differ by exactly one: PASS\n");

  for (int64_t e = 0; e < edge_desired_num; ++e) {
    int64_t u = edges[e].u;
    int64_t v = edges[e].v;
    if ((levels[v] < 0 && levels[u] >= 0) || (levels[v] >= 0 && levels[u] < 0)) {
      logger.error("u[%d] and v[%d] of edge in raw edges array: one in bfs tree, one not\n", u, v);
      return false;
    }
    if (abs(levels[u] - levels[v]) > 1) {
      logger.error("the levels of u[%d] and v[%d] of edge in raw edges array differ more than one\n", u, v);
      return false;
    }
  }
  logger.log("every edge in the input list has vertices with levels that differ by at most one or that both are not in the BFS tree: PASS\n");

  vector<bool> connected(vertex_num, false);
  for (int64_t e = 0; e < edge_desired_num; ++e) {
    int64_t u = edges[e].u;
    int64_t v = edges[e].v;
    if (u != v) {
      connected[v] = connected[u] = true;
    }
  }
  for (size_t v = 0; v < connected.size(); ++v) {
    if (connected[v] && -1 == bfs_tree[v]) {
      logger.error("v[%zu] is connected but not in bfs tree\n", v);
    }
  }
  logger.log("the BFS tree spans an entire connected component's vertices: PASS\n");
  return ret;
}

