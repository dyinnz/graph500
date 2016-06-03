/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-28
 ******************************************************************************/

#include <cassert>
#include <cstring>
#include <vector>
#include <queue>
#include <list>
#include "utility.h"
#include "verify.h"

using std::vector;

#if 0

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

#endif

const int Verifier::kMaxLevel = INT_MAX;

bool
Verifier::CheckRange() {
  for (int64_t v = 0; v < _local_v_num; ++v) {
    int64_t u = _parents[v];
    if (! (-1 == u || (0 <= u && u < _global_v_num)) ) {
      logger.mpi_log("%s(): verify %ld 's parent '%ld FAILED!\n", 
          __func__, v, u);
      return false;
    }
  }
  return true;
}

bool
Verifier::CheckParentOfRoot() {
  int64_t average = _global_v_num / settings.mpi_size;
  if (mpi_get_owner(_root, average) == settings.mpi_rank) {
    int64_t local_root = _root - _local_v_beg;
    if (_parents[local_root] != _root) {
      logger.mpi_log("%s(): Verify root[%ld]'s parent FAILED\n", _root);
      return false;
    }
  }
  return true;
}

bool 
Verifier::CheckParentOfOthers() {
  int64_t average = _global_v_num / settings.mpi_size;
  for (int64_t v = 0; v < _local_v_num; ++v) {
    int64_t global_u = _parents[v];
    int64_t global_v = v + _local_v_beg;
    if (global_v != _root) {
      if (_parents[v] == global_v) {
        logger.mpi_log("%s(): Verify v[%ld]'s parent FAILED\n", global_v);
        return false;
      }
    }
  }
  return true;
}

bool
Verifier::ComputeLevels() {
  bool result {true};

  // init
  vector<bool> visited(_local_v_num, false);
  vector<int> parent_levels(_local_v_num, kMaxLevel);
  _levels.resize(_local_v_num, kMaxLevel);

  if (_local_v_beg <= _root && _root < _local_v_end) {
    _levels[_root - _local_v_beg] = 0;
    visited[_root - _local_v_beg] = true;
  }

  // init MPI resource
  int64_t average = _global_v_num / settings.mpi_size;

  _win = new MPI_Win;
  memset(_win, 0, sizeof(MPI_Win));
  MPI_Win_create(_levels.data(), _local_v_num * sizeof(int), sizeof(int), 
      MPI_INFO_NULL, MPI_COMM_WORLD, _win);

  bool is_done {false};
  for (int level = 1; !is_done; ++level) {
    is_done = true;

    MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, *_win);
    // skip unconnected vertex
    for (int64_t v = 0; v < _local_v_num; ++v) if (!visited[v]) {
      int64_t global_v = _local_v_beg + v;
      int64_t global_u = _parents[v];
      if (-1 == global_u || _root == global_v) continue;
      int64_t u_owner = mpi_get_owner(global_u, average);

      if (u_owner == settings.mpi_rank) {
        // get parent level from local
        parent_levels[v] = _levels[global_u - _local_v_beg];

      } else {
        // get parent level from remote
        int64_t remote_local_u = global_u - u_owner * average;
        MPI_Get(parent_levels.data() + v, 1, MPI_INT, u_owner, 
            remote_local_u, 1, MPI_INT, *_win);
      }
    }
    MPI_Win_fence(MPI_MODE_NOSUCCEED, *_win);

    for (int64_t v = 0; v < _local_v_num; ++v)  {
      if (!visited[v] && parent_levels[v] < kMaxLevel) {

        if (parent_levels[v] != level-1) {
          is_done = true;
          result = false;
          logger.mpi_error("the parent of v[%ld] is not correct\n",
              v + _local_v_beg);

        } else {
          _levels[v] = parent_levels[v] + 1;
          visited[v] = true;
          is_done = false;

          logger.mpi_debug("update level of v[%ld] : level[%d]\n", 
              v + _local_v_beg, _levels[v]);
        }
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &is_done, 1, MPI_CHAR, MPI_LAND, MPI_COMM_WORLD);
  }

  for (int64_t v = 0; v < _local_v_num; ++v) {
    logger.mpi_debug("v[%ld]'s level is %d\n", v+_local_v_beg, _levels[v]);
  }

  return result;
}

bool
Verifier::CheckEdges() {
#ifdef DEBUG
  mpi_log_barrier();
#endif

  bool result {true};

  assert(_levels.size() == _local_v_num);

  vector<std::pair<int, int>> edges_levels(_local_raw.edge_num, 
                                           {kMaxLevel, kMaxLevel});
  int64_t average = _global_v_num / settings.mpi_size;

  // lambda for fetch level from local or remote
  auto fetch_level = [&](int64_t global_v, int &out_level) {
    int64_t onwer = mpi_get_owner(global_v, average);
    if (onwer == settings.mpi_rank) {
      out_level = _levels[global_v - _local_v_beg];

    } else {
      int64_t remote_local_v = global_v - onwer * average;
      MPI_Get(&out_level, 1, MPI_INT, onwer, 
          remote_local_v, 1, MPI_INT, *_win);
    }
  };

  // logger.mpi_debug("begin of win fence\n");

  // collect all levels needing
  MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, *_win);
  for (int64_t e = 0; e < _local_raw.edge_num; ++e) {
    fetch_level(raw_edge_u(e), edges_levels[e].first);
    fetch_level(raw_edge_v(e), edges_levels[e].second);
  }
  MPI_Win_fence(MPI_MODE_NOSUCCEED, *_win);

  // logger.mpi_debug("end of win fence\n");

  // calc
  for (auto &level_p : edges_levels) {
  /*for (int64_t e = 0; e < _local_raw.edge_num; ++e) {*/
    /*auto &level_p = edges_levels[e];*/
    if ((kMaxLevel == level_p.first && kMaxLevel != level_p.second) ||
        (kMaxLevel != level_p.first && kMaxLevel == level_p.second)) {
      logger.mpi_error("the levels of edges are not correct: u_l[%d],v_l[%d]\n",
          level_p.first, level_p.second);
      result = false;
    }
    if (abs(level_p.first - level_p.second) > 1) {
      logger.mpi_error("the levels of edges are not correct: u_l[%d],v_l[%d]\n",
          level_p.first, level_p.second);
      result = false;
    }

     /*logger.mpi_debug("check edges[%ld] u_l[%d] v_l[%d] \n", e, level_p.first, level_p.second);*/
  }

  MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_CHAR, MPI_BAND, MPI_COMM_WORLD);
  return result;
}

bool 
Verifier::Verify() {
  bool result {true};

  do {
    if (!CheckRange()) {
      result = false;
      break;
    }
    logger.debug("CheckRange PASS\n");

    if (!CheckParentOfRoot()) {
      result = false;
      break;
    }
    logger.debug("CheckParentOfRoot PASS\n");

    if (!CheckParentOfOthers()) {
      result = false;
      break;
    }
    logger.debug("CheckParentOfOthers PASS\n");

    if (!ComputeLevels()) {
      result = false;
      break;
    }
    logger.debug("ComputeLevels PASS\n");

    if (!CheckEdges()) {
      result = false;
      break;
    }
    logger.debug("CheckEdges PASS\n");

  } while (false);

  MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_CHAR, MPI_BAND, MPI_COMM_WORLD);
  return result;
}

bool 
VerifyBFSTree(int64_t *parents, 
    int64_t global_v_num, 
    int64_t root,
    LocalRawGraph &local_raw) {
  Verifier verifier(parents, global_v_num, root, local_raw);
  return verifier.Verify();
}

void
TEST_BFSTree::Init() {
  assert(2 == settings.mpi_size);

  global_v_num = 5;
  root = 0;
  raw.global_edge_num = 5;

  if (0 == settings.mpi_rank) {
    parents = new int64_t[2];
    parents[0] = 0;
    parents[1] = 0;

    raw.edge_num = 3;
    raw.edges = new Edge[raw.edge_num];
    raw.edges[0] = {0, 2};
    raw.edges[1] = {0, 1};
    raw.edges[2] = {1, 2};

  } else if (1 == settings.mpi_rank) {
    parents = new int64_t[3];
    parents[0] = 0;   // global 2
    parents[1] = 1;   // global 3
    parents[2] = -1;   // global 4

    raw.edge_num = 2;
    raw.edges = new Edge[raw.edge_num];
    raw.edges[0] = {3, 2};
    raw.edges[1] = {3, 1};
  }
}

void TEST_VerifyCase_1() {
  logger.log("Run %s()\n", __func__);
  TEST_BFSTree bfs_tree;
  bfs_tree.Init();
  logger.debug("Init %s() data ok\n", __func__);

  if (VerifyBFSTree(bfs_tree.parents, bfs_tree.global_v_num, bfs_tree.root,
        bfs_tree.raw)) {
    logger.log("TEST Case 1 PASS\n");
  } else {
    logger.error("TEST Case 1 FAILED\n");
  }
}
