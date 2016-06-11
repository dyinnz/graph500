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
using std::pair;

const int Verifier::kMaxLevel = INT_MAX;

bool
Verifier::CheckRange() {
  logger.mpi_debug("%s()\n", __func__);
  for (int64_t v = 0; v < _local_v_num; ++v) {
    int64_t u = _parents[v];
    if (! (-1 == u || (0 <= u && u < _global_v_num)) ) {
      logger.mpi_error("%s(): verify %ld 's parent '%ld FAILED!\n", 
          __func__, v, u);
      return false;
    }
  }
  return true;
}

bool
Verifier::CheckParentOfRoot() {
  logger.mpi_debug("%s()\n", __func__);
  int64_t average = _global_v_num / settings.mpi_size;
  if (mpi_get_owner(_root, average) == settings.mpi_rank) {
    int64_t local_root = _root - _local_v_beg;
    if (_parents[local_root] != _root) {
      logger.mpi_error("%s(): Verify root[%ld]'s parent FAILED\n", _root);
      return false;
    }
  }
  return true;
}

bool 
Verifier::CheckParentOfOthers() {
  logger.mpi_debug("%s()\n", __func__);
  int64_t average = _global_v_num / settings.mpi_size;
  for (int64_t v = 0; v < _local_v_num; ++v) {
    int64_t global_u = _parents[v];
    int64_t global_v = v + _local_v_beg;
    if (global_v != _root) {
      if (_parents[v] == global_v) {
        logger.mpi_error("%s(): Verify v[%ld]'s parent FAILED\n", global_v);
        return false;
      }
    }
  }
  return true;
}

bool
Verifier::ComputeLevels() {
  logger.mpi_debug("%s():\n", __func__);
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

    logger.mpi_debug("compute levels %d\n", level);

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
          //logger.mpi_error("the parent of v[%ld] is not correct\n",
              //v + _local_v_beg);

        } else {
          _levels[v] = parent_levels[v] + 1;
          visited[v] = true;
          is_done = false;

          //logger.mpi_debug("update level of v[%ld] : level[%d]\n", 
              //v + _local_v_beg, _levels[v]);
        }
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &is_done, 1, MPI_CHAR, MPI_LAND, MPI_COMM_WORLD);
  }

  //for (int64_t v = 0; v < _local_v_num; ++v) {
    //logger.mpi_debug("v[%ld]'s level is %d\n", v+_local_v_beg, _levels[v]);
  //}

  return result;
}

bool
Verifier::CheckEdgeDistance() {
  logger.mpi_debug("%s()\n", __func__);
#ifdef DEBUG
  mpi_log_barrier();
#endif

  bool result {true};

  assert(_levels.size() == _local_v_num);

  vector<pair<int, int>> edges_levels(_local_raw.edge_num, 
                                           {kMaxLevel, kMaxLevel});
  int64_t average = _global_v_num / settings.mpi_size;

  // lambda for fetch level from local or remote
  auto fetch_level = [&](int64_t global_v, int &out_level) {
    int64_t owner = mpi_get_owner(global_v, average);
    if (owner == settings.mpi_rank) {
      out_level = _levels[global_v - _local_v_beg];

    } else {
      int64_t remote_local_v = global_v - owner * average;
      MPI_Get(&out_level, 1, MPI_INT, owner, 
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
  for (size_t e = 0; e < edges_levels.size(); ++e) {
    auto &level_p = edges_levels[e];
    if ((kMaxLevel == level_p.first && kMaxLevel != level_p.second) ||
        (kMaxLevel != level_p.first && kMaxLevel == level_p.second)) {
      logger.mpi_error("the levels of edges are not correct: u[%ld]<->v[%ld]; u_l[%d],v_l[%d]\n",
          raw_edge_u(e), raw_edge_v(e),
          level_p.first, level_p.second);
      result = false;
    }
    if (abs(level_p.first - level_p.second) > 1) {
      logger.mpi_error("the levels of edges are not correct: u[%ld]<->v[%ld]; u_l[%d],v_l[%d]\n",
          raw_edge_u(e), raw_edge_v(e),
          level_p.first, level_p.second);
      result = false;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_CHAR, MPI_BAND, MPI_COMM_WORLD);
  return result;
}

vector<pair<int64_t, int64_t>>
Verifier::GetPairParents() {
  vector<pair<int64_t, int64_t>> pair_parents(_local_raw.edge_num,
      {-1, -1});

  MPI_Win parents_win;
  MPI_Win_create(_parents, _local_v_num * sizeof(int64_t), sizeof(int64_t),
      MPI_INFO_NULL, MPI_COMM_WORLD, &parents_win);

  int64_t average = _global_v_num / settings.mpi_size;

  auto fetch_parent = [&](int64_t global_v, int64_t &parent) {
    int64_t owner = mpi_get_owner(global_v, average);
    if (owner == settings.mpi_rank) {
      parent = _parents[global_v - _local_v_beg];

    } else {
      int64_t remote_local_v = global_v - owner * average;
      MPI_Get(&parent, 1, MPI_LONG_LONG, owner,
          remote_local_v, 1, MPI_LONG_LONG, parents_win);
    }
  };

  MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, parents_win);
  for (int64_t e = 0; e < _local_raw.edge_num; ++e) {
    fetch_parent(raw_edge_u(e), pair_parents[e].first);
    fetch_parent(raw_edge_v(e), pair_parents[e].second);
  }
  MPI_Win_fence(MPI_MODE_NOSUCCEED, parents_win);
  MPI_Win_free(&parents_win);

  for (int64_t e = 0; e < _local_raw.edge_num; ++e) {
    //logger.mpi_debug("edge u: %ld<-%ld, v: %ld<-%ld\n", 
        //raw_edge_u(e), pair_parents[e].first,
        //raw_edge_v(e), pair_parents[e].second);
  }

  return pair_parents;
}

vector<int8_t> 
Verifier::UpdateParentsValid(
    const vector<pair<int64_t, int64_t>> &pair_parents) {
  vector<int8_t> parents_valid(_local_v_num, false);
  int64_t average = _global_v_num / settings.mpi_size;
  int8_t kTrue {true};

  // update valid
  MPI_Win valid_win;
  MPI_Win_create(parents_valid.data(), _local_v_num * sizeof(int8_t), 
      sizeof(int8_t), MPI_INFO_NULL, MPI_COMM_WORLD, &valid_win);

  auto update_valid = [&](int64_t global_v) {
    int64_t owner = mpi_get_owner(global_v, average);
    if (owner == settings.mpi_rank) {
      parents_valid[global_v - _local_v_beg] = true;

    } else {
      int64_t remote_local_v = global_v - owner * average;
      MPI_Put(&kTrue, 1, MPI_CHAR, owner,
          remote_local_v, 1, MPI_CHAR, valid_win);
    }
  };

  MPI_Win_fence(MPI_MODE_NOPRECEDE, valid_win);
  for (int64_t e = 0; e < _local_raw.edge_num; ++e) {
    int64_t global_u = raw_edge_u(e);
    int64_t global_v = raw_edge_v(e);
    if (global_v == pair_parents[e].first) {
      // logger.mpi_debug("update edges %ld<-%ld\n", global_u, global_v);
      update_valid(global_u);
    } 
    if (global_u == pair_parents[e].second) {
      // logger.mpi_debug("update edges %ld<-%ld\n", global_v, global_u);
      update_valid(global_v);
    }
  }
  MPI_Win_fence(MPI_MODE_NOSUCCEED, valid_win);
  MPI_Win_free(&valid_win);
  return parents_valid;
}

bool
Verifier::CheckTreeEdgeInGraph() {
  logger.mpi_debug("%s()\n", __func__);
  vector<int8_t> parents_valid = UpdateParentsValid(GetPairParents());

  bool result {true};
  for (int64_t v = 0; v < _local_v_num; ++v) {
    if (!parents_valid[v] && -1 != _parents[v] && _root != v+_local_v_beg) {
      logger.mpi_error("one edge %ld<->%ld of bfs tree is not in raw graph\n",
          v + _local_v_beg, _parents[v]);
      result = false;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_CHAR, MPI_BAND, MPI_COMM_WORLD);
  return result;
}

bool 
Verifier::Verify() {
  mpi_log_barrier();
  logger.log("Run %s()\n", __func__);
  bool result {true};

  auto sync_result = [&] {
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_CHAR, MPI_BAND, MPI_COMM_WORLD);
    return result;
  };

  do {
    result = CheckRange();
    if (!sync_result()) {
      break;
    }
    logger.mpi_debug("CheckRange PASS\n");

    result = CheckParentOfRoot();
    if (!sync_result()) {
      break;
    }
    logger.mpi_debug("CheckParentOfRoot PASS\n");

    result = CheckParentOfOthers();
    if (!sync_result()) {
      break;
    }
    logger.mpi_debug("CheckParentOfOthers PASS\n");

    result = ComputeLevels();
    if (!sync_result()) {
      break;
    }
    logger.mpi_debug("ComputeLevels PASS\n");

    result = CheckEdgeDistance();
    if (!sync_result()) {
      break;
    }
    logger.mpi_debug("CheckEdgeDistance PASS\n");

  } while (false);

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
