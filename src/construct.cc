/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-27
 ******************************************************************************/

#include <cassert>
#include <cstring>
#include <vector>
#include <algorithm>

#include "utility.h"
#include "construct.h"

using std::vector;
using std::tuple;
using std::tie;
using std::make_tuple;

/*----------------------------------------------------------------------------*/


void LocalCSRGraph::GetVertexNumber() {
  int64_t max_vn { -1 };
  for (int64_t e = 0; e < _local_raw.edge_num; ++e) {
    max_vn = std::max(max_vn, raw_edge_u(e));
    max_vn = std::max(max_vn, raw_edge_v(e));
  }
  MPI_Allreduce(MPI_IN_PLACE, &max_vn, 1, MPI_LONG_LONG, MPI_MAX,
      MPI_COMM_WORLD);
  // _global_v_num = max_vn + 1;
  _global_v_num = (1L << settings.scale);

  tie(_local_v_beg, _local_v_end) = mpi_local_range(_global_v_num);
  _local_v_num = _local_v_end - _local_v_beg;

  logger.log("global vertex num: %ld\n", _global_v_num);
  logger.mpi_log("vertex range: [%ld, %ld)\n", _local_v_beg, _local_v_end);
}


tuple<vector<vector<Edge>>, vector<int>>
LocalCSRGraph::DivideEdgeByOwner() {

  int mpi_size = settings.mpi_size;

  // divide raw edges by their own
  vector<vector<Edge>> edges_lists(mpi_size);
  int64_t average = _global_v_num / mpi_size;

  for (int64_t e = 0; e < _local_raw.edge_num; ++e) {
    Edge &edge = _local_raw.edges[e];
    // skip self loop
    if (edge.u != edge.v) {
      int64_t u_owner = mpi_get_owner(edge.u, average),
              v_owner = mpi_get_owner(edge.v, average);

      if (u_owner == v_owner) {
        edges_lists[u_owner].push_back(edge);

      } else {
        edges_lists[u_owner].push_back(edge);
        edges_lists[v_owner].push_back(edge);
      }
    }
  }

  // get the number of edges
  vector<int> scatter_edges_num(mpi_size);
  for (size_t r = 0; r < scatter_edges_num.size(); ++r) {
    scatter_edges_num[r] = edges_lists[r].size();
  }

  return make_tuple(edges_lists, scatter_edges_num);
}


void LocalCSRGraph::SwapEdges() {

  int mpi_size = settings.mpi_size;
  int mpi_rank = settings.mpi_rank;

  vector<vector<Edge>> edges_lists;
  vector<int> scatter_edges_num;
  tie(edges_lists, scatter_edges_num) = DivideEdgeByOwner();

#if 0

  int64_t local_edge_num {0};
  vector<int> gather_count(mpi_size);
  vector<int> gather_offset(mpi_size);

  for (int base = 0; base < mpi_size; ++base) {

    // gather count
    MPI_Gather(&scatter_edges_num[base], 1, MPI_INT, 
        gather_count.data(), 1, MPI_INT, 
        base, MPI_COMM_WORLD);

    for (auto &num : gather_count) {
      local_edge_num += num;
      num *= 2;
    }

    // calc offset of recv buff
    gather_offset[0] = 0;
    for (size_t i = 1; i < gather_offset.size(); ++i) {
      gather_offset[i] = gather_offset[i-1] + gather_count[i-1];
    }

    // reserve memory
    if (base == settings.mpi_rank) {
      _edges.resize(local_edge_num);
      logger.mpi_log("desevered edges number: %ld\n", local_edge_num);
    }

    // gather edges
    MPI_Gatherv(
        edges_lists[base].data(), edges_lists[base].size() * 2, MPI_LONG_LONG,
        _edges.data(), gather_count.data(), gather_offset.data(), MPI_LONG_LONG,
        base, MPI_COMM_WORLD);
  }

#else

  MPI_Allreduce(MPI_IN_PLACE, scatter_edges_num.data(), mpi_size,
      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  logger.mpi_log("desevered edges number: %d\n", scatter_edges_num[mpi_rank]);

  // swap edges one by one
  _edges.resize(scatter_edges_num[mpi_rank]);
  for (int base = 0; base < mpi_size; ++base) {

    if (base == mpi_rank) {
      Edge *offset = _edges.data();
      // copy from local raw edges
      memcpy(offset, edges_lists[base].data(),
          edges_lists[base].size() * sizeof(Edge));
      offset += edges_lists[base].size();
      // logger.mpi_debug("current edge size: %ld\n", offset - _edges.data());

      // base recv from other mpi processes
      for (int i = 0; i < mpi_size; ++i) if (i != mpi_rank) {
        MPI_Status status;
        memset(&status, 0, sizeof(MPI_Status));
        MPI_Probe(i, 0, MPI_COMM_WORLD, &status);

        // TODO: Warning, may overflow
        int recv_count {0};
        MPI_Get_count(&status, MPI_LONG_LONG, &recv_count);

        MPI_Recv(offset, recv_count, MPI_LONG_LONG, i, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        offset += recv_count / 2;
        // logger.mpi_debug("current edge size: %ld\n", offset - _edges.data());
      }

      if (offset - _edges.data() != (ssize_t)_edges.size()) {
        logger.error("the edges size if not corrected! get %ld, wish %zu\n",
            offset - _edges.data(), _edges.size());
      }
    } else {
      // send to base
      MPI_Send(edges_lists[base].data(), edges_lists[base].size() * 2,
          MPI_LONG_LONG, base, 0, MPI_COMM_WORLD);
    }
  }

#endif

  /*
  for (auto &edge : _edges) {
    logger.mpi_debug("edge: u %ld, v %ld\n", edge.u, edge.v);
  }
  */
}


void LocalCSRGraph::ComputeOffset() {
  logger.mpi_log("%s(): \n", __func__);

  _adja_arrays.resize(_local_v_num);
  memset(_adja_arrays.data(), 0, sizeof(AdjacentPair) * _adja_arrays.size());

  // count the number of edges, use "end" temporarily
  for (Edge &edge : _edges) {
    if (_local_v_beg <= edge.u && edge.u < _local_v_end) {
      _adja_arrays[edge.u - _local_v_beg].end += 1;
    }
    if (_local_v_beg <= edge.v && edge.v < _local_v_end) {
      _adja_arrays[edge.v - _local_v_beg].end += 1;
    }
  }

  // compute prefix sum
  _adja_arrays[0].beg = 0;
  for (int64_t v = 1; v < _local_v_num; ++v) {
    _adja_arrays[v].beg  = _adja_arrays[v-1].end;
    _adja_arrays[v].end += _adja_arrays[v-1].end;
  }
  _csr_edge_num = _adja_arrays[_local_v_num-1].end;
  logger.mpi_log("CSR edge num: %d\n", _csr_edge_num);

  /*
  for (int64_t v = 0; v < _local_v_num; ++v) {
    logger.mpi_debug("adj beg %ld, end %d\n",
        _adja_arrays[v].beg, _adja_arrays[v].end);
  }
  */
}


void LocalCSRGraph::ConstructAdjacentArrays() {
  // reset the end of adjacent arrays
  for (int64_t v = 0; v < _local_v_num; ++v) {
    _adja_arrays[v].end = _adja_arrays[v].beg;
  }

  // create edge
  _csr_mem.resize(_csr_edge_num);

  for (Edge &edge : _edges) {
    int64_t local_u = edge.u - _local_v_beg,
            local_v = edge.v - _local_v_beg;
    if (0 <= local_u && local_u < _local_v_num) {
      _csr_mem[ (_adja_arrays[local_u].end)++ ] = edge.v;
    }
    if (0 <= local_v && local_v < _local_v_num) {
      _csr_mem[ (_adja_arrays[local_v].end)++ ] = edge.u;
    }
  }

  // pack
  for (int64_t u = 0; u < _local_v_num; ++u) {
    int64_t beg = adja_beg(u),
            end = adja_end(u);
    if (beg + 1 < end) {
      std::sort(&_csr_mem[beg], &_csr_mem[end]);
      auto end_pointer = std::unique(&_csr_mem[beg], &_csr_mem[end]);
      _adja_arrays[u].end = end_pointer - _csr_mem.data();
    }
  }

  /*
  for (int64_t u = 0; u < _local_v_num; ++u) {
    logger.mpi_debug("u adja beg %ld, end %ld\n", adja_beg(u), adja_end(u));
    for (auto iter = adja_beg(u); iter != adja_end(u); ++iter) {
      logger.mpi_debug("adja: %ld -> %ld\n", u+_local_v_beg, next_vertex(iter));
    }
  } */
}


void LocalCSRGraph::Construct() {
  assert(_local_raw.edges);
  assert(_local_raw.edge_num > 0);
  mpi_log_barrier();
  logger.log("begin constructing csr graph...\n");

  TickOnce tick;

  TickOnce tick_sub;
  GetVertexNumber();
  // logger.mpi_log("GetVertexNumber: TIME %fms\n", tick_sub());

  SwapEdges();
  logger.mpi_log("SwapEdges: TIME %fms\n", tick_sub());

  ComputeOffset();
  // logger.mpi_log("ComputeOffset: TIME %fms\n", tick_sub());

  ConstructAdjacentArrays();
  logger.mpi_log("ConstructAdjacentArrays: TIME %fms\n", tick_sub());

  MPI_Barrier(MPI_COMM_WORLD);
  logger.log("finish constructing csr graph. TIME %fms\n", tick());
}


bool LocalCSRGraph::IsConnect(int64_t global_v) {
  int64_t local_v = global_v - _local_v_beg;
  return adja_beg(local_v) != adja_end(local_v);
}
