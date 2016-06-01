/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-27
 ******************************************************************************/

#include <cassert>
#include <cstring>
#include <vector>
#include <tuple>
#include <algorithm>

#include "utility.h"
#include "construct.h"

using std::tuple;
using std::make_tuple;
using std::tie;
using std::vector;

void CSRGraph::GetVertexNumber() {
  assert(_edges);
  assert(_edge_desired_num > 0);

  // get vertex number
  int64_t max_vn { -1 };
  _Pragma("omp parallel for reduction (max : max_vn)")
    for (int64_t e = 0; e < _edge_desired_num; ++e) {
      max_vn = std::max(max_vn, _edges[e].u);
      max_vn = std::max(max_vn, _edges[e].v);
    }
  _vertex_num = max_vn + 1;

  // allocate memory
  _adja_arrays = new AdjacentPair[_vertex_num];
  memset(_adja_arrays, 0, sizeof(AdjacentPair) * _vertex_num);

  logger.log("find vertex num: %ld\n", _vertex_num);
}

void CSRGraph::ScanEdgeArray() {
  assert(_vertex_num > 0);

  // use "beg" for counting temporarily
  _Pragma("omp parallel for")
    for (int64_t e = 0; e < _edge_desired_num; ++e) {
      int64_t u = _edges[e].u;
      int64_t v = _edges[e].v;
      // skip self loop
      if (u != v) {
        _Pragma("omp atomic update") ++(_adja_arrays[u].end);
        _Pragma("omp atomic update") ++(_adja_arrays[v].end);
      }
    }

  /*
  for (int64_t v = 0; v < _vertex_num; ++v) {
    printf("v %ld beg %ld, end %ld\n", v, _adja_arrays[v].beg, _adja_arrays[v].end);
  }
  */

  // compute prefix sum
  _adja_arrays[0].beg = 0;
  for (int64_t v = 1; v < _vertex_num; ++v) {
    _adja_arrays[v].beg  = _adja_arrays[v-1].end;
    _adja_arrays[v].end += _adja_arrays[v-1].end;
  }
  _csr_edge_num = _adja_arrays[_vertex_num-1].end;

  logger.log("CSR edge num: %d\n", _csr_edge_num);

  /*
  for (int64_t v = 0; v < _vertex_num; ++v) {
    printf("v %ld beg %ld, end %ld\n", v, _adja_arrays[v].beg, _adja_arrays[v].end);
  }
  */
}

// may be inline
void CSRGraph::adjacent_link(int64_t u, int64_t v) {
  int64_t inserted;
  _Pragma("omp atomic capture") 
    inserted = (_adja_arrays[u].end)++;
  _csr_head[inserted] = v;
}

void CSRGraph::ConstructAdjacentArrays() {
  // allocate csr memory
  // make _csr_head[-1] work
  constexpr int kReserve = kAlignment / sizeof(int64_t);
  _csr_mem = new int64_t[kReserve + _csr_edge_num];
  _csr_head = _csr_mem + kReserve;
  memset(_csr_mem, -1, sizeof(int64_t) * (kReserve + _csr_edge_num));

  // reset the end of adjacent arrays
  _Pragma("omp parallel for")
    for (int64_t v = 0; v < _vertex_num; ++v) {
      _adja_arrays[v].end = _adja_arrays[v].beg;
    }

  _Pragma("omp parallel for")
  for (int64_t e = 0; e < _edge_desired_num; ++e) {
    int64_t u = _edges[e].u;
    int64_t v = _edges[e].v;
    if (u != v) {
      adjacent_link(u, v);
      adjacent_link(v, u);
    }
  }

  /*
  for (int64_t u = 0; u < _vertex_num; ++u) {
    printf("u %ld beg %ld, end %ld\n", u, _adja_arrays[u].beg, _adja_arrays[u].end);
    for (auto iter = adja_beg(u); iter != adja_end(u); ++iter) {
      int64_t v = next_vertex(iter);
      printf("%ld ", v);
    }
    printf("\n");
  }
  */

  // pack
  _Pragma("omp parallel for")
    for (int64_t v = 0; v < _vertex_num; ++v) {
      int64_t beg = _adja_arrays[v].beg;
      int64_t end = _adja_arrays[v].end;
      if (beg + 1 < end) {
        std::sort(_csr_head+beg, _csr_head+end);
        auto end_pointer = std::unique(_csr_head+beg, _csr_head+end);
        _adja_arrays[v].end = end_pointer - _csr_head;
      }
    }

  /*
  for (int64_t u = 0; u < _vertex_num; ++u) {
    printf("u %ld beg %ld, end %ld\n", u, _adja_arrays[u].beg, _adja_arrays[u].end);
    for (auto iter = adja_beg(u); iter != adja_end(u); ++iter) {
      int64_t v = next_vertex(iter);
      printf("%ld ", v);
    }
    printf("\n");
  }
  */
}

CSRGraph& CSRGraph::Construct() {
  logger.log("begin constructing csr graph...\n");
  GetVertexNumber();
  ScanEdgeArray();
  ConstructAdjacentArrays();
  logger.log("finishing constructing csr graph.\n");
  return *this;
}

/*----------------------------------------------------------------------------*/

void LocalCSRGraph::GetVertexNumber() {
  int64_t max_vn { -1 };
  for (int64_t e = 0; e < _local_raw.edge_num; ++e) {
    max_vn = std::max(max_vn, raw_edge_u(e));
    max_vn = std::max(max_vn, raw_edge_v(e));
  }
  MPI_Allreduce(MPI_IN_PLACE, &max_vn, 1, MPI_LONG_LONG, MPI_MAX, 
      MPI_COMM_WORLD);
  _global_v_num = max_vn + 1;

  tie(_local_v_beg, _local_v_end) = mpi_local_range(_global_v_num);
  _local_v_num = _local_v_end - _local_v_beg;

  logger.log("global vertex num: %ld\n", _global_v_num);
  logger.mpi_log("vertex range: [%ld, %ld)\n", _local_v_beg, _local_v_end);
}

#if 0
void LocalCSRGraph::CountScatteredAdjacentSize(int64_t *adja_size) {
  logger.mpi_log("%s\n", __func__);
  memset(adja_size, 0, sizeof(int64_t) * _global_v_num);
  // compute the size of adjacent arrays
  for (int64_t e = 0; e < _local_raw.edge_num; ++e) {
    if (edge_u(e) != edge_v(e)) {
      adja_size[edge_u(e)] += 1;
      adja_size[edge_v(e)] += 1;
    }
  }
}

tuple<LocalCSRGraph::AdjacentPair *, int64_t *>
LocalCSRGraph::BuildScatteredCSR(const int64_t *adja_size) {
  logger.mpi_log("%s\n", __func__);
  // compute the prefix
  auto scatter_adja = new AdjacentPair[_global_v_num];
  scatter_adja[0].beg = 0;
  for (int64_t v = 1; v < _global_v_num; ++v) {
    scatter_adja[v].beg = scatter_adja[v-1].beg + adja_size[v-1];
  }
  // reset the end position
  for (int64_t v = 0; v < _global_v_num; ++v) {
    scatter_adja[v].end = scatter_adja[v].beg;
  }

  /*
  for (int64_t v = 0; v < _global_v_num; ++v) {
    logger.mpi_log("v %ld: beg %ld\n", v, scatter_adja[v].beg);
  }
  */

  int64_t scatter_csr_size = 
    scatter_adja[_global_v_num-1].beg + adja_size[_global_v_num-1];
  logger.mpi_log("scatter_csr_size: %ld\n", scatter_csr_size);
  auto *scatter_csr_mem = new int64_t[scatter_csr_size];
  memset(scatter_csr_mem, 0, sizeof(int64_t) * scatter_csr_size);

  logger.mpi_log("build scattered csr...\n");

  // build the scattered csr
  for (int64_t e = 0; e < _local_raw.edge_num; ++e) {
    int64_t u = edge_u(e);
    int64_t v = edge_v(e);
    if (u != v) {
      /*
      logger.mpi_log(" u %ld, beg %ld, end %ld\n", 
          u, scatter_adja[u].beg, scatter_adja[v].end);
          */
      scatter_csr_mem[(scatter_adja[u].end)++] = v;
      scatter_csr_mem[(scatter_adja[v].end)++] = u;
    }
  }

  return make_tuple(scatter_adja, scatter_csr_mem);
}

void LocalCSRGraph::MergeAdjacentSize(int64_t *adja_size) {
  logger.mpi_log("%s\n", __func__);
  MPI_Allreduce(MPI_IN_PLACE, adja_size, _global_v_num, MPI_LONG_LONG,
      MPI_SUM, MPI_COMM_WORLD);
}

void LocalCSRGraph::ComputeOffset(const int64_t *adja_size) {
  logger.mpi_log("%s\n", __func__);
  _adja_arrays = new AdjacentPair[_local_v_num];
  memset(_adja_arrays, -1, sizeof(AdjacentPair) * _local_v_num);

  _adja_arrays[0].beg = 0;
  _adja_arrays[0].end = 0 + adja_size[_local_v_beg + 0];
  for (int64_t v = 1; v < _local_v_num; ++v) {
    _adja_arrays[v].beg = _adja_arrays[v-1].end;
    _adja_arrays[v].end = _adja_arrays[v].beg + adja_size[_local_v_beg + v];
  }
  _csr_edge_num = _adja_arrays[_local_v_num-1].end;
  logger.log("CSR edge num: %d\n", _csr_edge_num);

  /*
  for (int64_t v = 0; v < _local_v_num; ++v) {
    logger.mpi_debug("adj beg %ld, end %d\n", 
        _adja_arrays[v].beg, _adja_arrays[v].end);
  }
  */
}

void LocalCSRGraph::GatherEdges(const int64_t *adja_size,
    AdjacentPair *scatter_adja,
    int64_t *scatter_csr) {
  logger.mpi_log("%s\n", __func__);

  _csr_head = new int64_t[_csr_edge_num];
  memset(_csr_head, 0, sizeof(int64_t) * _csr_edge_num);

  int64_t average = _global_v_num / settings.mpi_size;

  for (int64_t v = 0; v < _global_v_num; ++v) {
    int64_t receiver = mpi_get_owner(v, average);
    int64_t local_v = v - _local_v_beg;
    // TODO:
  }
  logger.mpi_log("%s() finishing.. \n", __func__);
}

void LocalCSRGraph::Construct() {
  assert(_local_raw.edges);
  assert(_local_raw.edge_num > 0);

  MPI_Barrier(MPI_COMM_WORLD);
  logger.log("begin constructing csr graph...\n");

  GetVertexNumber();

  auto adja_size = new int64_t[_global_v_num];
  CountScatteredAdjacentSize(adja_size);

  AdjacentPair *scatter_adja {nullptr};
  int64_t *scatter_csr {nullptr};
  tie(scatter_adja, scatter_csr) = BuildScatteredCSR(adja_size);

  MergeAdjacentSize(adja_size);
  ComputeOffset(adja_size);
  GatherEdges(adja_size, scatter_adja, scatter_csr);

  MPI_Barrier(MPI_COMM_WORLD);
  if (0 == settings.mpi_rank) {
    logger.mpi_debug("\n");
    for (int64_t u = 0; u < _local_v_num; ++u) {
      fprintf(stderr, "u %ld 's son\n", u);
      for (auto iter = adja_beg(u); iter != adja_end(u); ++iter) {
        fprintf(stderr, "%ld ", next_vertex(iter));
      }
      fprintf(stderr, "\n");
    }
  }

  delete [] adja_size;
  delete [] scatter_adja;
  delete [] scatter_csr;

  MPI_Barrier(MPI_COMM_WORLD);
  logger.log("finishing constructing csr graph.\n");
}
#endif

void LocalCSRGraph::SwapEdges() {
  uint64_t mpi_rank = settings.mpi_rank;
  uint64_t mpi_size = settings.mpi_size;

  // split raw edges
  vector<vector<Edge>> edges_lists(mpi_size);
  int64_t average = _global_v_num / mpi_size;

  logger.mpi_debug("%s(): average: %ld\n", __func__, average);

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
  vector<int64_t> edges_numbers(mpi_size);
  for (size_t r = 0; r < edges_numbers.size(); ++r) {
    edges_numbers[r] = edges_lists[r].size();
  }
  MPI_Allreduce(MPI_IN_PLACE, edges_numbers.data(), mpi_size,
      MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  static_assert(sizeof(size_t) == sizeof(int64_t), 
      "the size of size_t and int64_t must be same");

  logger.mpi_log("desevered edges number: %ld\n", edges_numbers[mpi_rank]);

  // swap edges one by one
  _edges.resize(edges_numbers[mpi_rank]);
  Edge *offset = _edges.data();
  MPI_Status status;
  memset(&status, 0, sizeof(MPI_Status));
  // TODO: Warning, may overflow
  int recv_count {0};
  for (size_t base = 0; base < mpi_size; ++base) {
    if (base == mpi_rank) {
      memcpy(offset, edges_lists[base].data(), 
          edges_lists[base].size() * sizeof(Edge));
      offset += edges_lists[base].size();
      // logger.mpi_debug("current edge size: %ld\n", offset - _edges.data());

      // base recv from other mpi processes
      for (size_t i = 0; i < mpi_size; ++i) if (i != mpi_rank) {
        MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_LONG_LONG, &recv_count);
        MPI_Recv(offset, recv_count, MPI_LONG_LONG, i, 0, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        offset += recv_count / 2;
        // logger.mpi_debug("current edge size: %ld\n", offset - _edges.data());
      }

    } else {
      // send to base
      MPI_Send(edges_lists[base].data(), edges_lists[base].size() * 2, 
          MPI_LONG_LONG, base, 0, MPI_COMM_WORLD);
    }
  }

  if (offset - _edges.data() != _edges.size()) {
    logger.error("the edges size if not corrected! get %ld, wish %zu\n",
        offset - _edges.data(), _edges.size());
  }

  for (auto &edge : _edges) {
    logger.mpi_debug("edge: u %ld, v %ld\n", edge.u, edge.v);
  }
}

void LocalCSRGraph::Construct() {
  assert(_local_raw.edges);
  assert(_local_raw.edge_num > 0);
  MPI_Barrier(MPI_COMM_WORLD);
  logger.log("begin constructing csr graph...\n");

  GetVertexNumber();

  SwapEdges();

  MPI_Barrier(MPI_COMM_WORLD);
  logger.log("finish constructing csr graph.\n");
}
