/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-27
 ******************************************************************************/

#include <cassert>
#include <cstring>
#include <algorithm>

#include "utility.h"
#include "construct.h"

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

