/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-28
 ******************************************************************************/

#pragma once

#include "utility.h"

class LocalCSRGraph {
  private:
    LocalRawGraph &_local_raw;
    std::vector<Edge> _edges;

    int64_t _local_v_num {0};
    int64_t _global_v_num {0};
    int64_t _local_v_beg {0};
    int64_t _local_v_end {0};

    std::vector<int64_t> _csr_mem;
    int64_t _csr_edge_num {0};

    struct AdjacentPair {
      int64_t beg;
      int64_t end;
    };
    std::vector<AdjacentPair> _adja_arrays;

  private:
    int64_t raw_edge_u(int64_t e) { return _local_raw.edges[e].u; }
    int64_t raw_edge_v(int64_t e) { return _local_raw.edges[e].v; }

    int64_t edge_u(int64_t e) { return _edges[e].u; }
    int64_t edge_v(int64_t e) { return _edges[e].v; }

    void GetVertexNumber();
    void CountScatteredAdjacentSize(int64_t *adja_size);
    std::tuple<AdjacentPair *, int64_t *>
      BuildScatteredCSR(const int64_t *adja_size);
    void MergeAdjacentSize(int64_t *adja_size);
    void ComputeOffset(const int64_t *adja_size);
    void ComputeOffset();
    void GatherEdges(const int64_t *adja_size,
                     AdjacentPair *scatter_adja,
                     int64_t *scatter_csr);

    void SwapEdges();
    void ConstructAdjacentArrays();

  public:
    LocalCSRGraph(LocalRawGraph &local_raw) : _local_raw(local_raw) {}

    int64_t* csr_mem() { return _csr_mem.data(); }
    AdjacentPair* adja_arrays() { return _adja_arrays.data(); }

    int64_t global_v_num() { return _global_v_num; }
    int64_t local_v_num() { return _local_v_num; }

    int64_t adja_beg(int64_t u) { return _adja_arrays[u].beg; }
    int64_t adja_end(int64_t u) { return _adja_arrays[u].end; }
    int64_t next_vertex(int64_t offset) { return _csr_mem[offset]; }

    bool IsConnect(int64_t global_v);

    void Construct();
};


