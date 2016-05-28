/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-28
 ******************************************************************************/

#pragma once

class CSRGraph {
  private:
    struct AdjacentPair {
      int64_t beg;
      int64_t end;
    };
    AdjacentPair *_adja_arrays {nullptr};
    int64_t *_csr_mem {nullptr};
    int64_t *_csr_head {nullptr};
    int64_t _csr_edge_num {0};
    int64_t _vertex_num {0};

    const Edge *_edges;
    int64_t _edge_desired_num;

  public:
    static const int kAlignment = 64;

    CSRGraph& operator=(CSRGraph &&rhs) {
      _adja_arrays      = rhs._adja_arrays;
      _csr_mem          = rhs._csr_mem;
      _csr_head         = rhs._csr_head;
      _csr_edge_num     = rhs._csr_edge_num;
      _vertex_num       = rhs._vertex_num;
      rhs._adja_arrays  = nullptr;
      rhs._csr_mem      = nullptr;
      rhs._vertex_num   = 0;
      // shared memory
      _edges            = rhs._edges;
      _edge_desired_num = rhs._edge_desired_num;
      return *this;
    }

    CSRGraph(CSRGraph &&rhs) {
      _adja_arrays      = rhs._adja_arrays;
      _csr_mem          = rhs._csr_mem;
      _csr_head         = rhs._csr_head;
      _csr_edge_num     = rhs._csr_edge_num;
      _vertex_num       = rhs._vertex_num;
      rhs._adja_arrays  = nullptr;
      rhs._csr_mem      = nullptr;
      rhs._vertex_num   = 0;
      // shared memory
      _edges            = rhs._edges;
      _edge_desired_num = rhs._edge_desired_num;
    }

    CSRGraph operator=(const CSRGraph &rhs) = delete;
    CSRGraph(const CSRGraph &rhs) = delete;

    CSRGraph(const Edge * edges, int64_t edge_desired_num)
      : _edges(edges), _edge_desired_num(edge_desired_num) {}

    ~CSRGraph() {
      delete []_adja_arrays;
      delete []_csr_mem;
    }

  private:
    void adjacent_link(int64_t u, int64_t v);

    /**
     * @brief   find it because of not knowing how many vertex there;
     *          and allocate memory
     */
    void GetVertexNumber();

    /**
     * @brief   get how many adjacent vertexes of each
     */
    void ScanEdgeArray();

    /**
     * @brief   construct adjacent array
     */
    void ConstructAdjacentArrays();

  public:
    // AdjacentPair* adja_arrays() { return _adja_arrays; }
    int64_t *csr_head() { return _csr_head; }

    /**
     * how to iterate adjacent vertexes?
     * for (auto iter = adja_beg(u); iter != adja_end(u); ++iter) {
     *   int64_t v = next_vertex(u);
     *   // do some work on "v"
     * }
     */
    int64_t adja_beg(int64_t u) { return _adja_arrays[u].beg; }
    int64_t adja_end(int64_t u) { return _adja_arrays[u].end; }
    int64_t next_vertex(int64_t offset) { return _csr_head[offset]; }

    int64_t vertex_num() { return _vertex_num; }
    int64_t edge_desired_num() { return _edge_desired_num; }

    CSRGraph& Construct();
};

CSRGraph
ConstructCSRGraphSingle(const Edge *edges, int64_t edge_desired_num);
