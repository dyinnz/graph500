
#pragma once

#include <vector>
#include <tuple>
#include "utility.h"

class Verifier {
  private:
    int64_t *_parents {nullptr};
    int64_t _global_v_num {0};
    int64_t _root {0};
    LocalRawGraph &local_raw;

    int64_t _local_v_num {0};
    int64_t _local_v_beg {0};
    int64_t _local_v_end {0};

    std::vector<int> _levels;

    MPI_Win *_win {nullptr};

  public:
    Verifier(int64_t *parents, int64_t global_v_num, int64_t root, 
        LocalRawGraph &local_raw)
      : _parents(parents), _global_v_num(global_v_num), _root(root),
      local_raw(local_raw) {
        std::tie(_local_v_beg, _local_v_end) = mpi_local_range(global_v_num);
        _local_v_num = _local_v_end - _local_v_beg;
      }

    ~Verifier() {
      if (_win) {
        MPI_Win_free(_win);
        delete [] _win;
      }
    }

    bool CheckRange();
    bool CheckParentOfRoot();
    bool CheckParentOfOthers();
    bool ComputeLevels();

    bool Verify();
};

struct TEST_BFSTree {
  int64_t *parents { nullptr };
  int64_t global_v_num {0};
  int64_t root {0};
  LocalRawGraph raw { nullptr, 0, 0 };

  void Init();

  ~TEST_BFSTree() {
    delete [] parents;
    delete [] raw.edges;
  }
};

