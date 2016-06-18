/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-27
 ******************************************************************************/

#pragma once

#include <functional>
#include <tuple>
#include <string>
#include <chrono>

#include "mpi.h"

#include "simplelogger.h"

struct Settings {
  int mpi_rank {-1};
  int mpi_size {-1};
  int scale {5};
  int edge_factor {16};
  int sample_num {64};
  int64_t vertex_num {0};
  int64_t edge_desired_num {0};
  int64_t least_v_num {0};
  bool is_debug {false};
  bool is_verify {true};
  bool is_shuffle_edges {true};
  std::string file_in;
  std::string file_out;
};

struct Edge {
  int64_t u;
  int64_t v;
};

struct LocalRawGraph {
  Edge *edges;
  int64_t edge_num;
  int64_t global_edge_num;
};

typedef int32_t bit_type;
constexpr int kBitWidth {sizeof(bit_type) * 8};

extern dy_logger::Logger logger;
extern Settings settings;

class ScopeGuarder {
  public:
    ScopeGuarder(std::function<void()> &guard) : _guard(guard) {}
    ~ScopeGuarder() { _guard();  }

  private:
    std::function<void()> &_guard;
    ScopeGuarder& operator=(const ScopeGuarder&) = delete;
    ScopeGuarder& operator=(ScopeGuarder&&) = delete;
    ScopeGuarder(const ScopeGuarder&) = delete;
    ScopeGuarder(ScopeGuarder&&) = delete;
};

#define ScopeGuard(F) ScopeGuarder __FILE__##__LINE__##ScopeGuarder(F)


class TickOnce {
  public:
    TickOnce() : _last(std::chrono::system_clock::now()) {}

    float operator() () {
      auto ret = std::chrono::system_clock::now() - _last;
      _last = std::chrono::system_clock::now();
      return ret.count() / 1000000.0;
    }

  private:
    std::chrono::system_clock::time_point _last;
};


inline bool 
mpi_is_last_rank() {
  return settings.mpi_rank == settings.mpi_size - 1;
}


inline int64_t
mpi_get_owner(int64_t index, int64_t least) {
   return std::min(index/least, int64_t(settings.mpi_size-1));
}


inline void
mpi_log_barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
  fflush(stderr);
  fflush(stdout);
  logger.log("------------------------ MPI Barrier ------------------------\n");
}
