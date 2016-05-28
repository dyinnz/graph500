/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-27
 ******************************************************************************/

#pragma once

#include <functional>

#include "simplelogger.h"

struct Settings {
  int scale {5};
  int edge_factor {16};
  int sample_num {64};
  int64_t vertex_num {0};
  int64_t edge_desired_num {0};
};

struct Edge {
  int64_t u;
  int64_t v;
};

extern dy_logger::Logger logger;
extern Settings settings;

class ScopeGuarder {
  public:
    ScopeGuarder(std::function<void()> guard) : _guard(guard) {}
    ~ScopeGuarder() { _guard();  }

  private:
    std::function<void()> _guard;
    ScopeGuarder& operator=(const ScopeGuarder&) = delete;
    ScopeGuarder& operator=(ScopeGuarder&&) = delete;
    ScopeGuarder(const ScopeGuarder&) = delete;
    ScopeGuarder(ScopeGuarder&&) = delete;
};

#define ScopeGuard(F) ScopeGuarder __FILE__##__LINE__##ScopeGuarder(F)

