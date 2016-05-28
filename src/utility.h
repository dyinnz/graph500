/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-27
 ******************************************************************************/

#pragma once

#include <tuple>

#include "simplelogger.h"

struct Settings {
  int scale {5};
  int edge_factor {16};
  int64_t vertex_num {0};
  int64_t edge_desired_num {0};
};

struct Edge {
  int64_t u;
  int64_t v;
};

extern dy_logger::Logger logger;
extern Settings settings;

