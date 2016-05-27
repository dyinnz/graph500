/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-27
 ******************************************************************************/

#include <random>
#include <algorithm>

#include "utility.h"

using std::tuple;
using std::make_tuple;

constexpr double kGenParaA {0.57};
constexpr double kGenParaB {0.19};
constexpr double kGenParaC {0.19};

constexpr double kAB { kGenParaA+kGenParaB };
constexpr double kCNorm { kGenParaC / (1.0f - kGenParaA - kGenParaB) };
constexpr double kANorm { kGenParaA / (kGenParaA + kGenParaB) };

/**
 * @return    Edge*
 */
Edge*
GeneratorGraph(uint64_t vertex_num, uint64_t edge_desired_num) {
  logger.log("Begin generate edges array of graph...\n");
  Edge *edges = new Edge[edge_desired_num];

  auto beg = new double[edge_desired_num];
  auto end = new double[edge_desired_num];

  // kronecker generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0f, 1.0f);
  for (int i = 0; i < settings.scale; ++i) {
    for (uint64_t e = 0; e < edge_desired_num; ++e) {
      double beg_bit = (dis(gen) > kAB) ? 1.0f : 0.0f;
      double end_bit = ( dis(gen) > (kCNorm*beg_bit + kANorm*(1.0f - end_bit)) ) ?
                    1.0f : 0.0f;
      beg[e] += (1 << i) * beg_bit;
      end[e] += (1 << i) * end_bit;
    }
  }

  // shuffle vertex
  auto permut_vertex = new uint64_t[vertex_num];
  for (uint64_t i = 0; i < vertex_num; ++i) {
    permut_vertex[i] = i;
  }
  std::random_shuffle(permut_vertex, permut_vertex+vertex_num);
  for (uint64_t e = 0; e < edge_desired_num; ++e) {
    beg[e] = permut_vertex[uint64_t(beg[e])];
    end[e] = permut_vertex[uint64_t(end[e])];
  }
  delete []permut_vertex;

  /*
  for (uint64_t e = 0; e < edge_desired_num; ++e) {
    printf("e %lu beg %f, end %f\n", e, beg[e], end[e]);
  }
  */

  // shuffle
  auto permut_edge = new uint64_t[edge_desired_num];
  for (uint64_t e = 0; e < edge_desired_num; ++e) {
    permut_edge[e] = e;
  }
  std::random_shuffle(permut_edge, permut_edge+edge_desired_num);
  for (uint64_t e = 0; e < edge_desired_num; ++e) {
    edges[e].beg = (uint64_t)beg[permut_edge[e]];
    edges[e].end = (uint64_t)end[permut_edge[e]];
  }
  delete permut_edge;

  delete []beg;
  delete []end;

  /*
  for (uint64_t e = 0; e < edge_desired_num; ++e) {
    printf("e %lu beg %lu, end %lu\n", e, edges[e].beg, edges[e].end);
  }
  */

  logger.log("Finish generating graph...\n");
  return edges;
}

