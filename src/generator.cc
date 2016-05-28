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
 * @brief     kronecker graph generator
 * @return    the edges array
 */
Edge*
GeneratorGraph(int64_t vertex_num, int64_t edge_desired_num) {
  logger.log("begin generating edges array of graph...\n");
  Edge *edges = new Edge[edge_desired_num];

  auto U = new double[edge_desired_num];
  auto V = new double[edge_desired_num];

  // kronecker generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0f, 1.0f);
  for (int i = 0; i < settings.scale; ++i) {
    for (int64_t e = 0; e < edge_desired_num; ++e) {
      double beg_bit = (dis(gen) > kAB) ? 1.0f : 0.0f;
      double end_bit = ( dis(gen) > (kCNorm*beg_bit + kANorm*(1.0f - end_bit)) ) ?
                    1.0f : 0.0f;
      U[e] += (1 << i) * beg_bit;
      V[e] += (1 << i) * end_bit;
    }
  }

  // shuffle vertex
  auto permut_vertex = new int64_t[vertex_num];
  for (int64_t i = 0; i < vertex_num; ++i) {
    permut_vertex[i] = i;
  }
  std::random_shuffle(permut_vertex, permut_vertex+vertex_num);
  for (int64_t e = 0; e < edge_desired_num; ++e) {
    U[e] = permut_vertex[int64_t(U[e])];
    V[e] = permut_vertex[int64_t(V[e])];
  }
  delete []permut_vertex;

  // TODO: could use random_shuffle() directly to do this work?
  // shuffle edges
  auto permut_edge = new int64_t[edge_desired_num];
  for (int64_t e = 0; e < edge_desired_num; ++e) {
    permut_edge[e] = e;
  }
  std::random_shuffle(permut_edge, permut_edge+edge_desired_num);
  for (int64_t e = 0; e < edge_desired_num; ++e) {
    edges[e].u = (int64_t)U[permut_edge[e]];
    edges[e].v = (int64_t)V[permut_edge[e]];
  }
  delete permut_edge;

  delete []U;
  delete []V;

  /*
  for (int64_t e = 0; e < edge_desired_num; ++e) {
    printf("e %ld beg %ld, end %ld\n", e, edges[e].u, edges[e].v);
  }
  */

  logger.log("finish generating graph...\n");
  return edges;
}

