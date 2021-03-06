/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-27
 ******************************************************************************/

#include <cstring>
#include <tuple>
#include <random>
#include <algorithm>

#include "utility.h"
#include "prng_engine.hpp"

using std::tuple;
using std::make_tuple;
using std::tie;

constexpr double kGenParaA {0.57};
constexpr double kGenParaB {0.19};
constexpr double kGenParaC {0.19};

constexpr double kAB { kGenParaA+kGenParaB };
constexpr double kCNorm { kGenParaC / (1.0f - kGenParaA - kGenParaB) };
constexpr double kANorm { kGenParaA / (kGenParaA + kGenParaB) };


/**
 * Generate the raw edge tuples parallely  which need shuffling later,
 * using Kronecker Algorithm to generate.
 */
static void
GenerateEdgeTuples(int64_t mpi_rank,
                   int64_t scale,
                   double *U, double *V,
                   int64_t local_edge_num) {
  // kronecker generator
  sitmo::prng_engine gen(settings.mpi_rank);
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  for (int i = 0; i < scale; ++i) {
    for (int64_t e = 0; e < local_edge_num; ++e) {
      double beg_bit = (dis(gen) > kAB) ? 1.0 : 0.0;
      double end_bit = (dis(gen) > (kCNorm*beg_bit + kANorm*(1.0 - beg_bit))) ?
                    1.0 : 0.0;
      U[e] += (double)(1LL << i) * beg_bit;
      V[e] += (double)(1LL << i) * end_bit;
    }
  }
  /*
  for (int64_t e = 0; e < local_edge_num; ++e) {
    logger.mpi_debug("U %lf, V %lf\n", U[e], V[e]);
  }
  */
}


/**
 * Shuffle their own vertexes by substituting; do not have MPI communication
 */
static void
ShuffleVertexes(int64_t mpi_rank,
                double *U, double *V,
                int64_t local_edge_num,
                int64_t vertex_num) {

  // first, generate a permutation array which the size of is V
  std::vector<int64_t> permut_vertex(vertex_num);
  for (int64_t i = 0; i < vertex_num; ++i) {
    permut_vertex[i] = i;
  }
  std::mt19937_64 gen;
  std::shuffle(permut_vertex.begin(), permut_vertex.end(), gen);
  /*
  for (int64_t i = 0; i < vertex_num; ++i) {
    std::swap(permut_vertex[i], permut_vertex[gen()%(vertex_num-i)+i]);
  }
  */
  /*
  for (int64_t i = 0; i < vertex_num; ++i) {
    logger.debug("%d\n", permut_vertex[i]);
  }
  */

  // use the permutation to shuffle
  for (int64_t e = 0; e < local_edge_num; ++e) {
    U[e] = permut_vertex[int64_t(U[e])];
    V[e] = permut_vertex[int64_t(V[e])];
  }

  /*
  for (int64_t e = 0; e < local_edge_num; ++e) {
    logger.mpi_debug("shuffle U %lf, V %lf\n", U[e], V[e]);
  }
  */
}


/**
 * Shuffle the edge by swapping with others. There may be a lot of MPI
 * communication, bacause of swap discrete data with remote mpi processes.
 */
static void
ShuffleEdges(int64_t mpi_rank, double *U, double *V, int64_t edge_desired_num) {

  int64_t least_e_num = edge_desired_num / settings.mpi_size;
  int64_t e_beg = least_e_num * settings.mpi_rank;

  std::mt19937_64 gen;
  for (int64_t e = 0; e < edge_desired_num; ++e) {
    int64_t i = gen() % (edge_desired_num-e) + e;
    int64_t e_own = mpi_get_owner(e, least_e_num);
    int64_t i_own = mpi_get_owner(i, least_e_num);

    //logger.mpi_debug("e %ld, i %ld, e_own %ld, i_own %ld\n",
        //e, i, e_own, i_own);
    //MPI_Barrier(MPI_COMM_WORLD);

    if (e_own == i_own && i_own == mpi_rank)  {
      // local swap
      std::swap(U[e - e_beg], U[i - e_beg]);
      std::swap(V[e - e_beg], V[i - e_beg]);

    } else if (mpi_rank == e_own) {
      // swap with remote MPI process
      MPI_Sendrecv_replace(&U[e-e_beg], 1, MPI_DOUBLE, i_own, 0, i_own, 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv_replace(&V[e-e_beg], 1, MPI_DOUBLE, i_own, 0, i_own, 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } else if (mpi_rank == i_own) {
      // swap with remote MPI process
      MPI_Sendrecv_replace(&U[i-e_beg], 1, MPI_DOUBLE, e_own, 0, e_own, 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv_replace(&V[i-e_beg], 1, MPI_DOUBLE, e_own, 0, e_own, 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}


/**
 * Convert double type to int64_t
 */
static void
LoadEdges(double *U, double *V, Edge *edges, int64_t local_edge_num) {
  for (int64_t e = 0; e < local_edge_num; ++e) {
    edges[e].u = int64_t(U[e]);
    edges[e].v = int64_t(V[e]);
  }
}


/**
 * Genereate graph edges parallelly.
 */
LocalRawGraph
MPIGenerateGraph(int64_t vertex_num, int64_t edge_desired_num) {
  mpi_log_barrier();
  logger.log("begin generating edges array of graph...\n");

  TickOnce tick;

  LocalRawGraph local_raw;

  local_raw.global_edge_num = edge_desired_num;

  local_raw.edge_num = edge_desired_num / settings.mpi_size;
  if (mpi_is_last_rank()) {
    local_raw.edge_num += edge_desired_num % settings.mpi_size;
  }
  logger.mpi_log("local raw edge num: %ld\n", local_raw.edge_num);

  // allocate double memory to run Kronecker Alogrithm.
  auto U = new double[local_raw.edge_num];
  auto V = new double[local_raw.edge_num];
  memset(U, 0, sizeof(double) * local_raw.edge_num);
  memset(V, 0, sizeof(double) * local_raw.edge_num);

  TickOnce tick_sub;

  GenerateEdgeTuples(settings.mpi_rank, settings.scale,
      U, V, local_raw.edge_num);
  logger.mpi_log("GenerateEdgeTuples: TIME %fms\n", tick_sub());

  ShuffleVertexes(settings.mpi_rank, U, V, local_raw.edge_num, vertex_num);
  // logger.mpi_log("ShuffleVertexes: TIME %fms\n", tick_sub());

  if (settings.is_shuffle_edges) {
    ShuffleEdges(settings.mpi_rank, U, V, edge_desired_num);
    logger.mpi_log("ShuffleEdges: TIME %fms\n", tick_sub());
  } else {
  }

  local_raw.edges = new Edge[local_raw.edge_num];
  LoadEdges(U, V, local_raw.edges, local_raw.edge_num);

  for (int64_t e = 0; e < local_raw.edge_num; ++e) {
    logger.mpi_debug("edges: %ld\t%ld\n",
        local_raw.edges[e].u, local_raw.edges[e].v);
  }

  delete []U;
  delete []V;

  MPI_Barrier(MPI_COMM_WORLD);
  logger.log("finish generating graph. TIME %fms\n", tick());
  return local_raw;
}

