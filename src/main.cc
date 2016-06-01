/******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-27
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <random>

#include "unistd.h"

#include "utility.h"
#include "construct.h"

using namespace dy_logger;
using std::vector;

Logger logger;
Settings settings;

LocalRawGraph MPIGenerateGraph(int64_t vertex_num, int64_t edge_desired_num);
bool VerifyBFSTree(int64_t *bfs_tree,
                   int64_t vertex_num,
                   int64_t root,
                   Edge *edges,
                   int64_t edge_desired_num);

/*----------------------------------------------------------------------------*/

/**
 * @brief   parse input parameter, print them.
 */
static void
ParseParameters(int argc, char * const *argv) {
  int opt {0};
  while (-1 != (opt = getopt(argc, argv, "s:e:"))) {
    switch (opt) {
      case 's':
        settings.scale = atoi(optarg); break;

      case 'e':
        settings.edge_factor = atoi(optarg); break;

      default:
        logger.log("unrecognized parameters.");
        break;
    }
  }

  logger.log("Scale       : %d\n", settings.scale);
  logger.log("Edge factor : %d\n", settings.edge_factor);
}

/**
 * @brief   initialize other settings and some resources
 */
static void
Initialize() {
  settings.vertex_num = 1LL << settings.scale;
  settings.edge_desired_num = settings.edge_factor * settings.vertex_num;

  logger.log("Total vertexes      : %d\n", settings.vertex_num);
  logger.log("Total desired edges : %d\n", settings.edge_desired_num);
}

#if 0
static vector<int64_t>
SampleKeys(CSRGraph &csr) {
  logger.log("begin sampling %d keys...\n", settings.sample_num);

  vector<int64_t> roots;
  vector<int64_t> connected;
  roots.reserve(settings.sample_num);
  connected.reserve(settings.vertex_num);

  // check connection
  for (int64_t u = 0; u < settings.vertex_num; ++u) {
    if (csr.adja_beg(u) < csr.adja_end(u)) {
      connected.push_back(u);
    }
  }

  // random select
  std::random_device rd;
  std::mt19937_64 rand_gen(rd());
  int remain = settings.sample_num;
  while (remain > 0 && !connected.empty()) {
    int64_t index = rand_gen() % connected.size();
    roots.push_back(connected[index]);
    remain -= 1;

    connected[index] = connected.back();
    connected.pop_back();
  }

  logger.log("finishing smapling: %d keys.\n", roots.size());

  /*
  for (int64_t u : roots) {
    printf("%ld ", u);
  }
  printf("\n");
  */

  return roots;
}

#endif

int
main(int argc, char *argv[]) {

  // init mpi
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &settings.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &settings.mpi_size);
  fprintf(stderr, "--- MPI world: rank %d, size %d ---\n", 
      settings.mpi_rank, settings.mpi_size);

  logger.set_mpi_rank(settings.mpi_rank);
 #ifdef DEBUG
  logger.set_filter_level(Logger::kDebug);
  logger.debug("graph500 run in debug mode.\n");
#endif

  ParseParameters(argc, argv);
  Initialize();

  LocalRawGraph local_raw = MPIGenerateGraph(settings.vertex_num, 
                                             settings.edge_desired_num);

  LocalCSRGraph local_csr(local_raw);

  local_csr.Construct();

  delete [] local_raw.edges;
  local_raw.edges = nullptr;

  mpi_log_barrier();
  logger.log("graph500 exit!\n");
  MPI_Finalize();
  return 0;
}

