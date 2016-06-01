/******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-27
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>

#include "unistd.h"

#include "utility.h"
#include "construct.h"

using namespace dy_logger;
using std::vector;

Logger logger;
Settings settings;

LocalRawGraph MPIGenerateGraph(int64_t vertex_num, int64_t edge_desired_num);
int64_t* BuildBFSTree(LocalCSRGraph &local_csr, int64_t root);
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

static bool
CheckConnection(LocalCSRGraph &local_csr, int64_t index) {
  bool is_connect {false};

  int64_t index_owner = mpi_get_owner(index,
      settings.vertex_num / settings.mpi_size);
  if (settings.mpi_rank == index_owner) {
    is_connect = local_csr.IsConnect(index);
  }
  MPI_Allreduce(MPI_IN_PLACE, &is_connect, 1, MPI_CHAR, MPI_BOR,
      MPI_COMM_WORLD);
  return is_connect;
}

static vector<int64_t>
SampleKeys(LocalCSRGraph &local_csr) {
  mpi_log_barrier();
  logger.log("begin sampling %d keys...\n", settings.sample_num);

  vector<int64_t> roots;
  roots.reserve(settings.sample_num);
  std::unordered_map<int64_t, int64_t> swap_map;

  std::random_device rd;
  std::mt19937_64 rand_gen(rd());
  int64_t remain_vertex_num { settings.vertex_num };
  while (roots.size() < settings.sample_num && remain_vertex_num > 0) {

    // generate a random index
    int64_t index = rand_gen() % remain_vertex_num;
    MPI_Bcast(&index, 1, MPI_LONG_LONG, 0/*root*/, MPI_COMM_WORLD);
    // logger.debug("rand select index %ld\n", index);

    if (CheckConnection(local_csr, index)) {
      int64_t real_index = swap_map.find(index) == swap_map.end() ?
        index : swap_map[index];
      roots.push_back(real_index);

      // logger.debug("%ld map to %ld connect!\n", index, real_index);
    } else {
      // logger.debug("%ld not connect!\n", index);
    }

    // mark this index as invalid
    remain_vertex_num -= 1;
    swap_map[index] = remain_vertex_num;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  logger.log("sample %zu keys\n", roots.size());
  return roots;
}

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
  vector<int64_t> roots = SampleKeys(local_csr);
  for (auto root : roots) {
    // Run BFS here
    int64_t *bfs_tree = BuildBFSTree(local_csr, root);
    delete []bfs_tree;

  #ifdef DEBUG
    break;
  #endif
  }

  delete [] local_raw.edges;
  local_raw.edges = nullptr;

  mpi_log_barrier();
  logger.log("graph500 exit!\n");
  MPI_Finalize();
  return 0;
}

