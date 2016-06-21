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
using std::string;

Logger logger;
Settings settings;
vector<float> g_bfs_time;
vector<float> g_teps;

LocalRawGraph MPIGenerateGraph(int64_t vertex_num, int64_t edge_desired_num);
int64_t* BuildBFSTree(LocalCSRGraph &local_csr, int64_t root);
bool VerifyBFSTree(int64_t *parents, int64_t global_v_num, int64_t root,
    LocalRawGraph &local_raw);


void DumpLocalRawGraph(const string &filename, LocalRawGraph &local_raw);
LocalRawGraph LoadLocalRawGraph(const string &filename);


/*----------------------------------------------------------------------------*/


/**
 * @brief   parse input parameter, print them.
 */
static void
ParseParameters(int argc, char * const *argv) {

  auto parse_bool = [](char * const p, bool &b) {
    if ('0' == *p || 'F' == *p || 'f' == *p) {
      b = false;
    } else if ('1' == *p || 'T' == *p || 't' == *p) {
      b = true;
    } else {
      logger.mpi_error("Incorrect bool parameter: %s\n", p);
    }
  };

  int opt {0};
  while (-1 != (opt = getopt(argc, argv, "s:e:d:v:f:i:o:"))) {
    switch (opt) {
      case 's':
        settings.scale = atoi(optarg); break;

      case 'e':
        settings.edge_factor = atoi(optarg); break;

      case 'd':
        parse_bool(optarg, settings.is_debug); break;

      case 'v':
        parse_bool(optarg, settings.is_verify); break;

      case 'f':
        parse_bool(optarg, settings.is_shuffle_edges); break;

      case 'i':
        settings.file_in = optarg; break;

      case 'o':
        if (!settings.file_in.empty()) {
          logger.mpi_error(
              "input date filename have be set! invalide parameters\n");
        } else {
          settings.file_out = optarg;
        }
        break;

      default:
        logger.log("unrecognized parameters.");
        break;
    }
  }

  logger.log("Scale         : %d\n", settings.scale);
  logger.log("Edge factor   : %d\n", settings.edge_factor);
  logger.log("Debug mode    : %d\n", settings.is_debug);
  logger.log("Validation    : %d\n", settings.is_verify);
  logger.log("Shuffle edges : %d\n", settings.is_shuffle_edges);
  logger.log("Input file    : %s\n", settings.file_in.c_str());
  logger.log("Output file   : %s\n", settings.file_out.c_str());
}


/**
 * @brief   initialize other settings and some resources
 */
static void
Initialize() {
  settings.vertex_num = 1LL << settings.scale;
  settings.edge_desired_num = settings.edge_factor * settings.vertex_num;
  settings.least_v_num = settings.vertex_num / settings.mpi_size 
    / kBitWidth * kBitWidth;

  g_bfs_time.reserve(64);
  g_teps.reserve(64);

  logger.log("Total vertexes      : %ld\n", settings.vertex_num);
  logger.log("Total desired edges : %ld\n", settings.edge_desired_num);
  logger.log("Least vertex number : %ld\n", settings.least_v_num);
}


static bool
CheckConnection(LocalCSRGraph &local_csr, int64_t index) {
  bool is_connect {false};

  int64_t index_owner = mpi_get_owner(index, settings.least_v_num);
  if (settings.mpi_rank == index_owner) {
    is_connect = local_csr.IsConnect(index);
  }
  MPI_Allreduce(MPI_IN_PLACE, &is_connect, 1, MPI_CHAR, MPI_LOR,
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
  while ((ssize_t)roots.size() < settings.sample_num && remain_vertex_num > 0) {

    // generate a random index
    int64_t index = rand_gen() % remain_vertex_num;
    MPI_Bcast(&index, 1, MPI_LONG_LONG, 0/*root*/, MPI_COMM_WORLD);
    // logger.debug("rand select index %ld\n", index);

    if (CheckConnection(local_csr, index)) {
      int64_t real_index = swap_map.find(index) == swap_map.end() ?
        index : swap_map[index];
      roots.push_back(real_index);

      logger.debug("%ld map to %ld connect!\n", index, real_index);
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
  logger.set_filter_level(Logger::kDebug);
  ParseParameters(argc, argv);

  if (settings.is_debug) {
    logger.set_filter_level(Logger::kDebug);
  } else {
    logger.set_filter_level(Logger::kLog);
  }

  Initialize();

#if 0
  void TEST_VerifyCase_1();
  TEST_VerifyCase_1();
  MPI_Finalize();
  return 0;
#endif

  LocalRawGraph local_raw {nullptr, 0, 0};

  if (!settings.file_in.empty()) {
    local_raw = LoadLocalRawGraph(settings.file_in);

  } else {
    local_raw = MPIGenerateGraph(
        settings.vertex_num, settings.edge_desired_num);

    if (!settings.file_out.empty()) {
      TickOnce tick_dump;
      DumpLocalRawGraph(settings.file_out, local_raw);
      logger.mpi_log("graph exit because of completing dumping: TIME %fms\n", tick_dump());
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
      return 0;
    }
  }

  LocalCSRGraph local_csr(local_raw);
  local_csr.Construct();
  vector<int64_t> roots = SampleKeys(local_csr);
  for (auto root : roots) {
    // Run BFS here
    int64_t *bfs_tree = BuildBFSTree(local_csr, root);

    if (settings.is_verify) {
      if (VerifyBFSTree(bfs_tree, local_csr.global_v_num(), root, local_raw)) {
        logger.log("verify bfs rooted %ld pass\n", root);
      } else {
        logger.error("verify bfs rooted %ld failed\n", root);
      }

    } else {
      logger.log("skip validation\n");
    }

    delete []bfs_tree;

  }

  float sum_bfs_time = 0.0f;
  for (auto f : g_bfs_time) {
    sum_bfs_time += f;
  }

  float sum_teps = 0.0f;
  for (auto f : g_teps) {
    sum_teps += f;
  }

  logger.log("average BFS TIME: %fms\n", sum_bfs_time / g_bfs_time.size());
  logger.log("average TEPS: %e\n", sum_teps / g_teps.size());

  delete [] local_raw.edges;
  local_raw.edges = nullptr;

  mpi_log_barrier();
  logger.log("graph500 exit!\n");
  MPI_Finalize();
  return 0;
}

