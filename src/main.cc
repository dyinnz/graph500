/*******************************************************************************
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

Edge* GeneratorGraph(int64_t vertex_num, int64_t edge_desired_num);
int64_t* BuildBFSTree(CSRGraph &csr, int64_t root);
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

  logger.log("------------------------------------------------\n");
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

  logger.log("------------------------------------------------\n");
  logger.log("Total vertexes      : %d\n", settings.vertex_num);
  logger.log("Total desired edges : %d\n", settings.edge_desired_num);
}

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

int
main(int argc, char *argv[]) {
#ifdef DEBUG
  logger.set_filter_level(Logger::kDebug);
  logger.debug("graph500 run in debug mode.\n");
#endif

  ParseParameters(argc, argv);
  Initialize();

  Edge *edges = GeneratorGraph(settings.vertex_num, settings.edge_desired_num);

  CSRGraph csr(edges, settings.edge_desired_num);
  csr.Construct();

  vector<int64_t> roots = SampleKeys(csr);
  for (auto root : roots) {
    int64_t *bfs_tree = BuildBFSTree(csr, root);
    VerifyBFSTree(bfs_tree, csr.vertex_num(), root, 
        edges, settings.edge_desired_num);
#ifdef DEBUG
    break;
#endif
  }

  delete []edges;
  return 0;
}

