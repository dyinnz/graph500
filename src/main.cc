/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-27
 ******************************************************************************/

#include <iostream>

#include "unistd.h"

#include "utility.h"
#include "construct.h"

using namespace dy_logger;
using std::tuple;
using std::tie;

Logger logger;
Settings settings;

Edge* GeneratorGraph(int64_t vertex_num, int64_t edge_desired_num);

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

int
main(int argc, char *argv[]) {
#ifdef DEBUG
  logger.set_filter_level(Logger::kDebug);
#endif

  ParseParameters(argc, argv);
  Initialize();

  Edge *edges = GeneratorGraph(settings.vertex_num, settings.edge_desired_num);

  CSRGraph csr(edges, settings.edge_desired_num);
  csr.Construct();

  delete []edges;
  return 0;
}

