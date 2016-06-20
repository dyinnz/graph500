
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "utility.h"

using std::string;


/**
 * Dump the raw edges to file
 */
void DumpLocalRawGraph(const string &filename, LocalRawGraph &local_raw) {
  logger.mpi_log("%s():\n", __func__);

  std::stringstream ss;
  ss << filename << '_' << settings.mpi_rank;

  FILE *fp = fopen(ss.str().c_str(), "w");
  fwrite(&local_raw.global_edge_num, sizeof(local_raw.global_edge_num), 1, fp);
  fwrite(&local_raw.edge_num, sizeof(local_raw.edge_num), 1, fp);
  fwrite(local_raw.edges, sizeof(*local_raw.edges), local_raw.edge_num, fp);

  logger.mpi_log("write to file: %s\n", ss.str().c_str());
}


/**
 * Read the raw edges from file
 */
LocalRawGraph LoadLocalRawGraph(const string &filename) {
  std::stringstream ss;
  ss << filename << '_' << settings.mpi_rank;
  FILE *fp = fopen(ss.str().c_str(), "r");

  logger.mpi_log("read from file  : %s\n", ss.str().c_str());

  LocalRawGraph local_raw;
  fread(&local_raw.global_edge_num, sizeof(local_raw.global_edge_num), 1, fp);
  fread(&local_raw.edge_num, sizeof(local_raw.edge_num), 1, fp);

  logger.mpi_log("global edge num : %ld\n", local_raw.global_edge_num);
  logger.mpi_log("edge num        : %ld\n", local_raw.edge_num);

  local_raw.edges = new Edge[local_raw.edge_num];
  fread(local_raw.edges, sizeof(*local_raw.edges), local_raw.edge_num, fp);

  logger.mpi_log("finish reading from file\n", ss.str().c_str());

  return local_raw;
}
