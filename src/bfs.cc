/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-28
 ******************************************************************************/

#include <cstring>
#include <queue>
#include "utility.h"
#include "construct.h"
#include "bfs_gpu.h"

int64_t g_average;
int64_t g_local_v_num;
int64_t g_global_v_num;
int64_t g_local_v_beg;
int64_t g_local_v_end;
int64_t * __restrict__ g_local_adja_arrays {nullptr};
int64_t * __restrict__ g_local_csr_mem {nullptr};
int64_t * __restrict__ g_bfs_tree {nullptr};
bit_type * __restrict__ g_local_bitmap {nullptr};
bit_type * __restrict__ g_global_bitmap {nullptr};


static inline int64_t adja_beg(int64_t u) {
  return g_local_adja_arrays[u * 2];
}


static inline int64_t adja_end(int64_t u) {
  return g_local_adja_arrays[u * 2 + 1];
}


static inline int64_t next_vertex(int64_t offset) {
  return g_local_csr_mem[offset];
}


static inline int64_t local_to_global(int64_t local) {
  return g_local_v_beg + local;
}


static inline int64_t global_to_local(int64_t global) {
  return global - g_local_v_beg;
}


static inline void set_bitmap(bit_type  * __restrict__ bitmap, int64_t index) {
  bitmap[index] = true;
}


static inline bool test_bitmap(bit_type * __restrict__ bitmap, int64_t index) {
  return bitmap[index];
}


/*----------------------------------------------------------------------------*/


static void
SettingCSRGraph(LocalCSRGraph &local_csr, int64_t *bfs_tree) {
  g_average           = local_csr.global_v_num() / settings.mpi_size;
  g_local_v_num       = local_csr.local_v_num();
  g_global_v_num      = local_csr.global_v_num();
  g_local_v_beg       = local_csr.local_v_beg();
  g_local_v_end       = local_csr.local_v_end();
  g_local_adja_arrays = (int64_t*)local_csr.adja_arrays();
  g_local_csr_mem     = local_csr.csr_mem();

  g_local_bitmap      = new bit_type[local_csr.local_v_num()];
  g_global_bitmap     = new bit_type[local_csr.global_v_num()];
  memset(g_local_bitmap, 0, sizeof(bit_type) * local_csr.local_v_num());
  memset(g_global_bitmap, 0, sizeof(bit_type) * local_csr.global_v_num());

  g_bfs_tree          = bfs_tree;
}


static void
ReleaseMemory() {
  delete [] g_local_bitmap;
  delete [] g_global_bitmap;
}


static void
SetBFSRoot(int64_t root) {

  if (settings.mpi_rank == mpi_get_owner(root, g_average)) {
    int64_t local_root = global_to_local(root);
    g_bfs_tree[local_root] = root;
    set_bitmap(g_local_bitmap, local_root);
  }

  set_bitmap(g_global_bitmap, root);
}


static void
MPIGatherAllBitmap() {

  MPI_Allgather(g_local_bitmap, g_average, MPI_INT,
      g_global_bitmap, g_average, MPI_INT,
      MPI_COMM_WORLD);

  // the last process send the remainder vertexes to others
  int64_t remainder = g_global_v_num % settings.mpi_size;
  if (0 != remainder) {

    bit_type *bcast_buff = g_global_bitmap + g_global_v_num - remainder;

    if (settings.mpi_rank == settings.mpi_size-1) {
      bit_type *local_buff = g_local_bitmap * global_bitmap;
      memcpy(bcast_buff, local_buff, sizeof(bit_type) * remainder);
    }

    MPI_Bcast(bcast_buff, remainder, MPI_INT, settings.mpi_size-1,
        MPI_COMM_WORLD);
  }
}



static void
BFSBottomUp(int64_t * __restrict__ bfs_tree,
            bit_type * __restrict__ local_bitmap,
            bit_type * __restrict__ global_bitmap,
            bool &is_change) {

  is_change = false;

  for (int64_t local_v = 0; local_v < g_local_v_num; ++local_v) {

    int64_t global_v = local_to_global(local_v);
    if (-1 == bfs_tree[local_v]) {

      for (int64_t iter = adja_beg(local_v); iter < adja_end(local_v); ++iter) {
        int64_t global_u = next_vertex(iter);

        if (test_bitmap(global_bitmap, global_u)) {

          set_bitmap(local_bitmap, local_v);
          bfs_tree[local_v] = global_u;

          is_change = true;
          break;
        }
      }
    }
  }
}


static void
MPIBFS(int64_t root, int64_t *bfs_tree) {
  memset(bfs_tree, -1, sizeof(int64_t) * g_local_v_num);
  SetBFSRoot(root);

  /*
  for (int64_t v = 0; v < g_local_v_num; ++v) {
    logger.mpi_debug("before bfs, v[%ld]'s parent %ld; global bitmap %d\n",
        local_to_global(v), g_bfs_tree[v], 
        test_bitmap(g_global_bitmap, local_to_global(v)));
  }
  */

  TickOnce total_bfs_tick;
  TickOnce func_tick;

  for (;;) {
    bool is_change = false;

    if (false) {
      // BFSTopDown
    } else {

      func_tick();
      BFSBottomUp(g_bfs_tree, g_local_bitmap, g_global_bitmap, is_change);
      logger.mpi_log("bottom up TIME : %fms\n", func_tick());
    }

    MPI_Allreduce(MPI_IN_PLACE, &is_change, 1, MPI_BYTE, 
        MPI_BOR, MPI_COMM_WORLD);
    if (!is_change) {
      break;
    }

    func_tick();
    MPIGatherAllBitmap();
    logger.mpi_log("sync mpi TIME : %fms\n", func_tick());
  }

  for (int64_t v = 0; v < g_local_v_num; ++v) {
    logger.mpi_debug("after bfs, v[%ld]'s parent %ld\n", 
        local_to_global(v), g_bfs_tree[v]);
  }

  float bfs_time = total_bfs_tick();
  logger.log("bfs TIME %fms\n", bfs_time);
  logger.log("TEPS: %le\n", g_global_v_num * 16.0 / bfs_time * 1000.0);
}


/*----------------------------------------------------------------------------*/


int64_t *
BuildBFSTree(LocalCSRGraph &local_csr, int64_t root) {
  mpi_log_barrier();
  logger.log("begin bfs, root: %d...\n", root);

  int64_t *bfs_tree = new int64_t[local_csr.local_v_num()];
  memset(bfs_tree, 0, sizeof(int64_t) * local_csr.local_v_num());

  SettingCSRGraph(local_csr, bfs_tree);

  MPIBFS(root, bfs_tree);

  ReleaseMemory();

  MPI_Barrier(MPI_COMM_WORLD);
  logger.log("end bfs.\n");
  return bfs_tree;
}

