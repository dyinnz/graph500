/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-28
 ******************************************************************************/

#include <cassert>
#include <cstring>
#include <queue>
#include "utility.h"
#include "construct.h"
#include "bfs_gpu.h"

int64_t g_local_v_num;
int64_t g_global_v_num;
int64_t g_local_v_beg;
int64_t g_local_v_end;
int64_t * __restrict__ g_local_adja_arrays {nullptr};
int64_t * __restrict__ g_local_csr_mem {nullptr};
int64_t * __restrict__ g_bfs_tree {nullptr};


// for bitmap
int64_t g_local_bitmap_size;
int64_t g_global_bitmap_size;
MPI_Datatype g_mpi_bit_type;
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
  int64_t mem_pos = index / kBitWidth;
  int64_t bit_offset = index % kBitWidth;
  bitmap[mem_pos] |= 1 << bit_offset;
}


static inline bool test_bitmap(bit_type * __restrict__ bitmap, int64_t index) {
  int64_t mem_pos = index / kBitWidth;
  int64_t bit_offset = index % kBitWidth;
  return bitmap[mem_pos] & (1 << bit_offset);
}


/*----------------------------------------------------------------------------*/


static void
SettingCSRGraph(LocalCSRGraph &local_csr, int64_t *bfs_tree) {
  // graph
  g_local_v_num       = local_csr.local_v_num();
  g_global_v_num      = local_csr.global_v_num();
  g_local_v_beg       = local_csr.local_v_beg();
  g_local_v_end       = local_csr.local_v_end();
  g_local_adja_arrays = (int64_t*)local_csr.adja_arrays();
  g_local_csr_mem     = local_csr.csr_mem();
  g_bfs_tree          = bfs_tree;

  // bitmap
  g_local_bitmap_size =
    (local_csr.local_v_num() + kBitWidth - 1) / kBitWidth;

  g_global_bitmap_size =
    (local_csr.global_v_num() + kBitWidth - 1) / kBitWidth;

  g_local_bitmap      = new bit_type[g_local_bitmap_size];
  g_global_bitmap     = new bit_type[g_global_bitmap_size];
  memset(g_local_bitmap, 0, sizeof(bit_type) * g_local_bitmap_size);
  memset(g_global_bitmap, 0, sizeof(bit_type) * g_global_bitmap_size);

  if (1 == sizeof(bit_type)) {
    g_mpi_bit_type = MPI_CHAR;
  } else if (2 == sizeof(bit_type)) {
    g_mpi_bit_type = MPI_SHORT;
  } else if (4 == sizeof(bit_type)) {
    g_mpi_bit_type = MPI_INT;
  } else if (8 == sizeof(bit_type)) {
    g_mpi_bit_type = MPI_LONG_LONG;
  } else {
    g_mpi_bit_type = MPI_INT;
  }
}


static void
ReleaseMemory() {
  delete [] g_local_bitmap;
  delete [] g_global_bitmap;
}


static void
SetBFSRoot(int64_t root) {

  if (settings.mpi_rank == mpi_get_owner(root, settings.least_v_num)) {
    int64_t local_root = global_to_local(root);
    g_bfs_tree[local_root] = root;
    set_bitmap(g_local_bitmap, local_root);
  }

  set_bitmap(g_global_bitmap, root);
}


static void
MPIGatherAllBitmap() {

  int64_t local_bitmap_least_size = settings.least_v_num / kBitWidth;

  MPI_Allgather(g_local_bitmap, local_bitmap_least_size, g_mpi_bit_type,
      g_global_bitmap, local_bitmap_least_size, g_mpi_bit_type,
      MPI_COMM_WORLD);

  // the last process send the remainder vertexes to others
  int64_t remainder =
    g_global_bitmap_size - local_bitmap_least_size * settings.mpi_size;
  if (0 != remainder) {

    bit_type *bcast_buff = g_global_bitmap + g_global_bitmap_size - remainder;

    if (settings.mpi_rank == settings.mpi_size-1) {
      bit_type *local_buff = g_local_bitmap + g_local_bitmap_size - remainder;
      memcpy(bcast_buff, local_buff, sizeof(bit_type) * remainder);
    }

    MPI_Bcast(bcast_buff, remainder, g_mpi_bit_type, settings.mpi_size-1,
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

        int64_t global_pos = global_u / kBitWidth;
        bit_type  global_mask = 1 << (global_u % kBitWidth);

        if (global_bitmap[global_pos] & global_mask) {

          int64_t local_pos = local_v / kBitWidth;
          bit_type local_mask = 1 << (local_v % kBitWidth);

          local_bitmap[local_pos] |= local_mask;

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

  float total_calc_time {0.0f};
  float total_mpi_time {0.0f};
  float last_tick {0.0f};

  TickOnce total_bfs_tick;
  TickOnce func_tick;

  for (;;) {
    bool is_change = false;

    if (false) {
      // BFSTopDown
    } else {

      func_tick();
      BFSBottomUp(g_bfs_tree, g_local_bitmap, g_global_bitmap, is_change);
      last_tick = func_tick();
      logger.mpi_log("bottom up TIME : %fms\n", last_tick);
      total_calc_time += last_tick;
    }

    MPI_Allreduce(MPI_IN_PLACE, &is_change, 1, MPI_BYTE, 
        MPI_BOR, MPI_COMM_WORLD);
    if (!is_change) {
      break;
    }

    func_tick();
    MPIGatherAllBitmap();
    last_tick = func_tick();
    logger.mpi_log("sync mpi TIME : %fms\n", last_tick);
    total_mpi_time += last_tick;
  }

  for (int64_t v = 0; v < g_local_v_num; ++v) {
    logger.mpi_debug("after bfs, v[%ld]'s parent %ld\n", 
        local_to_global(v), g_bfs_tree[v]);
  }

  float bfs_time = total_bfs_tick();
  logger.log("bfs TIME %fms, calc TIME %lfms, mpi sync TIME %lf\n", 
      bfs_time, total_calc_time, total_mpi_time);
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

