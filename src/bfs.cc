/*******************************************************************************
 * Author: Dyinnz.HUST.UniqueStudio
 * Email:  ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Date:   2016-05-28
 ******************************************************************************/

#include <cassert>
#include <cstring>
#include <vector>
#include "utility.h"
#include "construct.h"
#include "bfs_gpu.h"

using std::vector;


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

int64_t * __restrict__ g_queue {nullptr};
int64_t q_beg, q_end;

vector<int64_t> g_parent_vec;


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

  g_queue = new int64_t[local_csr.global_v_num()];
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
BFSTopDown(int64_t * __restrict__ bfs_tree,
    bit_type * __restrict__ global_bitmap,
    int64_t * __restrict__ g_queue,
    int64_t * __restrict__ q_beg,
    int64_t * __restrict__ q_end,
    bool &is_change) {

  is_change = false;

  const int64_t old_end = * q_end;

  int64_t new_end = old_end;

  for (int64_t index = * q_beg; index < old_end; ++index) {

    const int64_t global_u = g_queue[index];

    const int64_t local_u = global_to_local(global_u);
    for (int64_t iter = adja_beg(local_u); iter < adja_end(local_u); ++iter) {
      int64_t global_v = next_vertex(iter);

      if (test_bitmap(global_bitmap, global_v)) {
        const int64_t local_v = global_to_local(global_v);
        set_bitmap(global_bitmap, global_v);

        if (g_local_v_beg <= global_v && global_v < g_local_v_end) {
          bfs_tree[local_v] = global_u;
        } else {
          g_queue[new_end] = global_v;
          g_parent_vec.push_back(global_u);
        }
        new_end ++;

        is_change = true;
        break;
      }
    }
  }

  *q_beg = old_end;
  *q_end = new_end;
}


static void
FillunvisitedFromBitmap(bit_type * __restrict__ local_bitmap, 
    vector<int64_t> &unvisited) {

  unvisited.clear();

  for (int64_t v = 0; v < g_local_v_num; ++v) {
    if (!test_bitmap(local_bitmap, v) && adja_beg(v) < adja_end(v)) {
      unvisited.push_back(v);
    }
  }
}


static void
BFSBottomUp(int64_t * __restrict__ bfs_tree,
    bit_type * __restrict__ local_bitmap,
    bit_type * __restrict__ global_bitmap,
    vector<int64_t> &unvisited_old,
    vector<int64_t> &unvisited_new,
    bool &is_change) {

  unvisited_new.clear();

  is_change = false;

  logger.mpi_log("unvisited size: %zu\n", unvisited_old.size());
  for (size_t i = 0; i < unvisited_old.size(); ++i) {
    int64_t local_v = unvisited_old[i];

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

    if (-1 == bfs_tree[local_v]) {
      unvisited_new.push_back(local_v);
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

  vector<int64_t> unvisited_old;
  vector<int64_t> unvisited_new;
  unvisited_old.reserve(g_local_v_num);
  unvisited_new.reserve(g_local_v_num);

  for (;;) {
    bool is_change = false;

    if (false) {
      BFSTopDown(g_bfs_tree,
          g_global_bitmap,
          g_queue,
          &q_beg,
          &q_end,
          is_change);

    } else {

      func_tick();

      static bool is_init_unvisited {false};
      if (!is_init_unvisited) {
        is_init_unvisited = true;

        FillunvisitedFromBitmap(g_local_bitmap, unvisited_old);
      }

      BFSBottomUp(g_bfs_tree, 
          g_local_bitmap, 
          g_global_bitmap, 
          unvisited_old,
          unvisited_new,
          is_change);
      std::swap(unvisited_old, unvisited_new);

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

