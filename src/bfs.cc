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

using std::vector;


int64_t g_local_v_num;
int64_t g_global_v_num;
int64_t g_local_v_beg;
int64_t g_local_v_end;
int64_t * __restrict__ g_local_adja_arrays {nullptr};
int64_t * __restrict__ g_local_csr_mem {nullptr};
int64_t * __restrict__ g_bfs_tree {nullptr};

struct QueuePair {
  int64_t parent;
  int64_t self;
};


// for bitmap
int64_t g_local_bitmap_size;
int64_t g_global_bitmap_size;
MPI_Datatype g_mpi_bit_type;
bit_type * __restrict__ g_local_bitmap {nullptr};
bit_type * __restrict__ g_global_bitmap {nullptr};


// for topdown
vector<QueuePair> g_current_queue;
vector<vector<QueuePair>> g_scatter_queues;


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

/* set the graph base data and alloc mem which bfs alg need use */
static void
SettingCSRGraph(LocalCSRGraph &local_csr, int64_t *bfs_tree) {
  // graph info

  g_local_v_num       = local_csr.local_v_num();
  g_global_v_num      = local_csr.global_v_num();
  g_local_v_beg       = local_csr.local_v_beg();
  g_local_v_end       = local_csr.local_v_end();
  g_local_adja_arrays = (int64_t*)local_csr.adja_arrays();
  g_local_csr_mem     = local_csr.csr_mem();
  g_bfs_tree          = bfs_tree;

  // bitmap info
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
MPIGatherQueue() {
  int64_t mpi_size = settings.mpi_size;
  int64_t mpi_rank = settings.mpi_rank;

  // gather the number
  vector<int> gather_nums(settings.mpi_size);
  for (size_t r = 0; r < gather_nums.size(); ++r) {
    gather_nums[r] = g_scatter_queues[r].size();
    /*
    logger.mpi_debug("%s(): scatter rank %ld size: %ld\n",
        __func__, r, gather_nums[r]);
        */
  }
  MPI_Allreduce(MPI_IN_PLACE, gather_nums.data(), mpi_size,
      MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // mpi sync
  g_current_queue.clear();
  g_current_queue.resize(gather_nums[mpi_rank]);
  logger.mpi_debug("%s(): total queue size: %ld\n",
      __func__, gather_nums[mpi_rank]);

  for (int base = 0; base < mpi_size; ++base) {

    if (base == mpi_rank) {
      QueuePair *offset = g_current_queue.data();
      memcpy(offset, g_scatter_queues[base].data(),
          g_scatter_queues[base].size() * sizeof(QueuePair));
      offset += g_scatter_queues[base].size();

      for (int i = 0; i < mpi_size; ++i) if (i != mpi_rank) {
        MPI_Status status;
        memset(&status, 0, sizeof(MPI_Status));
        MPI_Probe(i, 0, MPI_COMM_WORLD, &status);

        int recv_count {0};
        MPI_Get_count(&status, MPI_LONG_LONG, &recv_count);

        MPI_Recv(offset, recv_count, MPI_LONG_LONG, i, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        offset += recv_count / 2;

        //logger.mpi_debug("current queue size: %ld\n",
            //offset - g_current_queue.data());
      }

    } else {
      MPI_Send(g_scatter_queues[base].data(), g_scatter_queues[base].size() * 2,
          MPI_LONG_LONG, base, 0, MPI_COMM_WORLD);
    }
  }
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
InitQueue(vector<QueuePair> &current_queue,
    vector<vector<QueuePair>> &scatter_queues,
    int64_t root) {
  current_queue.clear();
  if (mpi_get_owner(root, settings.least_v_num) == settings.mpi_rank) {
    current_queue.push_back( {root, root} );
  }
  scatter_queues.resize(settings.mpi_size);
}


static void
BFSTopDown(int64_t * __restrict__ bfs_tree,
    bit_type * __restrict__ local_bitmap,
    bit_type * __restrict__ global_bitmap,
    vector<QueuePair> &current_queue,
    vector<vector<QueuePair>> &scatter_queues,
    bool &is_change) {

  // clear parents info
  for (auto &q : scatter_queues) {
    q.clear();
  }

  is_change = false;

  for (size_t i = 0; i < current_queue.size(); ++i) {
    const int64_t global_u = current_queue[i].self;
    const int64_t local_u = global_to_local(global_u);

    if (-1 == bfs_tree[local_u]) {
      bfs_tree[local_u] = current_queue[i].parent;
      set_bitmap(global_bitmap, global_u);
      set_bitmap(local_bitmap, local_u);
    }

    // for each edge judge if the vertex have be visited

    for (int64_t iter = adja_beg(local_u); iter < adja_end(local_u); ++iter) {
      int64_t global_v = next_vertex(iter);

      if (!test_bitmap(global_bitmap, global_v)) {
        set_bitmap(global_bitmap, global_v);

        // logger.mpi_debug("%s(): u %ld, v %ld\n", __func__, global_u, global_v);

        int64_t v_owner = mpi_get_owner(global_v, settings.least_v_num);
        scatter_queues[v_owner].push_back( {global_u, global_v} );

        is_change = true;
      }
    }
  }
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
BFSSwitch(vector<QueuePair> &current_queue,
    bit_type * __restrict__ local_bitmap,
    bit_type * __restrict__ global_bitmap,
    vector<int64_t> &unvisited) {

  for (size_t i = 0; i < current_queue.size(); ++i) {
    const int64_t global_u = current_queue[i].self;
    const int64_t local_u = global_to_local(global_u);

    if (-1 == g_bfs_tree[local_u]) {
      g_bfs_tree[local_u] = current_queue[i].parent;
      set_bitmap(global_bitmap, global_u);
      set_bitmap(local_bitmap, local_u);
    }
  }

  unvisited.clear();

  for (int64_t v = 0; v < g_local_v_num; ++v) {
    if (!test_bitmap(local_bitmap, v) && adja_beg(v) < adja_end(v)) {
      unvisited.push_back(v);
    }
  }

  MPIGatherAllBitmap();
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
  // scan each unvisited vertex
  for (size_t i = 0; i < unvisited_old.size(); ++i) {
    int64_t local_v = unvisited_old[i];

    int64_t global_v = local_to_global(local_v);
    if (-1 == bfs_tree[local_v]) {

      // scan each edge belong to local_v
      for (int64_t iter = adja_beg(local_v); iter < adja_end(local_v); ++iter) {
        int64_t global_u = next_vertex(iter);

        int64_t global_pos = global_u / kBitWidth;
        bit_type  global_mask = 1 << (global_u % kBitWidth);

        // if the vertex u si visited then mark u is v's parent
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

  for (int level = 0; ; ++level) {
    bool is_change = false;

    bool is_topdown = level < 2;

    if (is_topdown) {
      func_tick();

      static bool is_init_queue {false};
      if (!is_init_queue) {
        is_init_queue = true;

        InitQueue(g_current_queue, g_scatter_queues, root);
      }

      /*
      for (auto &p : g_current_queue) {
        logger.mpi_debug("current queue: %ld's parent %ld\n", p.self, p.parent);
      }
      */

      BFSTopDown(g_bfs_tree,
          g_local_bitmap,
          g_global_bitmap,
          g_current_queue,
          g_scatter_queues,
          is_change);

      last_tick = func_tick();
      logger.mpi_log("top down TIME : %fms\n", last_tick);
      total_calc_time += last_tick;

    } else {


      static bool is_init_unvisited {false};
      if (!is_init_unvisited) {
        is_init_unvisited = true;

        // FillunvisitedFromBitmap(g_local_bitmap, g_global_bitmap, unvisited_old);
        TickOnce switch_tick;

        /// init bottom up data
        BFSSwitch(g_current_queue, g_local_bitmap, g_global_bitmap, 
            unvisited_old);
        logger.mpi_log("switch TIME: %fms\n", switch_tick());
      }

      func_tick();

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

    if (is_topdown) {
      func_tick();
      MPIGatherQueue();
      last_tick = func_tick();
      logger.mpi_log("mpi sync queue TIME : %fms\n", last_tick);
      total_mpi_time += last_tick;

    } else {
      func_tick();
      MPIGatherAllBitmap();
      last_tick = func_tick();
      logger.mpi_log("mpi sync bitmap TIME : %fms\n", last_tick);
      total_mpi_time += last_tick;
    }
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

  // BFS alg
  MPIBFS(root, bfs_tree);

  ReleaseMemory();

  MPI_Barrier(MPI_COMM_WORLD);
  logger.log("end bfs.\n");
  return bfs_tree;
}

