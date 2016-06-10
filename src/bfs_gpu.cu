#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <set>
#include <string.h>
#include <queue>
#include <mpi.h>
//#include "translator_json_csr.h"
#include "wtime.h"
#include "graph.h"

#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 32768
#define STAND_THREADS 256
#define STAND_BLOCKS 256
#define THREAD_BIN_SIZE 32

#define INF (int)(1<<30)

#define visited_color 1
#define unvisited_color 0
#define frontier_color 2

//#define first_alpha 4

typedef int bit_type;

const char output_file[] = "bfs_result.txt";
const char time_detail[] = "time_detail.csv";


int pivot_selection_first(
        graph *g,
        index_t vertex_count)
{
    long int max_out_degree = 0;
    index_t index = 0;
    for(index_t j=0; j<vertex_count; ++j)
    {
        long int temp_out_degree = g->beg_pos[j+1] - g->beg_pos[j];
        if(temp_out_degree > max_out_degree)
        {
            max_out_degree = temp_out_degree;
            index = j;
        }
    }

    printf("first_pivot = %d, out_degree = %d\n", index, max_out_degree);
    return index;
}

__global__ void bfs_sync_color_top_down_first(
        index_t *d_adj_list,
        index_t *d_beg_pos,
        bit_type *d_vertex_status,
        bool *d_change,
        index_t *d_vertex_count,
        index_t *d_level,
        index_t *d_frontier_queue,
        index_t *d_thread_bin)
{
    index_t id = blockIdx.x*blockDim.x + threadIdx.x;
    const index_t THDS_COUNT=blockDim.x*gridDim.x;
    index_t vertex_count = *d_vertex_count;
    index_t level = *d_level;
    //    index_t bin_offset = 0;
    //    index_t id = tid;
    //    index_t begin_offset = 0;
    //    if(id != 0)
    //        begin_offset = tid * THREAD_BIN_SIZE;
    while(id < vertex_count)
    {
        if(d_vertex_status[id] == level)
        {
            for(index_t i=d_beg_pos[id]; i<d_beg_pos[id+1]; ++i)
            {
                index_t w = d_adj_list[i];

                ///atomic operation to guarantee only thread has w
                if(d_vertex_status[w] == unvisited_color)
                {
                    d_vertex_status[w] = level + 1;
                    *d_change = true;
//                    d_thread_bin[bin_offset] = w;
//                    bin_offset ++;
                }
            }
        }
        id += THDS_COUNT;
    }
}

__global__ void bfs_bottom_up_first(
        bit_type *d_vertex_status_bw,
        index_t *d_adj_list_bw,
        index_t *d_beg_pos_bw,
        bool *d_change,
        index_t *d_vertex_count,
        index_t *d_level)
{
    index_t id = blockIdx.x*blockDim.x + threadIdx.x;
    const index_t THDS_COUNT=blockDim.x*gridDim.x;
    index_t vertex_count = *d_vertex_count;
    index_t level = *d_level;
    while(id < vertex_count)
    {
        if(d_vertex_status_bw[id] == 0)
        {
            for(index_t i=d_beg_pos_bw[id]; i<d_beg_pos_bw[id+1]; ++i)
            {
                index_t w = d_adj_list_bw[i];
                if(d_vertex_status_bw[w] == level)
                {
                    d_vertex_status_bw[id] = level + 1;
                    *d_change = true;
                    break;
                    // printf("%d\n", id);
                }
            }
        }
        id += THDS_COUNT;
    }
}



void bfs_cu(index_t root,
            index_t *beg_pos,
            const index_t vert_count,
            vertex_t *csr,
            const index_t edge_count,
            const index_t first_alpha=2)
{
    srand(time(NULL));
    cudaSetDevice(0);

    graph *g = new graph(beg_pos, vert_count, csr, edge_count);
    const index_t vertex_count = g->vert_count + 1;

    bit_type *backward_vertex_status = (bit_type *)malloc(sizeof(bit_type)*vertex_count);
    index_t *frontier_queue = (index_t*)malloc(sizeof(index_t)*vertex_count);

    //------------------------------------------------------------------------
    //Deciding how many blocks to be used
    index_t number_of_blocks = 1;
    index_t number_of_threads_per_block = vertex_count;

    if(vertex_count > THREADS_PER_BLOCK)
    {
        number_of_blocks = (index_t)ceil(vertex_count/(double)THREADS_PER_BLOCK);
        number_of_threads_per_block = THREADS_PER_BLOCK;
        if(number_of_blocks > BLOCKS_PER_GRID)
            number_of_blocks = BLOCKS_PER_GRID;
    }

    //    printf("blocks = %d, threads = %d\n", number_of_blocks, number_of_threads_per_block);
    //------------------------------------------------------------------------
    //allocating auxiliars in CPU
    for(index_t i = 0 ; i < vertex_count ; ++i)
    {
        backward_vertex_status[i] = 0;//no colors
    }
    backward_vertex_status[root] = 1;
    //------------------------------------------------------------------------
    //Allocating GPU memory:
    printf("root = %ld\n", root);
    index_t *d_adj_list_reverse;
    cudaMalloc((void**) &d_adj_list_reverse, sizeof(index_t)*edge_count);
    cudaMemcpy( d_adj_list_reverse, g->csr, sizeof(index_t)*edge_count, cudaMemcpyHostToDevice);

    index_t *d_beg_pos_reverse;
    cudaMalloc((void**) &d_beg_pos_reverse, sizeof(index_t)*(vertex_count + 1));
    cudaMemcpy( d_beg_pos_reverse, g->beg_pos, sizeof(index_t)*(vertex_count + 1), cudaMemcpyHostToDevice);

    bit_type *d_backward_vertex_status;
    cudaMalloc((void**) &d_backward_vertex_status, sizeof(bit_type)*vertex_count);
    cudaMemcpy( d_backward_vertex_status, backward_vertex_status, sizeof(index_t)*vertex_count, cudaMemcpyHostToDevice);

    index_t *d_vertex_count;
    cudaMalloc((void **) &d_vertex_count, sizeof(index_t));
    cudaMemcpy(d_vertex_count, &(vertex_count), sizeof(index_t), cudaMemcpyHostToDevice);

    index_t offset = 0;
    index_t *d_offset;
    cudaMalloc((void **) &d_offset, sizeof(index_t));
    cudaMemcpy(d_offset, &(offset), sizeof(index_t), cudaMemcpyHostToDevice);

    index_t *d_frontier_queue;
    cudaMalloc((void **) &d_frontier_queue, sizeof(index_t) * vertex_count);

    index_t *d_thread_bin;
    cudaMalloc((void **) &d_thread_bin, sizeof(index_t) * STAND_BLOCKS * STAND_THREADS * THREAD_BIN_SIZE);
    cudaMemcpy(d_vertex_count, &(vertex_count), sizeof(index_t), cudaMemcpyHostToDevice);
    //CPU & GPU shared variable

    index_t * level;
    index_t * d_level;
    cudaHostAlloc((void **) &level, sizeof(index_t), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_level, level, 0);
    *level = 0;

    bool * change;
    bool * d_change;
    cudaHostAlloc((void **) &change, sizeof(bool), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_change, change, 0);
    *change = true;

    FILE * fp_time = fopen(time_detail, "w");
    double time = wtime();

    double bfs_bw_time;
    do {
        while(*level < first_alpha)
        {
            (*level) ++;
            *change = false;
            double temp_time_beg = wtime();

            bfs_sync_color_top_down_first<<<STAND_BLOCKS, STAND_THREADS>>>(
                    d_adj_list_reverse,
                    d_beg_pos_reverse,
                    d_backward_vertex_status,
                    d_change,
                    d_vertex_count,
                    d_level,
                    d_frontier_queue,
                    d_thread_bin);

            cudaDeviceSynchronize();
            double temp_time = wtime() - temp_time_beg;
            fprintf(fp_time, "%d, %g\n", *level, temp_time*1000);

            if (!*change) {
               break;
            }

        }

        bfs_bw_time = wtime() - time;
        printf("depth = %d\n", *level);
        printf("bfs top down time = %g (ms)\n", bfs_bw_time * 1000);

        if (!*change) {
           break;
        }


        while(*change)
        {
            (*level) ++;
            *change = false;
            double temp_time_beg = wtime();

            bfs_bottom_up_first<<<STAND_BLOCKS, STAND_THREADS>>>(
                    d_backward_vertex_status,
                    d_adj_list_reverse,
                    d_beg_pos_reverse,
                    d_change,
                    d_vertex_count,
                    d_level);

            cudaDeviceSynchronize();
            double temp_time = wtime() - temp_time_beg;
            fprintf(fp_time, "%d, %g\n", *level, temp_time*1000);
        }
    } while(*change);

    bfs_bw_time = wtime() - time;
    fclose(fp_time);
    printf("depth = %d\n", *level);
    printf("bfs bottom up time = %g (ms)\n", bfs_bw_time * 1000);
    printf("teps %g (ms)\n", edge_count / bfs_bw_time * 1000);
    cudaMemcpy(backward_vertex_status, d_backward_vertex_status, sizeof(index_t)*vertex_count, cudaMemcpyDeviceToHost);
    FILE * fp_out = fopen(output_file, "w");
    for(index_t i=0; i<vertex_count; ++i)
    {
        fprintf(fp_out, "%d %d\n", i, backward_vertex_status[i]);
    }
    fclose(fp_out);

    free(backward_vertex_status);
    cudaFree(d_adj_list_reverse);
    cudaFree(d_beg_pos_reverse);
    cudaFree(d_backward_vertex_status);
    cudaFree(d_change);
}

/*----------------------------------------------------------------------------*/
// distributed graph for cuda

// global device variables

__device__ int      gd_mpi_rank;
__device__ int      gd_mpi_size;
__device__ int64_t  gd_local_v_num;
__device__ int64_t  gd_global_v_num;
__device__ int64_t  gd_local_v_beg;
__device__ int64_t  gd_local_v_end;
__device__ int64_t  gd_average;

struct CudaInfo {
  int64_t blocks_number;
  int64_t threads_per_block;
};

struct HostInfo {
  int64_t mpi_rank;
  int64_t mpi_size;
  int64_t root;
  int64_t *adja_arrays;
  int64_t local_v_num;
  int64_t global_v_num;
  int64_t local_v_beg;
  int64_t local_v_end;
  int64_t *csr;
  int64_t csr_edge_num;
  int64_t *bfs_tree;
  // calc
  int64_t average;
  bool change;
  /*int64_t mpi_local_v_num;*/
  /*int64_t mpi_global_v_num;*/
  // tmp
  bit_type *local_bitmap;
  bit_type *global_bitmap;
};

struct CudaGraphMemory {
  int64_t *adja_arrays;
  int64_t *csr;
  int64_t *bfs_tree;
  bit_type *local_bitmap;
  bit_type *global_bitmap;
  bool *p_change;
};

#define debug_print(__format, ...) do { \
  printf("RANK-%d %s(): "__format, host_info.mpi_rank, __func__, ##__VA_ARGS__); \
} while (false);

/*----------------------------------------------------------------------------*/

void InitHostInfo(HostInfo &host_info) {
  // buffer size of all gather
  host_info.average = host_info.global_v_num / host_info.mpi_size;
  host_info.change = 0;

  memset(host_info.bfs_tree, -1, sizeof(int64_t) * host_info.local_v_num);
}

void InitCudaDevice(CudaInfo &cuda_info) {
  // TODO: change this value
  cuda_info.blocks_number = 8;
  cuda_info.threads_per_block = 64;
}

void HostAllocMemory(HostInfo &host_info) {
  host_info.local_bitmap = new bit_type[host_info.local_v_num];
  host_info.global_bitmap = new bit_type[host_info.global_v_num];

  memset(host_info.local_bitmap, 0, sizeof(bit_type) * host_info.local_v_num);
  memset(host_info.global_bitmap, 0, sizeof(bit_type) * host_info.global_v_num);
}

void HostFreeMemory(HostInfo &host_info) {
  delete []host_info.local_bitmap;
  delete []host_info.global_bitmap;
}

void SyncWithMPI(HostInfo &host_info, CudaGraphMemory &d_graph) {

  // without gpu direct

  /*------ bitmap ------*/
  cudaMemcpy(host_info.local_bitmap, d_graph.local_bitmap,
      sizeof(bit_type) * host_info.local_v_num, cudaMemcpyDeviceToHost);

  for (int v = 0; v < host_info.local_v_num; ++v) {
    debug_print("v[%ld] 's local bitmap %ld\n", v + host_info.local_v_beg,
        host_info.local_bitmap[v]);
  }

  // all mpi processes have the average vertexes at least
  MPI_Allgather(host_info.local_bitmap, host_info.average, MPI_INT,
      host_info.global_bitmap, host_info.average, MPI_INT,
      MPI_COMM_WORLD);

  // the last process send the remainder vertexes to others
  int64_t remainder = host_info.global_v_num % host_info.mpi_size;
  if (0 != remainder) {

    if (host_info.mpi_rank == host_info.mpi_size-1) {
      bit_type *send_buff = host_info.local_bitmap + host_info.local_v_num
        - remainder;
      MPI_Bcast(send_buff, remainder, MPI_INT, host_info.mpi_size-1, 
          MPI_COMM_WORLD);

    } else {
      bit_type *recv_buff = host_info.global_bitmap + host_info.global_v_num 
        - remainder;
      MPI_Bcast(recv_buff, remainder, MPI_INT, host_info.mpi_size-1,
          MPI_COMM_WORLD);
    }
  }

  cudaMemcpy(d_graph.global_bitmap, host_info.global_bitmap, 
      sizeof(bit_type) * host_info.global_v_num, cudaMemcpyHostToDevice);


  /*------ change ------*/
  cudaMemcpy(&host_info.change, d_graph.p_change,
      sizeof(bool), cudaMemcpyDeviceToHost);
  MPI_Allreduce(MPI_IN_PLACE, &host_info.change, 1, MPI_BYTE, 
      MPI_BOR, MPI_COMM_WORLD);
  cudaMemcpy(d_graph.p_change, &host_info.change, 
      sizeof(bool), cudaMemcpyHostToDevice);

  for (int v = 0; v < host_info.global_v_num; ++v) {
    debug_print("v[%ld] 's global bitmap %ld\n", v,
        host_info.global_bitmap[v]);
  }
}

void SetBFSRoot(HostInfo &host_info, CudaGraphMemory &d_graph) {
  for (int v = 0; v < host_info.global_v_num; ++v) {
    debug_print("v[%ld] 's global bitmap %ld\n", v,
        host_info.global_bitmap[v]);
  }

  if (host_info.local_v_beg <= host_info.root && host_info.root < host_info.local_v_end) {
    int64_t local_root = host_info.root - host_info.local_v_beg;
    host_info.bfs_tree[local_root] = host_info.root;
    host_info.local_bitmap[local_root] = true;

    cudaMemcpy(d_graph.bfs_tree, host_info.bfs_tree, 
        sizeof(int64_t) * host_info.local_v_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph.local_bitmap, host_info.local_bitmap, 
        sizeof(bit_type) * host_info.local_v_num, cudaMemcpyHostToDevice);
  }

  MPI_Allgather(host_info.local_bitmap, host_info.average, MPI_INT,
      host_info.global_bitmap, host_info.average, MPI_INT,
      MPI_COMM_WORLD);

  // the last process send the remainder vertexes to others
  int64_t remainder = host_info.global_v_num % host_info.mpi_size;
  if (0 != remainder) {

    if (host_info.mpi_rank == host_info.mpi_size-1) {
      bit_type *send_buff = host_info.local_bitmap + host_info.local_v_num
        - remainder;
      MPI_Bcast(send_buff, remainder, MPI_INT, host_info.mpi_size-1, 
          MPI_COMM_WORLD);

    } else {
      bit_type *recv_buff = host_info.global_bitmap + host_info.global_v_num 
        - remainder;
      MPI_Bcast(recv_buff, remainder, MPI_INT, host_info.mpi_size-1,
          MPI_COMM_WORLD);
    }
  }

  cudaMemcpy(d_graph.global_bitmap, host_info.global_bitmap, 
      sizeof(bit_type) * host_info.global_v_num, cudaMemcpyHostToDevice);

  for (int v = 0; v < host_info.global_v_num; ++v) {
    debug_print("v[%ld] 's global bitmap %ld\n", v,
        host_info.global_bitmap[v]);
  }
}

void CopyBFSTree(HostInfo &host_info, CudaGraphMemory &d_graph) {
  cudaMemcpy(host_info.bfs_tree, d_graph.bfs_tree,
      sizeof(int64_t) * host_info.local_v_num, cudaMemcpyDeviceToHost);

  for (int64_t v = 0; v < host_info.local_v_num; ++v) { 
    debug_print("v[%ld] 's parent %ld\n", v + host_info.local_v_beg,
        host_info.bfs_tree[v]);
  }
}

void CudaAllocMemory(HostInfo &host_info, CudaGraphMemory &d_graph) {
  // alloc and copy
  cudaMalloc((void**)&d_graph.adja_arrays, 
      sizeof(int64_t) * host_info.local_v_num * 2);
  cudaMalloc((void**)&d_graph.csr, sizeof(int64_t) * host_info.csr_edge_num);

  cudaMemcpy(d_graph.adja_arrays, host_info.adja_arrays,
      sizeof(int64_t) * host_info.local_v_num * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph.csr, host_info.csr, 
      sizeof(int64_t) * host_info.csr_edge_num, cudaMemcpyHostToDevice);

  // alloc and clear
  cudaMalloc((void**)&d_graph.bfs_tree,
      sizeof(int64_t) * host_info.local_v_num);
  cudaMalloc((void**)&d_graph.local_bitmap,
      sizeof(bit_type) * host_info.local_v_num);
      /*sizeof(bit_type) * host_info.mpi_local_v_num);*/
  cudaMalloc((void**)&d_graph.global_bitmap,
      sizeof(bit_type) * host_info.global_v_num);
      /*sizeof(bit_type) * host_info.mpi_global_v_num);*/
  cudaMalloc((void**)&d_graph.p_change, sizeof(bool));
  cudaMemset(d_graph.bfs_tree, -1, sizeof(int64_t) * host_info.local_v_num);
  cudaMemset(d_graph.local_bitmap, 0, 
      sizeof(bit_type) * host_info.local_v_num);
      /*sizeof(bit_type) * host_info.mpi_local_v_num);*/
  cudaMemset(d_graph.global_bitmap, 0,
      sizeof(bit_type) * host_info.global_v_num);
      /*sizeof(bit_type) * host_info.mpi_global_v_num);*/
  cudaMemset(d_graph.p_change, 0, sizeof(bool));

  // just copy
  cudaMemcpyToSymbol(gd_mpi_rank, &host_info.mpi_rank, sizeof(int));
  cudaMemcpyToSymbol(gd_mpi_size, &host_info.mpi_size, sizeof(int));
  cudaMemcpyToSymbol(gd_local_v_num, &host_info.local_v_num, sizeof(int64_t));
  cudaMemcpyToSymbol(gd_global_v_num, &host_info.global_v_num, sizeof(int64_t));
  cudaMemcpyToSymbol(gd_local_v_beg, &host_info.local_v_beg, sizeof(int64_t));
  cudaMemcpyToSymbol(gd_local_v_end, &host_info.local_v_end, sizeof(int64_t));

  int64_t average = host_info.global_v_num / host_info.mpi_size;
  cudaMemcpyToSymbol(gd_average, &average, sizeof(int64_t));
}

void CudaFreeMemory(CudaGraphMemory &d_graph) {
  cudaFree(d_graph.adja_arrays);
  cudaFree(d_graph.csr);
  cudaFree(d_graph.bfs_tree);
  cudaFree(d_graph.local_bitmap);
  cudaFree(d_graph.global_bitmap);
  cudaFree(d_graph.p_change);
}

__device__ int64_t local_to_global(int64_t local) {
  return gd_local_v_beg + local;
}

__device__ int64_t global_to_local(int64_t global) {
  return global - gd_local_v_beg;
}

__global__ void BFSTopDown() {

}

__global__ void BFSBottomUp(int64_t *adja_arrays,
                            int64_t *csr,
                            int64_t *bfs_tree,
                            bit_type *local_bitmap,
                            bit_type *global_bitmap,
                            bool *p_change) {

  const int64_t local_v_num  = gd_local_v_num;
  const int64_t kThreadsNumber = blockDim.x * gridDim.x;

  *p_change = false;

  for (int64_t local_v = blockIdx.x*blockDim.x + threadIdx.x;
      local_v < local_v_num; local_v += kThreadsNumber) {

    // unvisited
    int64_t global_v = local_to_global(local_v);
    if (-1 == bfs_tree[local_v]) {

      printf("v [%ld] not visited, beg %ld, end %ld\n", 
          global_v, adja_arrays[2*local_v], adja_arrays[2*local_v+1]);

      for (int64_t iter = adja_arrays[2*local_v]; 
          iter < adja_arrays[2*local_v+1]; ++iter) {

        int64_t global_u = csr[iter];
        printf("v[%ld] <-> u[%ld]\n", global_v, global_u);

        // its parent havs been visited
        if (global_bitmap[global_u]) {

          printf("get v[%ld] 's parent global_u %ld\n", global_v, global_u);

          local_bitmap[local_v] = true;
          bfs_tree[local_v] = global_u;

          *p_change = true;
        }
      }
    }
  }
}

void CudaBFS(int mpi_rank,
             int mpi_size,
             int64_t root, 
             int64_t *adja_arrays, 
             int64_t local_v_num, 
             int64_t global_v_num,
             int64_t local_v_beg,
             int64_t local_v_end,
             int64_t *csr,
             int64_t csr_edge_num,
             int64_t *bfs_tree) {

  CudaInfo cuda_info;
  HostInfo host_info = {
    mpi_rank,
    mpi_size,
    root,
    adja_arrays,
    local_v_num,
    global_v_num,
    local_v_beg,
    local_v_end,
    csr,
    csr_edge_num,
    bfs_tree,
    0,
    0,
    0,
    0,
  };
  CudaGraphMemory d_graph;

  InitHostInfo(host_info);

  cudaSetDevice(mpi_rank % 4);
  InitCudaDevice(cuda_info);

  HostAllocMemory(host_info);
  CudaAllocMemory(host_info, d_graph);

  /*--------------------------------------------------------------------------*/
  // debug print
  debug_print("mpi_rank: %d\n", host_info.mpi_rank);
  debug_print("mpi_size: %d\n", host_info.mpi_size);
  debug_print("root: %ld\n", host_info.root);
  debug_print("local_v_num: %ld\n", host_info.local_v_num);
  debug_print("global_v_num: %ld\n", host_info.global_v_num);
  debug_print("local_v_beg: %ld\n", host_info.local_v_beg);
  debug_print("local_v_end: %ld\n", host_info.local_v_end);
  debug_print("csr_edge_num: %ld\n", host_info.csr_edge_num);
  debug_print("average: %ld\n", host_info.average);
  debug_print("remainder: %ld\n", host_info.global_v_num % host_info.mpi_size);

  /*--------------------------------------------------------------------------*/

  SetBFSRoot(host_info, d_graph);

  for (int64_t v = 0; v < host_info.local_v_num; ++v) { 
    debug_print("v[%ld] 's parent %ld; global bitmap %d\n", 
        v + host_info.local_v_beg, host_info.bfs_tree[v], 
        host_info.global_bitmap[v+host_info.local_v_beg]);
  }
  debug_print("--------------------------\n");
  MPI_Barrier(MPI_COMM_WORLD);

  for (int64_t u = 0; u < host_info.local_v_num; ++u) {
    int64_t beg = host_info.adja_arrays[2*u],
            end = host_info.adja_arrays[2*u+1];
    debug_print("before cuda u adja beg %ld, end %ld\n", beg, end);
    for (int64_t iter = beg; iter != end; ++iter) {
      debug_print("before cuda: %ld -> %ld\n", 
          u+host_info.local_v_beg, host_info.csr[iter]);
    }
  }

  do {

    if (false) {
      BFSTopDown<<<cuda_info.blocks_number, cuda_info.threads_per_block>>>(
          );
    } else {
      BFSBottomUp<<<cuda_info.blocks_number, cuda_info.threads_per_block>>>(
          d_graph.adja_arrays,
          d_graph.csr,
          d_graph.bfs_tree,
          d_graph.local_bitmap,
          d_graph.global_bitmap,
          d_graph.p_change);
    }

    SyncWithMPI(host_info, d_graph);

  } while (host_info.change);

  CopyBFSTree(host_info, d_graph);

  cudaDeviceSynchronize();

  CudaFreeMemory(d_graph);
  HostFreeMemory(host_info);
}


