/*
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
*/

/*----------------------------------------------------------------------------*/
// distributed graph for cuda

#include "bfs_gpu.h"

#include <sys/time.h>
#include <stdlib.h>

#include <cuda_runtime.h>

inline double wtime()
{
	double time[2];
	struct timeval time1;
	gettimeofday(&time1, NULL);

	time[0]=time1.tv_sec;
	time[1]=time1.tv_usec;

	return time[0]+time[1]*1.0e-6;
}

// global device variables

__device__ int      gd_mpi_rank;
__device__ int      gd_mpi_size;
__device__ int64_t  gd_local_v_num;
__device__ int64_t  gd_global_v_num;
__device__ int64_t  gd_local_v_beg;
__device__ int64_t  gd_local_v_end;
__device__ int64_t  gd_average;


/*----------------------------------------------------------------------------*/
// initializtion and finalizaiton


static void InitHostInfo(HostInfo &host_info) {
  // buffer size of all gather
  host_info.average = host_info.global_v_num / settings.mpi_size;
  host_info.change = 0;

  memset(host_info.bfs_tree, -1, sizeof(int64_t) * host_info.local_v_num);
}


static void InitCudaDevice(CudaInfo &cuda_info) {
  // TODO: change this value
  cuda_info.blocks_number = 8;
  cuda_info.threads_per_block = 64;

  cudaSetDevice(settings.mpi_rank % 4);
}


static void HostAllocMemory(HostInfo &host_info) {
  host_info.local_bitmap = new bit_type[host_info.local_v_num];
  host_info.global_bitmap = new bit_type[host_info.global_v_num];

  memset(host_info.local_bitmap, 0, sizeof(bit_type) * host_info.local_v_num);
  memset(host_info.global_bitmap, 0, sizeof(bit_type) * host_info.global_v_num);
}


static void HostFreeMemory(HostInfo &host_info) {
  delete []host_info.local_bitmap;
  delete []host_info.global_bitmap;
}


static void CudaAllocMemory(HostInfo &host_info, CudaGraphMemory &d_graph) {

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
  cudaMalloc((void**)&d_graph.global_bitmap,
      sizeof(bit_type) * host_info.global_v_num);
  cudaMalloc((void**)&d_graph.p_change, sizeof(bool));

  cudaMemset(d_graph.bfs_tree, -1, sizeof(int64_t) * host_info.local_v_num);
  cudaMemset(d_graph.local_bitmap, 0,
      sizeof(bit_type) * host_info.local_v_num);
  // cudaMemset(d_graph.global_bitmap, 0,
      // sizeof(bit_type) * host_info.global_v_num);
  cudaMemset(d_graph.p_change, 0, sizeof(bool));

  // just copy
  cudaMemcpyToSymbol(gd_mpi_rank, &settings.mpi_rank, sizeof(int));
  cudaMemcpyToSymbol(gd_mpi_size, &settings.mpi_size, sizeof(int));
  cudaMemcpyToSymbol(gd_local_v_num, &host_info.local_v_num, sizeof(int64_t));
  cudaMemcpyToSymbol(gd_global_v_num, &host_info.global_v_num, sizeof(int64_t));
  cudaMemcpyToSymbol(gd_local_v_beg, &host_info.local_v_beg, sizeof(int64_t));
  cudaMemcpyToSymbol(gd_local_v_end, &host_info.local_v_end, sizeof(int64_t));
  cudaMemcpyToSymbol(gd_average, &host_info.average, sizeof(int64_t));
}


static void CudaFreeMemory(CudaGraphMemory &d_graph) {
  cudaFree(d_graph.adja_arrays);
  cudaFree(d_graph.csr);
  cudaFree(d_graph.bfs_tree);
  cudaFree(d_graph.local_bitmap);
  cudaFree(d_graph.global_bitmap);
  cudaFree(d_graph.p_change);
}


/*----------------------------------------------------------------------------*/


static void MPIGatherAllBitmap(HostInfo &host_info, CudaGraphMemory &d_graph) {

  MPI_Allgather(host_info.local_bitmap, host_info.average, MPI_INT,
      host_info.global_bitmap, host_info.average, MPI_INT,
      MPI_COMM_WORLD);

  // the last process send the remainder vertexes to others
  int64_t remainder = host_info.global_v_num % settings.mpi_size;
  if (0 != remainder) {

    if (settings.mpi_rank == settings.mpi_size-1) {
      bit_type *send_buff = host_info.local_bitmap + host_info.local_v_num
        - remainder;
      MPI_Bcast(send_buff, remainder, MPI_INT, settings.mpi_size-1,
          MPI_COMM_WORLD);

    } else {
      bit_type *recv_buff = host_info.global_bitmap + host_info.global_v_num
        - remainder;
      MPI_Bcast(recv_buff, remainder, MPI_INT, settings.mpi_size-1,
          MPI_COMM_WORLD);
    }
  }
}


static void SetBFSRoot(HostInfo &host_info, CudaGraphMemory &d_graph) {

  for (int v = 0; v < host_info.global_v_num; ++v) {
    logger.mpi_debug("v[%ld] 's global bitmap %ld\n", v,
        host_info.global_bitmap[v]);
  }

  if (host_info.local_v_beg <= host_info.root && host_info.root < host_info.local_v_end) {
    // set root

    int64_t local_root = host_info.root - host_info.local_v_beg;
    host_info.bfs_tree[local_root] = host_info.root;
    host_info.local_bitmap[local_root] = true;

    cudaMemcpy(d_graph.bfs_tree, host_info.bfs_tree,
        sizeof(int64_t) * host_info.local_v_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph.local_bitmap, host_info.local_bitmap,
        sizeof(bit_type) * host_info.local_v_num, cudaMemcpyHostToDevice);
  }

  // sync with others

  MPIGatherAllBitmap(host_info, d_graph);

  cudaMemcpy(d_graph.global_bitmap, host_info.global_bitmap,
      sizeof(bit_type) * host_info.global_v_num, cudaMemcpyHostToDevice);

  for (int v = 0; v < host_info.global_v_num; ++v) {
    logger.mpi_debug("v[%ld] 's global bitmap %ld\n", v,
        host_info.global_bitmap[v]);
  }
}


static void SyncWithMPI(HostInfo &host_info, CudaGraphMemory &d_graph) {

  // without gpu direct

  /*------ bitmap ------*/
  cudaMemcpy(host_info.local_bitmap, d_graph.local_bitmap,
      sizeof(bit_type) * host_info.local_v_num, cudaMemcpyDeviceToHost);

  for (int v = 0; v < host_info.local_v_num; ++v) {
    logger.mpi_debug("v[%ld] 's local bitmap %ld\n", v + host_info.local_v_beg,
        host_info.local_bitmap[v]);
  }

  MPIGatherAllBitmap(host_info, d_graph);

  cudaMemcpy(d_graph.global_bitmap, host_info.global_bitmap,
      sizeof(bit_type) * host_info.global_v_num, cudaMemcpyHostToDevice);

  for (int v = 0; v < host_info.global_v_num; ++v) {
    logger.mpi_debug("v[%ld] 's global bitmap %ld\n", v,
        host_info.global_bitmap[v]);
  }


  /*------ change ------*/
  cudaMemcpy(&host_info.change, d_graph.p_change,
      sizeof(bool), cudaMemcpyDeviceToHost);
  MPI_Allreduce(MPI_IN_PLACE, &host_info.change, 1, MPI_BYTE,
      MPI_BOR, MPI_COMM_WORLD);
  // cudaMemcpy(d_graph.p_change, &host_info.change,
      // sizeof(bool), cudaMemcpyHostToDevice);
}



static void CopyBFSTree(HostInfo &host_info, CudaGraphMemory &d_graph) {
  cudaMemcpy(host_info.bfs_tree, d_graph.bfs_tree,
      sizeof(int64_t) * host_info.local_v_num, cudaMemcpyDeviceToHost);

  for (int64_t v = 0; v < host_info.local_v_num; ++v) {
    logger.mpi_debug("v[%ld] 's parent %ld\n", v + host_info.local_v_beg,
        host_info.bfs_tree[v]);
  }
}


static __device__ int64_t local_to_global(int64_t local) {
  return gd_local_v_beg + local;
}


static __device__ int64_t global_to_local(int64_t global) {
  return global - gd_local_v_beg;
}


static __global__ void BFSTopDown() {
  // TODO:
}


static __global__ void BFSBottomUp(
    int64_t *adja_arrays,
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

      /*printf("v [%ld] not visited, beg %ld, end %ld\n", */
          /*global_v, adja_arrays[2*local_v], adja_arrays[2*local_v+1]);*/

      for (int64_t iter = adja_arrays[2*local_v];
          iter < adja_arrays[2*local_v+1]; ++iter) {

        int64_t global_u = csr[iter];
        /*printf("v[%ld] <-> u[%ld]\n", global_v, global_u);*/

        // its parent havs been visited
        if (global_bitmap[global_u]) {

          /*printf("get v[%ld] 's parent global_u %ld\n", global_v, global_u);*/

          local_bitmap[local_v] = true;
          bfs_tree[local_v] = global_u;

          *p_change = true;
          break;
        }
      }
    }
  }
}


void CudaBFS(int64_t root,
             int64_t *adja_arrays,
             int64_t local_v_num,
             int64_t global_v_num,
             int64_t local_v_beg,
             int64_t local_v_end,
             int64_t *csr,
             int64_t csr_edge_num,
             int64_t *bfs_tree) {

  HostInfo host_info {
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
  CudaInfo cuda_info;
  CudaGraphMemory d_graph;

  InitHostInfo(host_info);
  InitCudaDevice(cuda_info);

  HostAllocMemory(host_info);
  CudaAllocMemory(host_info, d_graph);

  /*--------------------------------------------------------------------------*/
  // debug print
  logger.mpi_debug("root:         %ld\n", host_info.root);
  logger.mpi_debug("local_v_num:  %ld\n", host_info.local_v_num);
  logger.mpi_debug("global_v_num: %ld\n", host_info.global_v_num);
  logger.mpi_debug("local_v_beg:  %ld\n", host_info.local_v_beg);
  logger.mpi_debug("local_v_end:  %ld\n", host_info.local_v_end);
  logger.mpi_debug("csr_edge_num: %ld\n", host_info.csr_edge_num);
  logger.mpi_debug("average:      %ld\n", host_info.average);
  logger.mpi_debug("remainder:    %ld\n",
      host_info.global_v_num % settings.mpi_size);
  /*--------------------------------------------------------------------------*/

  SetBFSRoot(host_info, d_graph);

  for (int64_t v = 0; v < host_info.local_v_num; ++v) {
    logger.mpi_debug("v[%ld] 's parent %ld; global bitmap %d\n",
        v + host_info.local_v_beg, host_info.bfs_tree[v],
        host_info.global_bitmap[v+host_info.local_v_beg]);
  }


  double time = wtime();
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

  double bfs_bw_time = wtime() - time;
  logger.log("bfs time %lf\n", bfs_bw_time*1000);

  CopyBFSTree(host_info, d_graph);

  cudaDeviceSynchronize();

  CudaFreeMemory(d_graph);
  HostFreeMemory(host_info);
}

