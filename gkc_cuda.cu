#define _in_
#define _out_

#define FILTER_KERNEL new_filter2 // modify this to change filter: mm_filter, sign_filter, new_filter, new_filter2
#define STR1(R)  #R
#define STR(R) STR1(R)

#include "gkc_cuda.hpp"
#include "types.h"
#include "utilities.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <string>
#include <iostream>

using namespace std;

__device__ __constant__ static const unsigned char d_basemap[256] = {
    255, 255, 255, 255, 255, 255, 255, 255, // 0..7
    255, 255, 255, 255, 255, 255, 255, 255, // 8..15
    255, 255, 255, 255, 255, 255, 255, 255, // 16..23
    255, 255, 255, 255, 255, 255, 255, 255, // 24..31
    255, 255, 255, 255, 255, 255, 255, 255, // 32..39
    255, 255, 255, 255, 255, 255, 255, 255, // 40..47
    255, 255, 255, 255, 255, 255, 255, 255, // 48..55
    255, 255, 255, 255, 255, 255, 255, 255, // 56..63
    255, 0, 255, 1, 255, 255, 255, 2, // 64..71
    255, 255, 255, 255, 255, 255, 255, 255, // 72..79
    255, 255, 255, 255, 3, 0, 255, 255, // 80..87
    255, 255, 255, 255, 255, 255, 255, 255, // 88..95
    255, 0, 255, 1, 255, 255, 255, 2, // 96..103
    255, 255, 255, 255, 255, 255, 255, 255, // 104..111
    255, 255, 255, 255, 3, 0, 255, 255, // 112..119
    255, 255, 255, 255, 255, 255, 255, 255, // 120..127
    255, 255, 255, 255, 255, 255, 255, 255, // 128..135
    255, 255, 255, 255, 255, 255, 255, 255, // 136..143
    255, 255, 255, 255, 255, 255, 255, 255, // 144..151
    255, 255, 255, 255, 255, 255, 255, 255, // 152..159
    255, 255, 255, 255, 255, 255, 255, 255, // 160..167
    255, 255, 255, 255, 255, 255, 255, 255, // 168..175
    255, 255, 255, 255, 255, 255, 255, 255, // 176..183
    255, 255, 255, 255, 255, 255, 255, 255, // 184..191
    255, 255, 255, 255, 255, 255, 255, 255, // 192..199
    255, 255, 255, 255, 255, 255, 255, 255, // 200..207
    255, 255, 255, 255, 255, 255, 255, 255, // 208..215
    255, 255, 255, 255, 255, 255, 255, 255, // 216..223
    255, 255, 255, 255, 255, 255, 255, 255, // 224..231
    255, 255, 255, 255, 255, 255, 255, 255, // 232..239
    255, 255, 255, 255, 255, 255, 255, 255, // 240..247
    255, 255, 255, 255, 255, 255, 255, 255  // 248..255
};

__device__ __constant__ static const unsigned char d_basemap_compl[256] = { // complement base
    255, 255, 255, 255, 255, 255, 255, 255, // 0..7
    255, 255, 255, 255, 255, 255, 255, 255, // 8..15
    255, 255, 255, 255, 255, 255, 255, 255, // 16..23
    255, 255, 255, 255, 255, 255, 255, 255, // 24..31
    255, 255, 255, 255, 255, 255, 255, 255, // 32..39
    255, 255, 255, 255, 255, 255, 255, 255, // 40..47
    255, 255, 255, 255, 255, 255, 255, 255, // 48..55
    255, 255, 255, 255, 255, 255, 255, 255, // 56..63
    255, 3, 255, 2, 255, 255, 255, 1, // 64..71
    255, 255, 255, 255, 255, 255, 255, 255, // 72..79
    255, 255, 255, 255, 0, 3, 255, 255, // 80..87
    255, 255, 255, 255, 255, 255, 255, 255, // 88..95
    255, 3, 255, 2, 255, 255, 255, 1, // 96..103
    255, 255, 255, 255, 255, 255, 255, 255, // 104..111
    255, 255, 255, 255, 0, 3, 255, 255, // 112..119
    255, 255, 255, 255, 255, 255, 255, 255, // 120..127
    255, 255, 255, 255, 255, 255, 255, 255, // 128..135
    255, 255, 255, 255, 255, 255, 255, 255, // 136..143
    255, 255, 255, 255, 255, 255, 255, 255, // 144..151
    255, 255, 255, 255, 255, 255, 255, 255, // 152..159
    255, 255, 255, 255, 255, 255, 255, 255, // 160..167
    255, 255, 255, 255, 255, 255, 255, 255, // 168..175
    255, 255, 255, 255, 255, 255, 255, 255, // 176..183
    255, 255, 255, 255, 255, 255, 255, 255, // 184..191
    255, 255, 255, 255, 255, 255, 255, 255, // 192..199
    255, 255, 255, 255, 255, 255, 255, 255, // 200..207
    255, 255, 255, 255, 255, 255, 255, 255, // 208..215
    255, 255, 255, 255, 255, 255, 255, 255, // 216..223
    255, 255, 255, 255, 255, 255, 255, 255, // 224..231
    255, 255, 255, 255, 255, 255, 255, 255, // 232..239
    255, 255, 255, 255, 255, 255, 255, 255, // 240..247
    255, 255, 255, 255, 255, 255, 255, 255  // 248..255
};

// raw read is not a significant VRAM usage, no need for 2-bit encoding
// the majority VRAM usage is caused by minimizer (positions) etc...

extern Logger *logger;

// =================================================
// ================ CLASS PinnedCSR ================
// =================================================
    PinnedCSR::PinnedCSR(vector<string> &reads) {
        this->n_reads = reads.size();
        size_capacity = 0;
        for (string &read: reads) {
            size_capacity += read.length();
        } // about cudaHostAlloc https://zhuanlan.zhihu.com/p/188246455
        CUDA_CHECK(cudaHostAlloc((void**)(&reads_offs), (this->n_reads+1)*sizeof(T_CSR_cap), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void**)(&reads_CSR), size_capacity+1, cudaHostAllocDefault));
        char *cur_ptr = reads_CSR;
        T_CSR_cap *offs_ptr = reads_offs;
        *offs_ptr = 0;
        for (string &read: reads) {
            memcpy(cur_ptr, read.c_str(), read.length());
            cur_ptr += read.length();
            offs_ptr++;
            *offs_ptr = *(offs_ptr-1) + read.length();
        }
    }
    PinnedCSR::PinnedCSR(vector<ReadPtr> &reads, bool keep_original/*=true*/) { // for sorting CSR (order the pointers as the sorting result)
        this->n_reads = reads.size();
        size_capacity = 0;
        for (const ReadPtr &read_ptr: reads) {
            size_capacity += read_ptr.len;
        } // about cudaHostAlloc https://zhuanlan.zhihu.com/p/188246455
        cerr<<"Pinned reads = "<<n_reads<<" tot_sizes = "<<size_capacity<<"bytes"<<endl;
        CUDA_CHECK(cudaHostAlloc((void**)(&reads_offs), (this->n_reads+1)*sizeof(T_CSR_cap), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void**)(&reads_CSR), size_capacity+1, cudaHostAllocDefault));
        char *cur_ptr = reads_CSR;
        T_CSR_cap *offs_ptr = reads_offs;
        *offs_ptr = 0;
        for (const ReadPtr &read_ptr: reads) {
            memcpy(cur_ptr, read_ptr.read, read_ptr.len);
            cur_ptr += read_ptr.len;
            offs_ptr++;
            *offs_ptr = *(offs_ptr-1) + read_ptr.len;
        }
    }
    PinnedCSR::~PinnedCSR() {
        CUDA_CHECK(cudaFreeHost(reads_offs));
        CUDA_CHECK(cudaFreeHost(reads_CSR));
    }


struct T_d_data {
    _in_ T_read_cnt reads_cnt;
    
    // Raw reads
    _in_ _out_ char *d_reads; // will be also used to store HPC reads (if HPC is enabled)
    _in_ T_CSR_cap *d_read_offs; // reads are in CSR format so offset array is required
    _in_ _out_ T_read_len *d_read_len;  // len == len(d_read_offs)  int
    
    // HPC reads info
    T_read_len *d_hpc_orig_pos;         // len == len(d_reads)      size_t  base original pos **in a read** (not in CSR)
    
    // Minimizers
    _out_ T_minimizer *d_minimizers;    // len == len(d_reads)      size_t
    _out_ T_kvalue *d_mm_pos;           // len == len(d_reads)      u_char  minimizer position in each window
    _out_ char *d_mm_strand;            // len == len(d_reads)      char    0 for forward, 1 for reverse complement, -1 for f==rc
    _out_ T_read_len *d_superkmer_offs; // len == len(d_reads)      int     supermer_offs **in a read**
}; // device data

struct T_h_data {
    _in_ T_read_cnt reads_cnt;
    
    // Raw reads
    _in_ _out_ char *reads;  // will be also used to store HPC reads (if HPC is enabled)
    _in_ T_CSR_cap *reads_offs;         // reads are in CSR format so offset array is required
    _in_ _out_ T_read_len *read_len;    // len == len(d_read_offs) int
    
    // HPC reads info
    T_read_len *hpc_orig_pos;  // len == len(d_reads)      size_t  base original pos **in a read** (not in CSR)
    
    // Minimizers
    _out_ T_minimizer *minimizers;      // len == len(d_reads)      size_t
    _out_ T_read_len *superkmer_offs;   // len == len(d_reads)      int     supermer_offs **in a read**
}; // host data


__global__ void GPU_HPCEncoding (
    _in_ T_read_cnt d_reads_cnt, _out_ T_read_len *d_read_len, 
    _in_ _out_ char *d_reads, _in_ T_CSR_cap *d_read_offs, 
    bool HPC, _out_ T_read_len *d_hpc_orig_pos=NULL) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_t = blockDim.x * gridDim.x;
    if (!HPC) { // only calculate read_len in global memory (optional but essential for HPC=true)
        for (T_read_cnt rid = tid; rid < d_reads_cnt; rid += n_t) {
            d_read_len[rid] = d_read_offs[rid+1] - d_read_offs[rid];
        }
        __syncthreads();
        return;
    }
    
    for (T_read_cnt rid = tid; rid < d_reads_cnt; rid += n_t) {
        T_read_len read_len = d_read_offs[rid+1] - d_read_offs[rid];
        T_read_len last_idx = 0, hpc_arr_idx = d_read_offs[tid], j;
        d_hpc_orig_pos[hpc_arr_idx] = 0;
        for (T_read_len i = 1; i < read_len; i++) {
            j = i + d_read_offs[rid];
            last_idx += (i-last_idx) * (d_reads[j] != d_reads[j-1]);
            hpc_arr_idx += (d_reads[j] != d_reads[j-1]);
            d_hpc_orig_pos[hpc_arr_idx] = last_idx;
            d_reads[hpc_arr_idx] = d_reads[j];
        }
        d_read_len[rid] = hpc_arr_idx + 1 - d_read_offs[rid];
    }
    return;
}

// ======== Minimizer Functions ========
// traditional minimizer
__device__ __forceinline__ bool mm_filter(T_minimizer mm, int p) {
    // return mm%101>80; // 20.36
    // return ((mm >> ((p-3)*2)) != 0) /*AAA*/ & (mm >> ((p-3)*2) != 0b000100) /*ACA*/; // 19.94
    // return ((mm >> (p-2)*2) & 0b11) + ((mm >> (p-3)*2) & 0b11) + ((mm >> (p-1)*2) & 0b11); // 20.03
    // return (mm >> (p-3)*2) * ((mm >> (p-5)*2) & 0b111111); // 20.02
    // return ((mm >> ((p-3)*2)) != 0) /*AAA*/ & (mm >> ((p-3)*2) != 0b000100) /*ACA*/ & (mm >> ((p-3)*2) != 0b001000); // 19.92
    // int i=0;
    // int s=0;
    // for (i=1; i<3; i++) {
    //     s += (mm >> ((p-2)*2)) > (mm>>((p-2-i))&0b1111);
    // }
    // return s==0;
    return true;
}
// new design: 2nd/3rd不都为a
__device__ __forceinline__ bool new_filter(T_minimizer mm, int p) {
    return ((mm >> (p-2)*2) & 0b11) + ((mm >> (p-3)*2) & 0b11);
}
__device__ __forceinline__ bool new_filter2(T_minimizer mm, int p) {
    return ((mm >> ((p-3)*2)) != 0) /*AAA*/ & (mm >> ((p-3)*2) != 0b000100) /*ACA*/ & (mm >> ((p-3)*2) != 0b001000);
}
// KMC2 signature
__device__ bool sign_filter(T_minimizer mm, int p) {
    T_minimizer t = mm;
    bool flag = true;
    for (int ii = 0; ii < p-2; ii ++) {
        flag *= ((t & 0b1111) != 0);
        t = t >> 2;
    }
    // printf("%d Minimizer: %x\n", flag & ((mm >> ((p-3)*2)) != 0) /*AAA*/ & (mm >> ((p-3)*2) != 0b000100), mm);
    return flag & ((mm >> ((p-3)*2)) != 0) /*AAA*/ & (mm >> ((p-3)*2) != 0b000100) /*ACA*/;
}
/*
 * [INPUT]  data.reads in [(Read#0), (Read#1)...]
 * [OUTPUT] data.minimizers in [(Read#0)[mm1, mm?, mm?, ...], (Read#1)...]
 * all thread do one read at the same time with coalesced global memory access
 * TODO: make a 32-bit minimizer version and compare the performance
 */
__global__ void GPU_GenMinimizer( // TODO: reverse complement of minimizer
    _in_ T_read_cnt d_reads_cnt, _in_ T_read_len *d_read_len, 
    _in_ char *d_reads, _in_ T_CSR_cap *d_read_offs, 
    _out_ T_minimizer *d_minimizers, 
    int K_kmer, int P_minimizer) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_t = blockDim.x * gridDim.x;
    int i, j, cur_kmer_i;
    T_minimizer mm_mask = T_MM_MAX >> (sizeof(T_minimizer)*8 - 2*P_minimizer);
    T_minimizer mm_set; // selected minimizer
    T_minimizer mm, new_mm, mm_rc, new_mm_rc; // rc for reverse complement
    
    bool mm_check; // whether is a legal minimizer/signature (filtered by mm_filter)

    for (i=0; i<d_reads_cnt; i++) {
        char *read = &d_reads[d_read_offs[i]]; // current read
        T_minimizer *minimizer_saving = &(d_minimizers[d_read_offs[i]]);
        T_read_len len = d_read_len[i];
        for (cur_kmer_i=tid; cur_kmer_i <= len-K_kmer; cur_kmer_i+=n_t) { // Coalesced Access
            // gen the first p-mer:
            new_mm = 0;
            for (j = cur_kmer_i; j < cur_kmer_i + P_minimizer; j++) {
                new_mm = (new_mm << 2) | d_basemap[*(read+j)];
            }
            mm_check = FILTER_KERNEL(new_mm, P_minimizer);
            mm = new_mm * mm_check + mm_mask * (!mm_check); // if not a minimizer, let it be maximal (no minimizer can be maximal because canonical)
            
            // gen the first RC p-mer:
            new_mm_rc = 0;
            for (j = cur_kmer_i + P_minimizer - 1; j >= cur_kmer_i; j--) {
                new_mm_rc = (new_mm_rc << 2) | d_basemap_compl[*(read+j)];
            }
            mm_check = FILTER_KERNEL(new_mm_rc, P_minimizer);
            mm_rc = new_mm_rc * mm_check + mm_mask * (!mm_check);

            mm_set = (mm_rc < mm) * mm_rc + (mm_rc >= mm) * mm;////////////
            
            // gen the next p-mers:
            for (j = cur_kmer_i + P_minimizer; j < cur_kmer_i + K_kmer; j++) {
                // gen new minimizers
                new_mm = ((new_mm << 2) | d_basemap[*(read+j)]) & mm_mask;
                new_mm_rc = (new_mm_rc >> 2) | (d_basemap_compl[*(read+j)] << (P_minimizer*2-2));
                // check new minimizers
                mm_check = FILTER_KERNEL(new_mm, P_minimizer);
                mm = new_mm * mm_check + mm * (!mm_check);
                mm_check = FILTER_KERNEL(new_mm_rc, P_minimizer);
                mm_rc = new_mm_rc * mm_check + mm_rc * (!mm_check);
                // set the best minimizer
                mm_set = (mm_set < mm) * mm_set + (mm_set >= mm) * mm;
                mm_set = (mm_set < mm_rc) * mm_set + (mm_set >= mm_rc) * mm_rc;//////////
            }
            minimizer_saving[cur_kmer_i] = mm_set;
            // printf("mmset %x\n",mm_set);
        }
    }
    // if (tid==0) printf("--kernel end");
    return;
}


/* [INPUT]  data.minimizers in [[mm1, mm1, mm2, mm3, ...], ...]
 * [OUTPUT] data.superkmer_offs in [[0, 2, 3, ...], ...]
*/
__global__ void GPU_GenSKM(
    _in_ T_read_cnt d_reads_cnt, _in_ T_read_len *d_read_len, 
    _in_ T_CSR_cap *d_read_offs, 
    _in_ T_minimizer *d_minimizers,
    _out_ T_read_len *d_superkmer_offs,
    int K_kmer, int P_minimizer) {
        
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_t = blockDim.x * gridDim.x;

    for (T_read_cnt rid = tid; rid < d_reads_cnt; rid += n_t) {
        T_read_len len = d_read_len[rid];                               // current read length
        T_minimizer *minimizers = &(d_minimizers[d_read_offs[rid]]);   // minimizer list pointer
        T_read_len *skm = &d_superkmer_offs[d_read_offs[rid]];         // superkmer list pointer
        T_read_len last_skm_pos = 0, skm_count = 0;                      // position of the last minimizer; superkmer count
        skm[0] = 0;
        for (T_read_len i = 1; i <= len-K_kmer; i++) {
            skm_count += (minimizers[i] != minimizers[i-1]); // current minimizer != last minimizer, new skm generated
            last_skm_pos = (minimizers[i] == minimizers[i-1]) * last_skm_pos + (minimizers[i] != minimizers[i-1]) * i;
            skm[skm_count] = last_skm_pos;
        }
        // assert(len!=0);
        skm[skm_count+1] = len-K_kmer+1;
        skm[skm_count+2] = 0;
    }
    return;
}

/// @brief Set device CSR offsets begin from 0.
/// @param d_reads_cnt 
/// @param d_read_offs 
/// @param add [0] for setting to zero, [positive] value for adding back
/// @return
__global__ void MoveOffset(_in_ T_read_cnt d_reads_cnt, _in_ _out_ T_CSR_cap *d_read_offs, long long add=0) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_t = blockDim.x * gridDim.x;
    add = add - (add==0) * d_read_offs[0];
    for (T_read_cnt rid = tid; rid <= d_reads_cnt; rid += n_t) {
        d_read_offs[rid] += add;
    }
    return;
}


// post-GPU task functions
__host__ void SaveSKMs (CountTask task) {
    if (task == CountTask::SKMPartition) {

    } else if (task == CountTask::SKMPartWithPos) {

    }
}
__host__ void CalcSKMPartSize (T_read_cnt reads_cnt, T_read_len *superkmer_offs, 
    T_CSR_cap *reads_offs, T_minimizer *minimizers, 
    int n_partitions, int k, atomic<size_t> part_sizes[]) {
    int i;
    T_CSR_cap skm_offs_idx;
    T_CSR_cap offs_move = reads_offs[0];
    T_read_len superkmer_len;
    for (i=0; i<reads_cnt; i++) {
        skm_offs_idx = reads_offs[i] - offs_move + 1; // [1, n] 0~1 1~2 ... n-1~n
        while (superkmer_offs[skm_offs_idx] != 0) {
            superkmer_len = superkmer_offs[skm_offs_idx]-1 - superkmer_offs[skm_offs_idx-1] + k;
            part_sizes[minimizers[superkmer_offs[skm_offs_idx-1] + reads_offs[i] - offs_move] % n_partitions] += superkmer_len;
            skm_offs_idx++;
        }
    }
}

__host__ void GPUReset(int did) {
    // do not call it after host malloc
    CUDA_CHECK(cudaSetDevice(did));
    CUDA_CHECK(cudaDeviceReset());
    CUDA_CHECK(cudaDeviceSynchronize());
    return;
}

// provide pinned_reads from the shortest to the longest read
__host__ void GenSuperkmerGPU (PinnedCSR &pinned_reads, 
    int K_kmer, int P_minimizer, bool HPC, CUDAParams gpars, CountTask task,
    int SKM_partitions, atomic<size_t> skm_part_sizes[]) {
    
    int time_all=0, time_filter=0;

    CUDA_CHECK(cudaSetDevice(gpars.device_id));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaStream_t streams[gpars.n_streams];
    T_d_data gpu_data[gpars.n_streams];
    T_h_data host_data[gpars.n_streams];
    T_CSR_cap batch_size[gpars.n_streams];
    T_read_cnt bat_beg_read[gpars.n_streams];//, bat_end_read[gpars.n_streams];

    int i, started_streams;
    for (i=0; i<gpars.n_streams; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    
    T_read_cnt items_per_stream = gpars.NUM_BLOCKS_PER_GRID * gpars.NUM_THREADS_PER_BLOCK;
    T_read_cnt cur_read = 0, end_read;
    i = 0; // i for which stream
    while (cur_read < pinned_reads.n_reads) {

        // TODO: check if last round is finished if CPU postprocess func is async

        for (i = 0; i < gpars.n_streams && cur_read < pinned_reads.n_reads; i++, cur_read += items_per_stream) {
            bat_beg_read[i] = cur_read;
            end_read = min(cur_read + items_per_stream, pinned_reads.n_reads); // the last read in this stream batch
            // bat_end_read[i] = end_read;
            host_data[i].reads_cnt = gpu_data[i].reads_cnt = end_read-cur_read;
            batch_size[i] = pinned_reads.reads_offs[end_read] - pinned_reads.reads_offs[cur_read]; // read size in bytes
            // gpu_data[i].offs_move = pinned_reads.reads_offs[cur_read];
            logger->log("STREAM "+to_string(i)+" Batch read count="+to_string(gpu_data[i].reads_cnt));

            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            // 1. cudaMalloc (5000 reads / GB)
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_reads), sizeof(char) * (batch_size[i]+1), streams[i]));//                // 8192 threads(reads) * 20 KB/read     = 160MB VRAM
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_read_offs), sizeof(T_CSR_cap) * (gpu_data[i].reads_cnt+1), streams[i])); // 8192 threads(reads) * 8 B/read       =  64MB VRAM
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_read_len), sizeof(T_read_len) * (gpu_data[i].reads_cnt), streams[i]));   // 8192 threads(reads) * 4 B/read       =  32MB VRAM
            if (HPC) {// cost a lot VRAM
                CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_hpc_orig_pos), sizeof(T_read_len) * (batch_size[i]), streams[i]));   // 8192 threads(reads) * 20K * 4B/read  = 640MB VRAM
            } else {
                gpu_data[i].d_hpc_orig_pos = NULL;
            }
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_minimizers), sizeof(T_minimizer) * (batch_size[i]), streams[i]));        // 8192 threads(reads) * 20K * 4B/read  = 640MB VRAM
            CUDA_CHECK(cudaMallocAsync((void**) &(gpu_data[i].d_superkmer_offs), sizeof(T_read_len) * (batch_size[i]), streams[i]));     // 8192 threads(reads) * 20K * 4B/read  = 640MB VRAM
            
            // 2. cudaMemcpy (H2D) // TODO: async
            CUDA_CHECK(cudaMemcpyAsync(gpu_data[i].d_reads, &(pinned_reads.reads_CSR[pinned_reads.reads_offs[cur_read]]), batch_size[i], cudaMemcpyHostToDevice, streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(gpu_data[i].d_read_offs, &(pinned_reads.reads_offs[cur_read]), sizeof(T_CSR_cap) * (gpu_data[i].reads_cnt+1), cudaMemcpyHostToDevice, streams[i]));
            
            // 3. GPU Computing
            WallClockTimer wct;
            MoveOffset<<<gpars.NUM_BLOCKS_PER_GRID, gpars.NUM_THREADS_PER_BLOCK, 0, streams[i]>>>(
                gpu_data[i].reads_cnt, gpu_data[i].d_read_offs, 0
            );
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            
            GPU_HPCEncoding<<<gpars.NUM_BLOCKS_PER_GRID, gpars.NUM_THREADS_PER_BLOCK, 0, streams[i]>>>(
                gpu_data[i].reads_cnt, gpu_data[i].d_read_len, 
                gpu_data[i].d_reads, gpu_data[i].d_read_offs, 
                HPC, gpu_data[i].d_hpc_orig_pos
            );
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));

            WallClockTimer wct2;
            // cudaEventRecord(start, streams[i]);
            GPU_GenMinimizer<<<gpars.NUM_BLOCKS_PER_GRID, gpars.NUM_THREADS_PER_BLOCK, 0, streams[i]>>>(
                gpu_data[i].reads_cnt, gpu_data[i].d_read_len,
                gpu_data[i].d_reads, gpu_data[i].d_read_offs,
                gpu_data[i].d_minimizers, 
                K_kmer, P_minimizer
            );
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            time_filter += wct2.stop(true);
            // cudaEventRecord(stop, streams[i]);
            // cudaEventElapsedTime(&time_tmp, start, stop); time_all += time_tmp; time_filter += time_tmp;

            GPU_GenSKM<<<gpars.NUM_BLOCKS_PER_GRID, gpars.NUM_THREADS_PER_BLOCK, 0, streams[i]>>>(
                gpu_data[i].reads_cnt, gpu_data[i].d_read_len,
                gpu_data[i].d_read_offs, 
                gpu_data[i].d_minimizers,
                gpu_data[i].d_superkmer_offs,
                K_kmer, P_minimizer
            );
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            time_all += wct.stop(true);
            // pinned_reads.reads_offs[cur_read]
        }
        started_streams = i;
        for (i = 0; i < started_streams; i++) {
            // Malloc on host for temporary result storage
            // TODO: add if on task to indicate whether to new and D2H
            if (HPC) {
                host_data[i].hpc_orig_pos = new T_read_len[batch_size[i]];
                host_data[i].read_len = new T_read_len[gpu_data[i].reads_cnt];
            }
            host_data[i].minimizers = new T_minimizer[batch_size[i]];
            host_data[i].reads = &(pinned_reads.reads_CSR[pinned_reads.reads_offs[bat_beg_read[i]]]); // used pinned memory to store the output
            host_data[i].reads_offs = &(pinned_reads.reads_offs[bat_beg_read[i]]); // used pinned memory to store the output, !! offs not begin from 0
            host_data[i].superkmer_offs = new T_read_len[batch_size[i]];

            // D2H memory copy
            if (HPC) {
                CUDA_CHECK(cudaMemcpyAsync(host_data[i].hpc_orig_pos, gpu_data[i].d_hpc_orig_pos, sizeof(T_read_len) * batch_size[i], cudaMemcpyDeviceToHost, streams[i]));
                CUDA_CHECK(cudaMemcpyAsync(host_data[i].read_len, gpu_data[i].d_read_len, sizeof(T_read_len) * host_data[i].reads_cnt, cudaMemcpyDeviceToHost, streams[i]));
            }
            CUDA_CHECK(cudaMemcpyAsync(host_data[i].minimizers, gpu_data[i].d_minimizers, sizeof(T_minimizer) * batch_size[i], cudaMemcpyDeviceToHost, streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(host_data[i].reads, gpu_data[i].d_reads, sizeof(char) * batch_size[i], cudaMemcpyDeviceToHost, streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(host_data[i].superkmer_offs, gpu_data[i].d_superkmer_offs, sizeof(T_read_len) * batch_size[i], cudaMemcpyDeviceToHost, streams[i]));

            // Free device memory
            if (HPC) CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_hpc_orig_pos, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_minimizers, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_read_len, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_reads, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_read_offs, streams[i]));
            CUDA_CHECK(cudaFreeAsync(gpu_data[i].d_superkmer_offs, streams[i]));
            
            // Wait for CUDA
            CUDA_CHECK(cudaStreamSynchronize(streams[i])); // move this into async post_proc_func
            logger->log("GPU Done"+to_string(i));
            // CPU post-process // TODO: async & use bind to pass the post process function
            CalcSKMPartSize(host_data[i].reads_cnt, host_data[i].superkmer_offs, host_data[i].reads_offs, host_data[i].minimizers, SKM_partitions, K_kmer, skm_part_sizes);
            
            // clean host variables
            // (TODO: 如果post-process 用 async, 则free放在post-process函数里，且保证host_data非引用传递给async proc func以防下一轮更新)
            if (HPC) {
                delete [] host_data[i].hpc_orig_pos;
                delete [] host_data[i].read_len;
            }
            delete [] host_data[i].minimizers;
            delete [] host_data[i].superkmer_offs;
            // logger->log("Batch Done "+to_string(i));
        }
    }
    logger->log("FILTER KERNEL: " STR(FILTER_KERNEL) "");
    logger->log("Kernel Functions Time: ALL = "+to_string(time_all)+"ms FILTER = "+to_string(time_filter)+"ms");
}