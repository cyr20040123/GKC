#ifndef _TYPES_H
#define _TYPES_H

#define _in_
#define _out_
#define _tmp_

#include <cstdlib> // for size_t

typedef size_t T_CSR_cap;
typedef int T_CSR_count;

typedef unsigned short T_skm_len;

typedef int T_read_len;
typedef int T_read_cnt;
// typedef size_t T_CSR_cap;
typedef unsigned int T_minimizer; // support minimizer with max length = 16
const T_minimizer T_MM_MAX = (T_minimizer)(0xffffffffffffffff);
typedef unsigned char T_kvalue;

struct ReadPtr {
    const char* read;
    T_read_len len;
};

typedef unsigned long long T_kmer;
typedef unsigned long long T_2bitkmer;

typedef unsigned char byte;

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

typedef unsigned int T_kmer_cnt;
const T_kmer_cnt MAX_KMER_CNT = T_kmer_cnt(0xffffffffffffffff);
struct T_kmc{
    T_kmer kmer;
    T_kmer_cnt cnt;
};

#endif