#ifndef _TYPES_H
#define _TYPES_H

#define _in_
#define _out_
#define _tmp_

#include <cstdlib>

typedef size_t T_CSR_cap;
typedef int T_CSR_count;

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

typedef unsigned char Byte;

#endif