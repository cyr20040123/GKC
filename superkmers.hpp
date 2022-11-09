#ifndef _SUPERKMERS_HPP
#define _SUPERKMERS_HPP

#define COUNT_SKM_SIZE
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <atomic>
#include "concurrent_queue.h"
#include "types.h"
using namespace std;

static const unsigned char basemap[256] = {
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
static char bit2char[4] = {'A', 'C', 'G', 'T'};

template<typename T_len, typename T_attr=bool>
class CRead {
public:
    unique_ptr<byte[]> compressed_read;
    T_len len = 0;
    T_attr attr;
    CRead() {}
    CRead(T_len _len, const byte* _compressed, T_attr _attr=NULL) {
        len = _len;
        compressed_read = make_unique<byte[]>(compressed_length(len));//
        memcpy(compressed_read.get(), _compressed, compressed_length(len));
        attr = _attr;
    }
    CRead(const char* raw_read, int _len=-1, T_attr _attr=NULL) {
        len = _len==-1 ? strlen(raw_read) : _len;
        compressed_read = make_unique<byte[]>(compressed_length(len));//
        for (int i=0; i<len; i++) {
            compressed_read[i/4] = (compressed_read[i/4] << 2) | (basemap[raw_read[i]] & 0b11);
        }
        if (len%4!=0) compressed_read[len/4] = compressed_read[len/4] << (2*(4-len%4));
        attr = _attr;
    }
    CRead(string raw_read, T_attr _attr=NULL) : CRead(raw_read.c_str(), raw_read.length(), _attr) {}
    CRead(const CRead &other) : CRead(other.len, other.compressed_read.get(), other.attr) {}

    CRead &operator=(const CRead &other) {
        if (this == &other) return *this;
        byte* tmp = compressed_read.release();
        if (tmp!=nullptr) delete [] tmp;

        // CRead(other); :
        len = other.len;
        compressed_read = make_unique<byte[]>(compressed_length(len));
        memcpy(compressed_read.get(), other.compressed_read.get(), compressed_length(len));
        attr = other.attr;
        
        return *this;
    }
    string get() {
        unique_ptr<char[]>tmp = make_unique<char[]>(len+1);
        for (int i=0; i<len; i++) tmp[i] = this->get(i);
        tmp[len] = 0;
        string res(tmp.get());
        return res;
    }
    string get(int beg, int end) {
        unique_ptr<char[]>tmp = make_unique<char[]>(end-beg+1);
        for (int i=beg; i<end; i++) tmp[i-beg] = this->get(i);
        tmp[end-beg] = 0;
        string res(tmp.get());
        return res;
    }
    char get(int idx) {
        return bit2char[(compressed_read[idx/4] >> (2*(3-idx%4))) & 0b11];
    }
    char get_2bit(int idx) {
        return (compressed_read[idx/4] >> (2*(3-idx%4))) & 0b11;
    }
    T_kmer get_2bit_kmer(int beg, int k) {
        T_kmer res = 0;
        for (int i=beg; i<beg+k; i++) {
            res = (res << 2) | this->get_2bit(i);
        }
        return res;
    }
    T_kmer get_2bit_kmer_V2(int beg, int k, T_kmer kmer_mask) {
        T_kmer res = 0;
        int i;
        // beg = 3 k = 16 (3~19)
        // fetch 0~8
        // fetch 8~16
        for (i = beg; i <= beg+k-4; i += 4) {
            res = (res << 8) | compressed_read[i/4];
        }
        for (i = (i>>2)<<2; i < beg+k; i++) {
            res = (res << 2) | this->get_2bit(i);
        }
        return res & kmer_mask;
    }
    // const char* serialize(_out_ T_len &slen) {
    //     slen = compressed_length(len);
    //     return compressed_read.get();
    // }
    T_len length() {return len;}
    T_attr get_attr() {return attr;}
    static inline T_len compressed_length(T_len _len) {return (_len+3)/4;}
};

class SKMStore {
private:
    
public:
    moodycamel::ConcurrentQueue<CRead<T_skm_len>> skms;
    #ifdef COUNT_SKM_SIZE
    atomic<size_t> tot_size{0};
    #endif
    SKMStore () {}
    ~SKMStore () {}
    void add_skm (const char* beg, T_skm_len len) {
        assert(skms.enqueue(CRead<T_skm_len>(beg, len))); // exit if enqueue fails
        #ifdef COUNT_SKM_SIZE
        tot_size += len;
        // tot_size += (len+3)/4;
        #endif
    }
    bool try_pop_skm (CRead<T_skm_len> &skm) {
        return skms.try_dequeue(skm);
    }
    size_t try_pop_skm_bulk (CRead<T_skm_len> *skm_arr, size_t cnt) {
        return skms.try_dequeue_bulk(skm_arr, cnt);
    }
    // T_kmer* extract_kmers (T_kvalue k) {
    //     size_t n_skms = skms.size_approx(); // only call synchronously after skm generation finished
    //     T_kmer *kmers = new T_kmer[xxx_compressed_size - n_skms * (k-1)];
    //     //... device vector
    //     return kmers;
    // }
    static void add_skms (vector<SKMStore*> &skm_partitions, T_kvalue k, T_read_cnt reads_cnt, const char *csr_reads, T_CSR_cap *reads_offs, T_read_len *superkmer_offs, T_minimizer *minimizers) {
        int n_partitions = skm_partitions.size();
        T_read_cnt i;
        int part;

        T_CSR_cap skm_offs_idx, skm_beg_idx;
        T_CSR_cap offs_move = reads_offs[0];
        T_read_len superkmer_len;
        for (i=0; i<reads_cnt; i++) {
            skm_offs_idx = reads_offs[i] - offs_move + 1; // [1, n] 0~1 1~2 ... n-1~n
            while (superkmer_offs[skm_offs_idx] != 0) {
                superkmer_len = superkmer_offs[skm_offs_idx] - superkmer_offs[skm_offs_idx-1] + k-1;
                skm_beg_idx = superkmer_offs[skm_offs_idx-1] + reads_offs[i] - offs_move;
                part = minimizers[skm_beg_idx] % n_partitions;
                skm_partitions[part]->add_skm(&(csr_reads[skm_beg_idx]), superkmer_len);
                skm_offs_idx++;
            }
        }
        return;
    }
};

// int main(int argc, const char **argvs) {
//     int n_partitions = atoi(argvs[1]);
//     int i;
//     vector<SKMPartition*> skm_part_vec;
//     for (i=0; i<n_partitions; i++) skm_part_vec.push_back(new SKMPartition());
    

//     for (i=0; i<n_partitions; i++) delete skm_part_vec[i];
//     return 0;
// }

#endif
