// #define TIMER

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
// #include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "types.h"
#include "superkmers.hpp"
#ifdef TIMER
#include "utilities.hpp" // timer
#endif
#include <vector>
using namespace std;

// struct differentfromprev {
//     differentfromprev() {}
//     __host__ __device__
//         bool operator()(const T_kmer& x, const T_kmer& y) const { 
//             return x!=y;
//         }
// };
struct sameasprev {
    sameasprev() {}
    __host__ __device__
        bool operator()(const T_kmer& x, const T_kmer& y) const { 
            return x==y;
        }
};
struct canonicalkmer {
    canonicalkmer() {}
    __host__ __device__
        T_kmer operator()(const T_kmer& x, const T_kvalue k) const {
            T_kmer x1 = ~x, res=0;
            for (T_kvalue i=0; i<k; i++) {
                res = (res << 2) | (x1 & 0b11);
                x1 = x1 >> 2;
            }
            return res < x ? res : x;
        }
};
struct replaceidx {
    replaceidx() {}
    __host__ __device__
        T_read_len operator()(const T_read_len& x, const T_read_len& y) const {
            return x*y;
        }
};
struct is_zero {
    __host__ __device__
        bool operator()(const T_read_len x)
        {
            return x==0;
        }
};
__host__ size_t kmc_counting_GPU (T_kvalue k,
                               SKMStore &skms_store, int gpuid,
                               unsigned short kmer_min_freq, unsigned short kmer_max_freq,
                               _out_ vector<T_kmc> &kmc_result_curthread) {
    // using CUDA Thrust
    // size_t est_kmer = skms_store.tot_size - skms_store.skms.size_approx() * (k-1);
    // size_t db_skm = skms_store.skms.size_approx();
    // size_t est_skm = 0;
    
    // 0. Extract kmers from SKMStore: (~65-85% time)
    #ifdef TIMER
    double wcts[10];
    WallClockTimer wct0;
    #endif
    
    if (skms_store.tot_size == 0) return 0;
    thrust::host_vector<T_kmer> kmers_h;
    size_t tot_kmers;
    T_skm_len skm_len;
    T_kmer kmer_mask = T_kmer(0xffffffffffffffff>>(64-k*2));
    T_kmer kmer;//, kmer1;
    // V1:
    // CRead<T_skm_len> skm;
    // while (skms_store.try_pop_skm(skm)) {
    //     skm_len = skm.length();
    //     // first kmer
    //     // kmer = skm.get_2bit_kmer_V2(0, k, kmer_mask); // 2.7-2.8s
    //     kmer = skm.get_2bit_kmer(0, k); // 2.8-2.9s
    //     kmers_h.push_back(kmer);
    //     // latter kmers
    //     for (T_skm_len j=k; j<skm_len; j++) {
    //         kmer = ((kmer << 2) | skm.get_2bit(j)) & kmer_mask;
    //         kmers_h.push_back(kmer);
    //     }
    // }
    // V2 - bulk: 2.5-2.7s
    const size_t BULK_SIZE = 10000; // 100k too large
    CRead<T_skm_len> skm_bulk[BULK_SIZE];
    size_t pop_cnt;
    pop_cnt = skms_store.try_pop_skm_bulk(skm_bulk, BULK_SIZE);
    while (pop_cnt) {
        for (size_t i=0; i<pop_cnt; i++) {
            skm_len = skm_bulk[i].length();
            // first kmer
            kmer = skm_bulk[i].get_2bit_kmer_V2(0, k, kmer_mask);
            // kmer1 = skm.get_2bit_kmer(0, k);
            kmers_h.push_back(kmer);
            // latter kmers
            for (T_skm_len j=k; j<skm_len; j++) {
                kmer = ((kmer << 2) | skm_bulk[i].get_2bit(j)) & kmer_mask;
                kmers_h.push_back(kmer);
            }
        }
        pop_cnt = skms_store.try_pop_skm_bulk(skm_bulk, BULK_SIZE);
    }
    
    thrust::device_vector<T_kmer> kmers_d(kmers_h);
    tot_kmers = kmers_d.size();
    // cerr<<est_kmer<<"|"<<db_skm<<"|"<<est_skm<<"|"<<tot_kmers<<endl;
    
    // 1. convert to canonical kmers (~3-8% time)
    #ifdef TIMER
    wcts[0] = wct0.stop();
    WallClockTimer wct1;
    #endif
    thrust::constant_iterator<T_kvalue> ik(k);
    thrust::transform(thrust::device, kmers_d.begin(), kmers_d.end(), ik, kmers_d.begin(), canonicalkmer());
    
    // 2. sort: [ABCBBAC] -> [AABBBCC] (kmers_d) (~5-15% time)
    #ifdef TIMER
    wcts[1] = wct1.stop();
    WallClockTimer wct2;
    #endif
    thrust::sort(thrust::device, kmers_d.begin(), kmers_d.end()/*, thrust::greater<T_kmer>()*/);
    thrust::host_vector<T_kmer> sorted_kmers_h = kmers_d;
    
    // 3. find changes: [AABBBCC] -> [0,0,1,0,0,1,0] (comp_vec_d)
    // thrust::device_vector<bool> comp_vec_d(kmers_d.size());
    // thrust::transform(thrust::device, kmers_d.begin()+1 /*x beg*/, kmers_d.end() /*x end*/, kmers_d.begin()/*y beg*/, comp_vec_d.begin()+1/*res beg*/, differentfromprev());
    // comp_vec_d[0] = 1; //
    // int distinct_kmer_cnt = thrust::reduce(thrust::device, comp_vec_d.begin(), comp_vec_d.end()) + 1;
    
    // 3. find changes: [AABBBCC] -> [0,1,0,1,1,0,1] (same_flag_d)
    #ifdef TIMER
    wcts[2] = wct2.stop();
    WallClockTimer wct3;
    #endif
    thrust::device_vector<bool> same_flag_d(kmers_d.size());
    thrust::transform(thrust::device, kmers_d.begin()+1 /*x beg*/, kmers_d.end() /*x end*/, kmers_d.begin()/*y beg*/, same_flag_d.begin()+1/*res beg*/, sameasprev());
    same_flag_d[0] = 0; //
    
    // 4. remove same idx: [0123456] [0101101] -> [0,2,5] (idx_d)
    #ifdef TIMER
    wcts[3] = wct3.stop();
    WallClockTimer wct4;
    #endif
    thrust::device_vector<T_read_len> idx_d(kmers_d.size());
    thrust::sequence(thrust::device, idx_d.begin(), idx_d.end());
    auto new_end_d = thrust::remove_if(thrust::device, idx_d.begin(), idx_d.end(), same_flag_d.begin(), thrust::identity<bool>()); // new_end_d is an iterator
    
    // 4. replace with index: [0,0,1,0,0,1,0] -> [0,0,2,0,0,5,0] (comp_vec_d)
    // thrust::device_vector<T_read_len> seq_d(kmers_d.size());
    // thrust::sequence(thrust::device, seq_d.begin(), seq_d.end());
    // thrust::transform(thrust::device, comp_vec_d.begin() /*x*/, comp_vec_d.end(), seq_d.begin()/*y*/, comp_vec_d.begin()/*res*/, replaceidx());

    // // 5. skip repeats: [0,0,2,0,0,5,0] -> [0,2,5] (comp_vec_d)
    // auto new_end_d = thrust::remove_if(thrust::device, comp_vec_d.begin(), comp_vec_d.end(), is_zero());

    // 5. copy device_vector back to host_vector
    #ifdef TIMER
    wcts[4] = wct4.stop();
    WallClockTimer wct5;
    #endif
    thrust::host_vector<T_read_len> idx_h(idx_d.begin(), new_end_d);
    idx_h.push_back(tot_kmers); // [0,2,5] -> [0,2,5,7] A2 B3 C2
    
    #ifdef TIMER
    wcts[5] = wct5.stop();
    WallClockTimer wct6;
    #endif
    size_t total_kmer_cnt = 0;
    int i;
    T_kmer_cnt cnt;
    for(i=0; i<idx_h.size()-1; i++) {
        cnt = idx_h[i+1]-idx_h[i] > MAX_KMER_CNT ? MAX_KMER_CNT : idx_h[i+1]-idx_h[i];
        total_kmer_cnt += idx_h[i+1]-idx_h[i];
        // Add kmer-cnt to result vector:
        // if (cnt >= kmer_min_freq && cnt <= kmer_max_freq) {
        //     kmc_result_curthread.push_back({sorted_kmers_h[idx_h[i]], cnt});
        // }
    }
    #ifdef TIMER
    wcts[6] = wct6.stop();
    cout<<wcts[0]<<"\t"<<wcts[1]<<"\t"<<wcts[2]<<"\t"<<wcts[3]<<"\t"<<wcts[4]<<"\t"<<wcts[5]<<"\t"<<wcts[6]<<endl;
    #endif
    // return total_kmer_cnt; // total kmer
    return idx_h.size()-1; // total distinct kmer
}

// kmc_counting_GPU (skms, gpuid, kmer_min_freq, kmer_max_freq, kmc_result_curthread);