#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
// #include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "types.h"
#include "superkmers.hpp"
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

    // 0. Extract kmers from SKMStore:
    thrust::host_vector<T_kmer> kmers_h;
    size_t tot_kmers;
    T_skm_len skm_len;
    T_kmer kmer_mask = T_kmer(0xffffffffffffffff>>(64-k*2));
    T_kmer kmer;
    CRead<T_skm_len> skm;
    while (skms_store.try_pop_skm(skm)) {
        skm_len = skm.length();
        // first kmer
        kmer = skm.get_2bit_kmer(0, k);
        kmers_h.push_back(kmer);
        // latter kmers
        for (T_skm_len j=k; j<skm_len; j++) {
            kmer = ((kmer << 2) | skm.get_2bit(j)) & kmer_mask;
            kmers_h.push_back(kmer);
        }
    }
    thrust::device_vector<T_kmer> kmers_d(kmers_h);
    tot_kmers = kmers_d.size();
    // cerr<<est_kmer<<"|"<<db_skm<<"|"<<est_skm<<"|"<<tot_kmers<<endl;

    // 1. sort: [ABCBBAC] -> [AABBBCC] (kmers_d)
    thrust::sort(thrust::device, kmers_d.begin(), kmers_d.end()/*, thrust::greater<T_kmer>()*/);
    thrust::host_vector<T_kmer> sorted_kmers_h = kmers_d;

    // 2. find changes: [AABBBCC] -> [0,0,1,0,0,1,0] (comp_vec_d)
    // thrust::device_vector<bool> comp_vec_d(kmers_d.size());
    // thrust::transform(thrust::device, kmers_d.begin()+1 /*x beg*/, kmers_d.end() /*x end*/, kmers_d.begin()/*y beg*/, comp_vec_d.begin()+1/*res beg*/, differentfromprev());
    // comp_vec_d[0] = 1; //
    // int distinct_kmer_cnt = thrust::reduce(thrust::device, comp_vec_d.begin(), comp_vec_d.end()) + 1;
    
    // 2. find changes: [AABBBCC] -> [0,1,0,1,1,0,1] (same_flag_d)
    thrust::device_vector<bool> same_flag_d(kmers_d.size());
    thrust::transform(thrust::device, kmers_d.begin()+1 /*x beg*/, kmers_d.end() /*x end*/, kmers_d.begin()/*y beg*/, same_flag_d.begin()+1/*res beg*/, sameasprev());
    same_flag_d[0] = 0; //
    
    // 3. remove same idx: [0123456] [0101101] -> [0,2,5] (idx_d)
    thrust::device_vector<T_read_len> idx_d(kmers_d.size());
    thrust::sequence(thrust::device, idx_d.begin(), idx_d.end());
    auto new_end_d = thrust::remove_if(thrust::device, idx_d.begin(), idx_d.end(), same_flag_d.begin(), thrust::identity<bool>()); // new_end_d is an iterator

    // 3. replace with index: [0,0,1,0,0,1,0] -> [0,0,2,0,0,5,0] (comp_vec_d)
    // thrust::device_vector<T_read_len> seq_d(kmers_d.size());
    // thrust::sequence(thrust::device, seq_d.begin(), seq_d.end());
    // thrust::transform(thrust::device, comp_vec_d.begin() /*x*/, comp_vec_d.end(), seq_d.begin()/*y*/, comp_vec_d.begin()/*res*/, replaceidx());

    // // 4. skip repeats: [0,0,2,0,0,5,0] -> [0,2,5] (comp_vec_d)
    // auto new_end_d = thrust::remove_if(thrust::device, comp_vec_d.begin(), comp_vec_d.end(), is_zero());

    // 4. copy device_vector back to host_vector
    thrust::host_vector<T_read_len> idx_h(idx_d.begin(), new_end_d);
    idx_h.push_back(tot_kmers); // [0,2,5] -> [0,2,5,7] A2 B3 C2
    
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
    return total_kmer_cnt; // total kmer
    return idx_h.size()-1; // total distinct kmer
}

// kmc_counting_GPU (skms, gpuid, kmer_min_freq, kmer_max_freq, kmc_result_curthread);