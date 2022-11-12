#include "utilities.hpp"
#include <atomic>
#include <future>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <bits/stdc++.h>
#include "csr.hpp"
#include "read_loader.hpp"
#include "superkmers.hpp"
#include "gkc_cuda.hpp"
#include "kmer_counting.hpp"
#include "thread_pool.hpp"
using namespace std;

Logger *logger;

void calVarStdev(vector<size_t> &vecNums) // calc avg max min var std cv (Coefficient of variation)
{
    size_t max_val = 0, min_val = 0xffffffffffffffff;
	size_t sumNum = accumulate(vecNums.begin(), vecNums.end(), 0);
	size_t mean = sumNum / vecNums.size();
	double accum = 0.0;
	for_each(vecNums.begin(), vecNums.end(), [&](const size_t d) {
		accum += (d - mean)*(d - mean);
        if (d>max_val) max_val = d;
        if (d<min_val) min_val = d;
	});
	double variance = accum / vecNums.size();
	double stdev = sqrt(variance);
    
    logger->log("SKM TOT_LEN="+to_string(sumNum));
    // logger->log("SKM TOT_LEN="+sumNum); // seg fault ???
    stringstream ss;
	ss << "AVG=" << mean << "\tMAX=" << max_val << "\tmin=" << min_val << "\tvar=" << variance << "\tSTD=" << stdev << "\tCV=" << stdev/double(mean) << endl;
    logger->log(ss.str());
}

// ==== count skm sizes ====
/*
void process_reads_v2 (vector<ReadPtr> &reads, CUDAParams gpars, atomic<size_t> skm_part_sizes[]) {
    
    sort(reads.begin(), reads.end(), sort_comp); // TODO: remove and compare
    stringstream ss;
    ss << "----\tBATCH\tn_reads = " << reads.size() << "\tmin_len = " << reads.begin()->len << "\tmax_len = " << reads.rbegin()->len <<"\t----";
    logger->log(ss.str());
    PinnedCSR pinned_reads(reads);
    logger->log("Pinned Reads\tcount = "+to_string(pinned_reads.get_n_reads())+"\tsize="+to_string(pinned_reads.size_capacity));
    
    // function<void(T_h_data)> process_func = [&skm_part_sizes](T_h_data hd){CalcSKMPartSize_instream(hd.reads_cnt, hd.superkmer_offs, hd.reads_offs, hd.minimizers, PAR.SKM_partitions, PAR.K_kmer, skm_part_sizes);};
    function<void(T_h_data)> process_func = [&skm_part_sizes](T_h_data hd){CalcSKMPartSize_instream(hd.reads_cnt, hd.superkmer_offs, hd.reads_offs, hd.minimizers, PAR.SKM_partitions, PAR.K_kmer, skm_part_sizes);};
    GenSuperkmerGPU (pinned_reads, PAR.K_kmer, PAR.P_minimizer, false, gpars, CountTask::SKMPartition, PAR.SKM_partitions, process_func);
    // GenSuperkmerGPU (pinned_reads, PAR.K_kmer, PAR.P_minimizer, false, gpars, CountTask::SKMPartition, PAR.SKM_partitions, skm_part_sizes);
}

void skm_part_size_v2 (CUDAParams gpars) {
    atomic<size_t> skm_part_sizes[512];
    for (int i=0; i<PAR.SKM_partitions; i++) skm_part_sizes[i]=0;

    GPUReset(gpars.device_id); // must before not after pinned memory allocation

    ReadLoader::work_while_loading(
        [gpars, &skm_part_sizes](vector<ReadPtr> &reads){process_reads_v2(reads, gpars, skm_part_sizes);},
        4, PAR.read_files[0], 1000, true, 20*ReadLoader::MB
    );
    logger->log("All reads processed.");

    vector<size_t> part_sizes;
    for (int i=0; i<PAR.SKM_partitions; i++) part_sizes.push_back(skm_part_sizes[i]);
    calVarStdev(part_sizes);
    logger->logvec(part_sizes, "Partition Sizes:");
    return;
}*/

// ==== gen skm and count ====
void process_reads_count(vector<ReadPtr> &reads, CUDAParams gpars, vector<SKMStore*> &skm_part_vec) {
    sort(reads.begin(), reads.end(), sort_comp); // TODO: remove and compare
    stringstream ss;
    ss << "----\tBATCH\tn_reads = " << reads.size() << "\tmin_len = " << reads.begin()->len << "\tmax_len = " << reads.rbegin()->len <<"\t----";
    logger->log(ss.str());
    PinnedCSR pinned_reads(reads);
    T_read_cnt read_cnt = pinned_reads.get_n_reads();
    logger->log("Pinned: "+to_string(pinned_reads.get_n_reads())+" size = "+to_string(pinned_reads.size_capacity));
    
    function<void(T_h_data)> process_func = [&skm_part_vec, read_cnt](T_h_data hd){
        SKMStore::add_skms (skm_part_vec, PAR.K_kmer, hd.reads_cnt, hd.reads, hd.reads_offs, hd.superkmer_offs, hd.minimizers);
    };
    GenSuperkmerGPU (pinned_reads, PAR.K_kmer, PAR.P_minimizer, false, gpars, CountTask::SKMPartition, PAR.SKM_partitions, process_func);
}
// void gen_skm_and_count(CUDAParams gpars) {
//     GPUReset(gpars.device_id); // must before not after pinned memory allocation

//     vector<SKMStore*> skm_part_vec;
//     int i, tid;
//     for (i=0; i<PAR.SKM_partitions; i++) skm_part_vec.push_back(new SKMStore());//
    
//     // 1st phase: loading and generate superkmers
//     logger->log("**** Phase 1: Loading and generate superkmers ****", Logger::LV_NOTICE);
//     WallClockTimer wct1;
    
//     ReadLoader::work_while_loading(
//         [gpars, &skm_part_vec](vector<ReadPtr> &reads){process_reads_count(reads, gpars, skm_part_vec);},
//         PAR.N_threads, PAR.read_files[0], PAR.Batch_read_loading, false, PAR.Buffer_fread_size_MB*ReadLoader::MB
//     );
    
//     double p1_time = wct1.stop();
//     logger->log("**** All reads loaded and SKMs generated (Phase 1 ends) ****", Logger::LV_NOTICE);
//     logger->log("     Phase 1 Time: " + to_string(p1_time) + " sec", Logger::LV_INFO);

//     size_t skm_tot_len = 0;
//     for(i=0; i<PAR.SKM_partitions; i++) {
//         skm_tot_len += skm_part_vec[i]->tot_size;
//     }
//     logger->log("SKM TOT LEN = " + to_string(skm_tot_len));

//     // cout<<"Countinue? ..."; char tmp; cin>>tmp;
//     GPUReset(gpars.device_id);

//     // 2nd phase: superkmer extraction and kmer counting
//     logger->log("**** Phase 2: Superkmer extraction and kmer counting ****", Logger::LV_NOTICE);
//     WallClockTimer wct2;
    
//     int n_threads = PAR.N_threads;
    
//     vector<T_kmc> kmc_result[PAR.SKM_partitions];
    
//     future<size_t> distinct_kmer_cnt[PAR.SKM_partitions];
//     size_t distinct_kmer_cnt_tot = 0;
//     logger->log("(with "+to_string(n_threads)+" threads)");
//     for (i=0, tid=0; i < PAR.SKM_partitions; i++, tid=(tid+1)%n_threads) {
//         cerr<<" T"<<tid<<"_"<<i;
//         if (distinct_kmer_cnt[tid].valid())
//             distinct_kmer_cnt_tot += distinct_kmer_cnt[tid].get();
//         distinct_kmer_cnt[tid] = std::async(
//             std::launch::async, 
//             [&skm_part_vec, &gpars, &kmc_result, i] () {
//                 return kmc_counting_GPU (PAR.K_kmer, *(skm_part_vec[i]), gpars.device_id, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result[i]);
//             }
//         );
//     }
//     cerr<<endl;
//     for (tid=0; tid<n_threads; tid++) {
//         if (distinct_kmer_cnt[tid].valid()) {
//             distinct_kmer_cnt_tot += distinct_kmer_cnt[tid].get();
//         }
//     }
//     logger->log("Total number of distinct kmers: "+to_string(distinct_kmer_cnt_tot));
    
//     double p2_time = wct2.stop();
//     logger->log("**** Kmer counting finished (Phase 2 ends) ****", Logger::LV_NOTICE);
//     logger->log("     Phase 2 Time: " + to_string(p2_time) + " sec", Logger::LV_INFO);

//     for (i=0; i<PAR.SKM_partitions; i++) delete skm_part_vec[i];//
//     return;
// }
void gen_skm_and_count_with_TP(CUDAParams gpars) {
    GPUReset(gpars.device_id); // must before not after pinned memory allocation

    vector<SKMStore*> skm_part_vec;
    int i, tid;
    for (i=0; i<PAR.SKM_partitions; i++) skm_part_vec.push_back(new SKMStore());//
    
    // 1st phase: loading and generate superkmers
    logger->log("**** Phase 1: Loading and generate superkmers ****", Logger::LV_NOTICE);
    WallClockTimer wct1;
    
    ReadLoader::work_while_loading_V2(
        [gpars, &skm_part_vec](vector<ReadPtr> &reads){process_reads_count(reads, gpars, skm_part_vec);},
        2, PAR.read_files[0], PAR.Batch_read_loading, true, PAR.Buffer_fread_size_MB*ReadLoader::MB
    );
    
    double p1_time = wct1.stop();
    logger->log("**** All reads loaded and SKMs generated (Phase 1 ends) ****", Logger::LV_NOTICE);
    logger->log("     Phase 1 Time: " + to_string(p1_time) + " sec", Logger::LV_INFO);

    size_t skm_tot_len = 0;
    for(i=0; i<PAR.SKM_partitions; i++) {
        skm_tot_len += skm_part_vec[i]->tot_size;
    }
    logger->log("SKM TOT LEN = " + to_string(skm_tot_len));

    // cout<<"Countinue? ..."; char tmp; cin>>tmp;
    // GPUReset(gpars.device_id);

    // 2nd phase: superkmer extraction and kmer counting
    logger->log("**** Phase 2: Superkmer extraction and kmer counting ****", Logger::LV_NOTICE);
    logger->log("(with "+to_string(PAR.N_threads)+" threads)");
    
    WallClockTimer wct2;
    int n_threads = PAR.N_threads;
    ThreadPool<size_t> tp(n_threads);
    
    vector<T_kmc> kmc_result[PAR.SKM_partitions];
    future<size_t> distinct_kmer_cnt[PAR.SKM_partitions];

    for (i=0; i<PAR.SKM_partitions; i++) {
        distinct_kmer_cnt[i] = tp.commit_task([&skm_part_vec, &gpars, &kmc_result, i] () {
            return kmc_counting_GPU (PAR.K_kmer, *(skm_part_vec[i]), gpars.device_id, PAR.kmer_min_freq, PAR.kmer_max_freq, kmc_result[i]);
        });
    }
    tp.finish();

    size_t distinct_kmer_cnt_tot = 0;
    for (i=0; i<PAR.SKM_partitions; i++) {
        if (distinct_kmer_cnt[i].valid()) {
            distinct_kmer_cnt_tot += distinct_kmer_cnt[i].get();
        }
    }
    cerr<<endl;
    
    logger->log("Total number of distinct kmers: "+to_string(distinct_kmer_cnt_tot));
    
    double p2_time = wct2.stop();
    logger->log("**** Kmer counting finished (Phase 2 ends) ****", Logger::LV_NOTICE);
    logger->log("     Phase 2 Time: " + to_string(p2_time) + " sec", Logger::LV_INFO);

    // for (i=0; i<PAR.SKM_partitions; i++) delete skm_part_vec[i];// deleted in kmc_counting_GPU
    return;
}

int main (int argc, char** argvs) {
    cerr<<"================ PROGRAM BEGINS ================"<<endl;
    Logger _logger(0, Logger::LV_DEBUG, false, "./log/");
    logger = &_logger;
    PAR.ArgParser(argc, argvs);
    // ReadLoader rl(4, PAR.read_files[0]);
    // rl.load_file();
    // cout<<rl.read_cnt<<endl;

    stringstream ss;
    for(int i=0; i<argc; i++) ss<<argvs[i]<<" ";
    logger->log(ss.str());

    CUDAParams gpars;
    gpars.device_id = 0;
    gpars.n_streams = 6;
    gpars.NUM_BLOCKS_PER_GRID = 8;
    gpars.NUM_THREADS_PER_BLOCK = 256;

    // test_skm_part_size(gpars);
    // test_skm_part_size_async(gpars);
    // skm_part_size_v2(gpars);
    
    WallClockTimer wct_oa;
    // gen_skm_and_count(gpars);
    cerr<<"----------------------------------------------"<<endl;
    gen_skm_and_count_with_TP(gpars);
    cerr<<"================ PROGRAM ENDS ================"<<endl;
    cout<<wct_oa.stop()<<endl;
    return 0;
}