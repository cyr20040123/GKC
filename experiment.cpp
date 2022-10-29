#include "utilities.hpp"
#include <atomic>
#include <future>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <bits/stdc++.h>
#include "gkc_cuda.hpp"
#include "csr.hpp"
#include "read_loader.hpp"
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
bool sort_comp (const ReadPtr x, const ReadPtr y) {
    return x.len < y.len;
}
void test_skm_part_size (CUDAParams gpars) {
    GPUReset(gpars.device_id); // must before pinned memory allocation

    atomic<size_t> skm_part_sizes[512];
    for (int i=0; i<PAR.SKM_partitions; i++) skm_part_sizes[i]=0;
    const char* filename = PAR.read_files[0].c_str();
    CSR<char> csr;
    vector<ReadPtr> reads;
    // ReadLoader::LoadReadsToCSR(fa_filename, csr, PAR.K_kmer);
    // for (int i=0; i<csr.items(); i++) {
    //     reads.push_back({
    //         &(csr.get_raw_data()[csr.get_raw_offs()[i]]), 
    //         T_read_len(csr.get_raw_offs()[i+1] - csr.get_raw_offs()[i])
    //     });
    // }
    sort(reads.begin(), reads.end(), sort_comp); // TODO: remove and compare
    stringstream ss;
    ss << "n_reads: " << reads.size() << "\tmin_len: " << reads.begin()->len << "\tmax_len:" << reads.rbegin()->len << "\tavg_len:" << csr.size()/reads.size() << endl;
    logger->log(ss.str());
    PinnedCSR pinned_reads(reads);
    logger->log("Pinned:"+to_string(pinned_reads.get_n_reads()));
    WallClockTimer wct;
    GenSuperkmerGPU (pinned_reads, PAR.K_kmer, PAR.P_minimizer, false, gpars, CountTask::SKMPartition, PAR.SKM_partitions, skm_part_sizes);
    logger->log("Time of supermer generation: " + to_string(wct.stop()));
    vector<size_t> part_sizes;
    for (int i=0; i<PAR.SKM_partitions; i++) part_sizes.push_back(skm_part_sizes[i]);
    calVarStdev(part_sizes);
    logger->logvec(part_sizes, "Partition Sizes:");
}

void process_reads(vector<ReadPtr> &reads, CUDAParams gpars, atomic<size_t> skm_part_sizes[]) {
    
    sort(reads.begin(), reads.end(), sort_comp); // TODO: remove and compare
    stringstream ss;
    ss << "n_reads: " << reads.size() << "\tmin_len: " << reads.begin()->len << "\tmax_len:" << reads.rbegin()->len << endl;
    logger->log(ss.str());
    PinnedCSR pinned_reads(reads);
    logger->log("Pinned:"+to_string(pinned_reads.get_n_reads()));
    WallClockTimer wct;
    GenSuperkmerGPU (pinned_reads, PAR.K_kmer, PAR.P_minimizer, false, gpars, CountTask::SKMPartition, PAR.SKM_partitions, skm_part_sizes);
    logger->log("Time of supermer generation: " + to_string(wct.stop()));
    vector<size_t> part_sizes;
    for (int i=0; i<PAR.SKM_partitions; i++) part_sizes.push_back(skm_part_sizes[i]);
    calVarStdev(part_sizes);
    logger->logvec(part_sizes, "Partition Sizes:");
}
void test_skm_part_size_async (CUDAParams gpars) {
    atomic<size_t> skm_part_sizes[512];
    for (int i=0; i<PAR.SKM_partitions; i++) skm_part_sizes[i]=0;

    GPUReset(gpars.device_id); // must before pinned memory allocation

    vector<ReadPtr> reads;
    ReadLoader rl(4, PAR.read_files[0]);
    // future<void> file_loading_res = async(std::launch::async, &ReadLoader::load_file, &rl);
    future<void> file_loading_res = async(std::launch::async, [&rl](){return rl.load_file();});
    
    T_read_cnt n_read_loaded = 0;
    T_read_cnt batch_size = 5000;
    bool loading_not_finished = true;
    while (loading_not_finished) {
        future_status status;
        status = file_loading_res.wait_for(10ms);
        switch (status) {
            case future_status::deferred:
            case future_status::timeout:
                if (rl.get_read_cnt() - n_read_loaded >= batch_size) {
                    reads.clear();
                    n_read_loaded += rl.get_reads_async(reads, n_read_loaded, batch_size);
                    // ... process reads
                    process_reads(reads, gpars, skm_part_sizes);
                }
                break;
            case future_status::ready:  // all reads are loaded
                reads.clear();
                n_read_loaded += rl.get_reads_async(reads, n_read_loaded, -1);
                // ... process reads
                process_reads(reads, gpars, skm_part_sizes);
                loading_not_finished = false;
                break;
        }
    }
    logger->log("All reads processed: "+to_string(n_read_loaded));
    return;
}

int main (int argc, char** argvs) {
    Logger _logger(0, Logger::LV_DEBUG, true, "./log/");
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
    gpars.n_streams = 1;
    gpars.NUM_BLOCKS_PER_GRID = 16;
    gpars.NUM_THREADS_PER_BLOCK = 256;

    // test_skm_part_size(gpars);
    test_skm_part_size_async(gpars);
    return 0;
}