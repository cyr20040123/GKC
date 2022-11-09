// BUG: LINE_BUF_SIZE (4M) 大于 CUR_BUF_SIZE (1M) 会卡死（100%) (不是卡死，是争抢资源缓慢，busy wait循环里放1ms wait就不卡了)
// BUG: CUR_BUF_SIZE 较小(10M) 概率卡死
//  - : CUR_BUF_SIZE 过小(4M 1M 0.5M) 读取结果数量不正确 // DEBUGED

#define DEBUG

#include <chrono>
#include <iostream>
// #include <fstream>
#include <functional>
#include <string>
#include <vector>
#include <cassert>
#include "../types.h"
// #include <sys/mman.h>   // mmap
// #include <fcntl.h>      // open for mmap (unix only)
#include <cstdio>
#include "../read_loader.hpp"
using namespace std;

// ================ CLASS WallClockTimer ================
class WallClockTimer {  // evaluating wall clock time
private:
    std::chrono::_V2::system_clock::time_point start;
    std::chrono::_V2::system_clock::time_point end;
public:
    WallClockTimer();                           // Initialize a wall-clock timer and start.
    double stop(bool millisecond = false);      // Return end-start duration in sec from init or restart till now.
    void restart();                             // Restart the clock start time.
};
WallClockTimer::WallClockTimer() {
    start = chrono::high_resolution_clock::now();
}
double WallClockTimer::stop(bool millisecond /* = false*/) {
    end = chrono::high_resolution_clock::now();
    if (millisecond) return chrono::duration_cast<chrono::milliseconds>(end - start).count();
    else return chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.0;
}
void WallClockTimer::restart() {
    start = chrono::high_resolution_clock::now();
}


void fs_loader(string filename) {
    // ifstream fin(filename, ios::in);
}

void mm_loader(string filename, int n_threads = 4) {
    ReadLoader rl(n_threads, filename);
    WallClockTimer wct;
    rl.load_file();
    double sec = wct.stop();
    cout<<"Finish in "<<sec<<" sec, avg speed is "<<rl.get_file_size(filename.c_str())/1024/1024/sec<<" MB/s."<<endl;
    cout<<"Total # of reads:  "<<rl.read_cnt<<endl;
    vector<string*>* thread_reads = rl.get_thread_reads();
    size_t total_len = 0;
    for (int t=0; t<n_threads; t++) {
        for (string *s: thread_reads[t]) total_len += s->length();
    }
    cout<<"Total read length: "<<total_len<<endl;
    vector<ReadPtr> reads;
    cout<<rl.get_reads_async(reads, 0, -1)<<endl;
    cout<<reads.size()<<endl;
}

int main(int argc, const char **argvs) {
    // string FILENAME = "/home/cyr/downloads/pacbio_filtered.fastq";
    // string FILENAME = "/mnt/f/study/bio_dataset/hg002pacbio/man_files/SRR8858432.man.fasta";
    // string FILENAME = "/mnt/f/study/testing_data/SRR8858432.fastq";
    string FILENAME = "/mnt/f/study/bio_dataset/Ecoli/pacbio_filtered.fastq";
    // fs_loader(FILENAME);
    // system("sudo sh -c \"free && sync && echo 3 > /proc/sys/vm/drop_caches && free\"");
    mm_loader(FILENAME, atoi(argvs[1]));
    #ifdef DEBUG
    cout<<"checksum:"<<checksum<<endl;
    #endif
    return 0;
}