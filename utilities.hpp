#ifndef _UTILITIES_HPP
#define _UTILITIES_HPP

#include "types.h"
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
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


// ================ CLASS Logger ================
class Logger { // for CLI logging and file logging
public:
    enum LogLevel {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG};
    string loglabel[6] = {"[FATAL]", "[ERROR]", "[NOTICE]", "[WARNING]", "[INFO]", "[DEBUG]"};
private:
    LogLevel log_lv = LV_WARNING; // output level only equal or smaller than this value
    bool to_file = false;
    int my_rank = 0;
    ofstream flog;
    string path, logfilename;
    string timestring(bool short_format = false);
public:
    /**
     * @brief Logger initialization.
     * @param  process_id       : (optional) For multithreading process identification.
     * @param  log_level        : (optional) Only record with no greater level logs. Can be {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG}.
     * @param  to_log_file      : (optional) Whether to write logs into file.
     * @param  logfile_folder   : (optional) Save log file to which folder.
     */
    Logger (int process_id = 0, LogLevel log_level = LV_WARNING, bool to_log_file = false, string logfile_folder="./");
    
    /**
     * @brief Call this function to log a string.
     * @param  info         : Log text.
     * @param  lv           : (optional) Log level. Can be {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG}.
     * @param  with_time    : (optional) Whether to log with current wall clock time.
     */
    // void log(const char *info, LogLevel lv = LV_DEBUG, bool with_time = false);
    void log(string info, LogLevel lv = LV_DEBUG, bool with_time = false);
    
    /**
     * @brief Call this function to log a vector. If len<5 items will be output with TAB between else one item per line.
     * @param  vec_data     : The vector to be logged. Must be printable.
     * @param  info         : The log text.
     * @param  lv           : (optional) Log level. Can be {LV_FATAL, LV_ERROR, LV_NOTICE, LV_WARNING, LV_INFO, LV_DEBUG}.
     * @param  with_time    : (optional) Whether to log with current wall clock time.
     */
    template<typename T_item>
    void logvec(vector<T_item> &vec_data, string info, LogLevel lv = LV_DEBUG, bool with_time = false);

    // Deconstructor
    ~Logger();
};

    template<typename T_item>
    void Logger::logvec(vector<T_item> &vec_data, string info, LogLevel lv/* = LV_DEBUG*/, bool with_time/* = false*/) {
        if (lv <= log_lv) {
            this->log(info, lv, with_time);
            if (vec_data.size()<5) { // use \t
                cerr<<"    ";
                for (T_item &item: vec_data) cerr<<to_string(item)<<"\t";
                cerr<<endl;
                if (to_file && flog.is_open()) {
                    flog<<"    ";
                    for (T_item &item: vec_data) flog<<to_string(item)<<"\t";
                    flog<<endl;
                }
            } else {
                int i=0;
                for (T_item &item: vec_data) {
                    if (i++ % 10 == 0) cerr<<endl;
                    cerr<<"\t"<<to_string(item);
                }
                cerr<<endl;
                i=0;
                if (to_file && flog.is_open()) {
                    for (T_item &item: vec_data) {
                        flog<<"\t"<<to_string(item);
                        if (i++ % 10 == 0) flog<<endl;
                    }
                    flog<<endl;
                }
            }
        }
    }
    

//
class ReadLoader {
private:
    const size_t KB = 1024;
    const size_t MB = 1048576;
    size_t CUR_BUF_SIZE = 200 * MB;     // size to be determined
    size_t LINE_BUF_SIZE = 100 * KB;    // should be larger than 3 * max_read_len
    string _filename;
    char** _buf_cur;
    string* _buf_prev_remain;
    int _n_threads;
    future<int> *_proc_res;
    mutex *_pbuf_mtxs;
    bool *_pbuf_set;
    vector<string> *_thread_reads;
    vector<T_read_cnt> *_thread_bat_split_pos; // store the split position of each batch to keep the original order of the reads

    T_read_cnt _load_fastq (int tid, bool last_round = false) {  // 1. process prev, 2. set next prev, 3. process cur (consider 1 process only)
        string buf;
        if (!last_round) buf.resize(CUR_BUF_SIZE + LINE_BUF_SIZE);
        buf = "";
        
        // ---- Process _buf_prev_remain ----
        while(!_pbuf_set[tid]); // busy wait
        _pbuf_mtxs[tid].lock(); // only lock tid and lock/unlock tid+1
        // ... proc pbuf
        buf += _buf_prev_remain[tid];
        // -
        _pbuf_set[tid] = false;
        _pbuf_mtxs[tid].unlock();
        
        // ---- Set the next _buf_prev_remain ----
        size_t remain_pos;
        if (!last_round) {
            while(_pbuf_set[(tid+1)%_n_threads]); // busy wait
            _pbuf_mtxs[(tid+1)%_n_threads].lock();
            // ... set next pbuf
            buf += _buf_cur[tid];
            remain_pos = buf.rfind('\n');
            while (buf[remain_pos+1] != '@') {
                remain_pos = buf.rfind('\n', remain_pos-1);
            }
            _buf_prev_remain[(tid+1)%_n_threads] = string(buf.begin()+remain_pos+1, buf.end());
            // -
            _pbuf_set[(tid+1)%_n_threads] = true;
            _pbuf_mtxs[(tid+1)%_n_threads].unlock();
        } else {
            remain_pos = buf.length();
        }

        // ---- Process _buf_cur ----
        if (buf[0] != '@') {
            cerr << tid << " Error: wrong file format (not fastq). " << buf[0] << endl;
            exit(1);
        }
        size_t i = 0, j;
        while (i < remain_pos) {
            i = buf.find('\n', i+1); // line end of read info // TODO: save read info
            j = buf.find('\n', i+1); // line end of read
            if (buf[j-1] == '\r') j--;
            _thread_reads[tid].push_back(string(buf.begin()+i+1, buf.begin()+j));
            if (last_round) break;
            i = buf.find('\n', j+1); // line end of '+'
            i = buf.find('\n', i+1); // line end of quality
        }
        _thread_bat_split_pos[tid].push_back(_thread_reads[tid].size()); // record split position
        // ...
        return *(_thread_bat_split_pos[tid].rbegin()) - *(_thread_bat_split_pos[tid].rbegin()+1);
    }
    T_read_cnt _load_fasta (int tid, bool last_round = false) {
        return 0;
    }
public:
    T_read_cnt read_cnt = 0;

    ReadLoader (int n_threads, string filename, size_t memory_size = 0) {
        size_t file_size = get_file_size(filename.c_str());
        _filename = filename;
        int i;
        this->_n_threads = n_threads;
        _proc_res = new future<int> [n_threads];//
        _buf_cur = new char* [n_threads];//
        _buf_prev_remain = new string [n_threads];//
        _pbuf_mtxs = new mutex [n_threads];//
        _pbuf_set = new bool [n_threads];//
        _thread_reads = new vector<string> [n_threads];//
        _thread_bat_split_pos = new vector<T_read_cnt> [n_threads];//
        if (file_size == 0) {
            CUR_BUF_SIZE = 20 * MB;
        } else if (file_size / n_threads < 100 * MB) {
            CUR_BUF_SIZE = max (LINE_BUF_SIZE, (file_size + 1*KB) / n_threads);
        } else if (file_size / n_threads < 400 * MB) {
            CUR_BUF_SIZE = (file_size + 1*KB) / (n_threads * 2);
        } else {
            CUR_BUF_SIZE = min (400 * MB, (file_size + 1*KB) / (n_threads * 4));
        }
        cerr << "CUR_BUF_SIZE = " << CUR_BUF_SIZE / MB << "MB \t#threads = " << n_threads << endl;

        _pbuf_set[0] = true;
        _pbuf_mtxs[0].unlock(); // unlock only when _buf_prev_remain is prepared
        for (i=0; i<n_threads; i++) {
            _buf_cur[i] = new char [CUR_BUF_SIZE+2];
            _buf_prev_remain[i].resize(LINE_BUF_SIZE);
            _buf_prev_remain[i] = "";
            _thread_bat_split_pos[i].push_back(0);
        }
    }
    ~ReadLoader () {
        for (int i=0; i<_n_threads; i++) {
            delete _buf_cur[i];
        }
        delete _buf_cur;//
        delete [] _buf_prev_remain;//
        delete [] _proc_res;//
        delete [] _pbuf_mtxs;//
        delete [] _pbuf_set;//
        delete [] _thread_reads;//
        delete [] _thread_bat_split_pos;//
    }
    void load_file () {
        // Open file:
        FILE *fqfile = fopen(_filename.c_str(), "rb");
        if (fqfile == NULL) {
            cerr << "Unable to open: " << _filename << endl;
            exit(1);
        }

        // Determine the file type:
        bool fastq = *(_filename.rbegin()) == 'q';
        std::function<T_read_cnt(int)> proc_func;
        if (fastq) {
            cerr << "fastq" << endl;
            proc_func = [this](int tid) -> T_read_cnt {return this->_load_fastq(tid);}; // [this] pass by value   // todo: type
        } else {
            proc_func = [this](int tid) -> T_read_cnt {return this->_load_fasta(tid);}; // [&this] pass by ref   // todo: type
        }

        // Load and process the file:
        int i = 0, i_break;
        size_t tmp_size; // i_thread
        bool not_1st_loop = false;
        while ((tmp_size = fread(_buf_cur[i], sizeof(char), CUR_BUF_SIZE, fqfile)) > 0) {
            _buf_cur[i][tmp_size] = 0;
            _proc_res[i] = async(std::launch::async, proc_func, i);
            i = (i+1) % _n_threads;
            not_1st_loop |= i==0;
            if (not_1st_loop) read_cnt += _proc_res[i].get(); // wait for the previous round
        }
        i_break = i;
        for (i = (i+1)%_n_threads; i != i_break; i = (i+1)%_n_threads) {
            if (not_1st_loop) read_cnt += _proc_res[i].get(); // wait for the previous round
        }
        read_cnt += _load_fastq(i_break, true); // process the data in the last pbuf
    }
    
    vector<string>* get_thread_reads() {return _thread_reads;}
    vector<T_read_len>* get_thread_bat_split_pos() {return _thread_bat_split_pos;}
    
    static size_t get_file_size(const char *filename) {
        struct stat statbuf;
        if (stat(filename, &statbuf) != 0) {
            cerr<<"ERROR "<<errno<<": can't get file size of "<<filename;
            perror("");
            exit(errno);
        }
        return statbuf.st_size;
    }
};



// ================ CLASS GlobalParams ================
static class GlobalParams {
public:
    unsigned int K_kmer = 21;   // length of kmer
    unsigned int P_minimizer = 7, SKM_partitions = 128; // minimizer length and superkmer partitions
    unsigned short kmer_min_freq = 1, kmer_max_freq = 1000; // count kmer cnt in [min,max] included
    bool HPC = false;           // homopolymer compression assembly
    bool CPU_only = false;
    int Kmer_filter = 25;       // percentage
    int N_threads = 8;          // threads per process
    string tmp_file_folder = "./tmp/";
    string log_file_folder = "./log/";
    vector<string> read_files;
    vector<string> local_read_files;

    void ArgParser(int argc, char* argvs[]);
} PAR;

#endif