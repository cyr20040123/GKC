# GKC
GPU-accelerated K-mer Counting Library
# 

# Kmer Counting Summary

## ***Project***
 - A GPU-accelerated k-mer counting library for genome analysis
## ***Motivation***
1. Current k-mer counter are great (time and memory efficient), can reach the I/O bottleneck already
2. Many genome analysis tasks such as sequence alignment and de novo assembly require k-mer counting but with specific requirements *(e.g., counting only a portion of kmers, mark kmer's positions)*
3. No existing library.
4. GPU Qword radix sort for kmer counting supporting at most 64-mer. and can be expanded longer.
## ***Application Examples***
1. **[Minimap2 aligner]** (k=25,p=15) Find minimizer in ω-windows (both strand) and store their positions. *Output:* (minimizer, pos, strand)
2. **[Wtdbg2 assembler]** Count a portion of kmers (by default 1/4) and store their positions. *Output:* {kmer1: [pos1, pos2], ...}
3. **[KMC3 kmer counter]** Partition kmers by signature (minimizer with specific rules) and count kmers for each partition.
4. **DALIGN, ...**

## ***Procedures***
### **GPU Partitioning**
1. Generate Minimizer
   - normal minimizer
   - (optional for memory-efficient approaches) minimizer with specific rules (for signature or counting a portion of kmers)
2. Generate Superkmer
   - GPU mark superkmer positions
   - CPU produces supermers (*for counting with position:* store the skm position as well)
### **GPU Counting**
 - CPU load kmers from supermers (kmer locations can be generated simultaneously)
 - GPU radix sort -> Kmer list
### **Kmer Storage / CPU Counting**
1. Hash table
2. Counting quotient filter (more memory efficient, better for larger k [Squeakr, 2017])

## ***Questions***
### **GPU Partitioning**
1. Use 32-bit integer for shorter k-mer / minimizer will be faster?
2. Which minimizer will be better (in both time and bin space)?
3. Best setting of grid size and number of streams, given global memory size, CUDA cores, memory bandwidth, etc.
### **GPU Counting**
1. Compare GPU radix sort (LSD/MSD) / GPU hash table.
### **Kmer Storage / CPU Counting**

