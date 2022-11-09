# Developer Handbook
## TODO
### DEBUG
 - cudaHostAlloc fails because of limited resource.
## Hardware Settings
### **CPU:** AMD Ryzen 7 5800H
 - 8-core CPU was set to 90% in Windows at 2.85GHz
 ```bash
 # Limit CPU usage:
 cpulimit -e test -l 800
 ```
### **GPU:** Nvidia Geforce RTX 3060 Laptop
 - Tensor Core
 - around 70 watts at 1400MHz
```bash
nvidia-smi -pm 1            # enable persistance mode (via WSL2)
# nvidia-smi -pl 125        # set power limit to 125W
nvidia-smi -lgc 1400,1400   # lock the gpu clock (via Windows)
nvidia-smi -rgc             # Resets the Gpu clocks to the default values.
```

### Free Linux Memory Buffer/Cache
 - To ensure a cold start every time
 ```bash
 sudo sh -c "free && sync && echo 3 > /proc/sys/vm/drop_caches && free"
 ```

## _
## Baselines
138749517
136695098
135435562
### Datasets
| Dataset   | Format    | File Size | Bases     | Reads     |K-mers    |
| ----      | ----      | ----      | ----      | ----      | ----      |
| Ecoli     | Fastq     | 266 MB    | ?         | 16890     | 135435562 |
| SRR8858432| Fasta     | 1.66 GB   | ?         | 138688    | 1148781224|

### 0. GKC
#### **Result**

| Dataset   | Thread| RAM (GB)  | Time (sec)    |
| ----      | ----  | ----      | ----          |
| *Ecoli q* | 2     |           | 6.90 = 1.70 + 3.85|
|           | 4     | 1.89      | 5.60 = 1.30 + 2.75|
|           | 8     |           | 4.80 = 1.45 + 2.45|
| *8858432* | 2     | 11.10     | 77.9 = 18.9 + 49.2|
|           | 4     | 12.11     | 56.6 = 12.8 + 34.6|
|           | 6     | 15.56     | 47.8 = 11.8 + 25.7|

### 1. KMC3
#### **Command**
```bash
# Running Command
time kmc -k28 -r -t4 -ci1 -w /home/cyr/downloads/pacbio_filtered.fastq NA.res /hom
e/cyr/downloads/
```
#### **Result**

| Dataset   | Thread| RAM (GB)  | Time (sec)    |
| ----      | ----  | ----      | ----          |
| *Ecoli q* | 2     | 1.25      | 9.20 = 4.02 + 5.08|
|           | 4     | 1.25      | 5.14 = 1.66 + 3.47|
|           | 8     | 1.25      | 3.67 = 0.99 + 2.65|
| *8858432* | 2     | 13.50     | 83.0 = 39.5 + 43.4|
|           | 4     | 13.50     | 37.8 = 14.4 + 23.3|
|           | 8     | 13.50     | 24.3 = 7.9 + 16.4|

- *limiting CPU usage won't affect its time performance*
- *mnt in WSL will affect 10%~20% 1st-stage time*




## _
## Method Details
### About Reverse Complement
Since k-mer counting only count the k-mers that is the lexicographically smaller of itself and its reverse complement (RC),
 we need to make sure that the k-mer and its RC k-mer be partitioned into the same bin (i.e., they must have the same 
 minimizer value). Therefore, we need to use canonical minimizer when generating the superkmer. Also, in the counting phase
 , we need to extract the canonical k-mer from the superkmer for counting.  
  
As for minimap2, it keeps minimizer only when minimizer != its RC minimizer. This condition (mm == RCmm) can be kept in 
 k-mer counting but not in minimap2.