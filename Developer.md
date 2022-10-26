# Developer Handbook
## Hardware Settings
### CPU AMD Ryzen 7 5800H
- 8-core CPU was set at 90% at 2.85GHz
### GPU Nvidia Geforce RTX 3060 Laptop
 - Tensor Core
 - around 70 watts at 1400MHz
```bash
nvidia-smi -pm 1            # enable persistance mode
#nvidia-smi -pl 125         # set power limit to 125W
nvidia-smi -lgc 1400,1400   # lock the gpu clock
nvidia-smi -rgc             # Resets the Gpu clocks to the default values.
```
### Free Linux Memory Buffer/Cache
 - To ensure a cold start every time
 ```bash
 sudo sh -c "free && sync && echo 3 > /proc/sys/vm/drop_caches && free"
 ```

## Method Details
### About Reverse Complement
Since k-mer counting only count the k-mers that is the lexicographically smaller of itself and its reverse complement (RC),
 we need to make sure that the k-mer and its RC k-mer be partitioned into the same bin (i.e., they must have the same 
 minimizer value). Therefore, we need to use canonical minimizer when generating the superkmer. Also, in the counting phase
 , we need to extract the canonical k-mer from the superkmer for counting.  
  
As for minimap2, it keeps minimizer only when minimizer != its RC minimizer. This condition (mm == RCmm) can be kept in 
 k-mer counting but not in minimap2.