## Notice
- This is a **customed-implementation** version of NetCLR [1] framework to check the performance of
augmented network traces in realistc website fingerprinting w/ own manipulation rules. The original code is splitted into several python files for convenience.
- If you implement this code, specify the path of `DATA_ABSOLUTE_PATH` and `LOG_ABSOLUTE_PATH` in main.py.
(* Also you can add other rules in `Augmentor` at `augmentor.py`)
- This code is re-implemented that fits to 'sizes' and 'timestamps' features unlike the original code, 'bursts'.
## Usage Examples
```
python main.py --o "models/nj_4_e_" --logs_name "nj_4_e_100.txt" -b 256 -n 100 -t 0.5 --method 0
```
## Directory Structure
```
\---src
    |   augmentor.py
    |   backbone.py
    |   common.py
    |   main.py
    |   netclr.py
    |   usage.py
    |   
    +---finetuning
    |   |   backbone.py
    |   |   cw_train.py
    |   |   graphing.py
    |   |   main.py
    |   |   single.py
    |   |
    |   +---log
    |   +---results
    |
    +---logs
    +---models
    +---others
    |   \---cfg2
    |           main2.py
```
## References
[1]
Alireza Bahramali, Ardavan Bozorgi, and Amir Houmansadr. 2023. Realistic Website Fingerprinting By Augmenting Network Traces. In Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security (CCS ’23), November 26–30, 2023, Copenhagen, Denmark. ACM, New York, NY, USA, 15 pages.
https://doi.org/10.1145/3576915.3616639 

```
@inproceedings{3576915.3616639,
author = {Bahramali, Alireza and Bozorgi, Ardavan, and Houmansadr, Amir},
title = {Realistic Website Fingerprinting By Augmenting Network Traces},
booktitle = {Proceedings of
the 2023 ACM SIGSAC Conference on Computer and Communications Security},
series = {CCS '23},
year = {2023},
location = {Copenhagen, Denmark},
numpages = {15},
url = {https://doi.org/10.1145/3576915.3616639},
doi = {10.1145/3576915.3616639},
publisher = {ACM},
address = {New York, NY, USA},
}
```
