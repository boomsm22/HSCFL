# HSCFL: Hybrid Server-Client Federated Recommendation with User Privacy Heterogeneity
This repository contains the official implementation of the paper "HSCFL: Hybrid Server-Client Federated Recommendation with User Privacy Heterogeneity".
## Overview
HSCFL is a hybrid server-client federated learning framework designed for recommendation with user privacy heterogeneity. It separates parameter-efficient training of a shared model from the subsequent optimization of recommendations for open users:

1. Server-client collaborative training (SCCT).
2. Server-side multi-source fusion (SMSF).
## Environment
This code was developed and tested on the following python environment：
```text
python == 3.9.21
pytorch == 1.21.1 (cuda:11.3)
scipy == 1.12.0
numpy == 1.23.5
scikit-learn=1.6.1
bottleneck=1.4.2
```
## Run
We use three real-world datasets: gowalla, yelp2018 and amazon.
Firstly, divide users into two groups, i.e., open users and private users.
```python
python user_divide_preprocess.py --dataset gowalla --rate 0.5
```
Secondly, run HSCFL. We adopt the optimal settings of the backbone models, and tune only the learning rate and framework-specific parameters. Specifically, for the SCCT stage, we tune the ranks of LoRA , the dimensions of the gate and the leaning rate. For the SMSF stage, we tune the distillation weight and the number of the sampled negative items. The best parameters for each dataset are provided in the `HSCFL.sh` file, you can find the corresponding code to run for each dataset in this file.




