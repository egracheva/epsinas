# Zero-shot NAS with `epsinas` metric


<div align=justify>

This is the official repository of the `epsinas` zero-shot NAS metric.
This repository contains the code reproducing our results. It may also help you to implement the metric within your own framework.

<h3> Quick Links: </h3>

[**Summary**](#summary)
[**Setup**](#setup)
[**Data**](#data)

# Summary

`epsinas` evaluates neural architecture without training, gradients compuration and true labels. The key feature of the metric is that it requires only two initializations with constant shared weights. It turns out that the MAE between the raw outputs of two initializations is a good estimator for the architecture performance, as we show in [our paper](https://arxiv.org/abs/2302.04406).

It can also be readily implemented within other NAS frameworks. In our paper, we use it together with random search (RS) and ageing evolution (EA).

# Setup
1. [Install PyTorch GPU/CPU](https://pytorch.org/get-started/locally/) for your setup.
2. Clone and setup `epsinas`
To run our code we recommend creating a new conda environment. 

```bash
git clone -b epsinas https://github.com/egracheva/epsinas
cd epsinas
conda create -n epsinas  python=3.6
conda activate epsinas
```

Run setup.py file with the following command, which will install all the packages listed in [`requirements.txt`](requirements.txt).
```bash
pip install -e .
```

# Data

Download all the NAS benchmarks and their associated datasets ( for mac users, please make sure you have wget installed).
```bash
source scripts/bash_scripts/download_data.sh all 
```
Alternatively, you can download the benchmark for a specific search space and dataset/task as follows:
```bash
source scripts/bash_scripts/download_data.sh <search_space> <dataset> 
source scripts/bash_scripts/download_data.sh nb201 cifar10
source scripts/bash_scripts/download_data.sh nb201 all 
```


Download all the `epsinas` proxies evaluations, which contains the scores for each proxy and validation accuracy for each architecutre. The ```gdown (pip install gdown)``` package is required to download from google drive. The following command will download the data.

```bash
source scripts/bash_scripts/download_epsinas_scores.sh <search_space>
source scripts/bash_scripts/download_epsinas_scores.sh nb201
source scripts/bash_scripts/download_epsinas_scores.sh nb101
source scripts/bash_scripts/download_epsinas_scores.sh nbnlp
source scripts/bash_scripts/download_epsinas_scores.sh all
```
