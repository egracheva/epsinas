# Zero-shot NAS with *epsinas* metric

<div align=justify>

This is the official repository of the `epsinas` zero-shot NAS metric.  
This repository contains the code reproducing the results of [our paper](https://arxiv.org/abs/2302.04406).  

<h3> Quick Links: </h3>

[**Summary**](#summary)

[**Setup**](#setup)

[**Data**](#data)

  - [**Release data**](#release-data)

  - [**Benchmark data**](#benchmark-data)

[**Reproducibility**](#reproducibility)

## Summary

*epsinas* evaluates neural architecture without training, gradients compuration and true labels. The key feature of the metric is that it requires only two initializations with constant shared weights. It turns out that the MAE between the raw outputs of two initializations is a good estimator for the architecture performance, as we show in [our paper](https://arxiv.org/abs/2302.04406).

It can also be readily implemented within other NAS frameworks. In our paper, we use it together with random search (RS) and ageing evolution (EA).

## Setup
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

## Data

#### Release data

All the results of our study, including figures and data-files for each ablation study can be on Hugging Face:

[egracheva/epsinas-release-data](https://huggingface.co/datasets/egracheva/epsinas-release-data)

Size: ~330 MB

Note: There is no need to download the data to reproduce our results.  
The required files are automatically loaded within the code.  
However, you **will** need the benchmark NAS data and the datasets used for training (see below).

### Benchmark data

#### NAS-Bench-101

Download [NAS-Bench-101](https://github.com/google-research/nasbench) benchmark data. The file corresponds to the subset of the dataset with only models trained at 108 epochs:

https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord

Size: ~499 MB, SHA256: 4c39c3936e36a85269881d659e44e61a245babcb72cb374eacacf75d0e5f4fd1

Please place the file in the `epsinas/NAS-Bench-101` directory.

#### NAS-Bench-201

Download the [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201) API file:

https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs

Please place the file in the `epsinas/NAS-Bench-201/api` directory.

Download the image data from [Google drive](https://drive.google.com/drive/folders/1L0Lzq8rWpZLPfiQGd6QR8q5xLV88emU7) and put it into the `epsinas/NAS-Bench-201/datasets` directory.

#### NAS-Bench-NLP

Download the [data.zip](https://drive.google.com/file/d/1eEWo4KEUEkQZcBOpzlj6xGV8ESPNYrRh), [train_logs_single_run.zip](https://drive.google.com/file/d/1GeydyYJ2o_j72cRhSicRH5Qk1W-tPrJ3) and [train_logs_multi_runs.zip](https://drive.google.com/file/d/1GE6MXjuUGX5X0iGn0LZ0Tp8GQ7uHdFGl) files.
Unzip them into the `data`, `train_logs_single_run`, `train_logs_multi_runs` directories, respectively.

## Reproducibility

Reproduce all of the results by running the Notebooks in the corresponding directories:

```
- NAS-Bench-201
  -- Reproduce.ipynb reproduces **epsinas** metric computation for the whole search space
  -- Graphics.ipynb reproduces figures and tables reported in the paper
  -- Ablation.ipynb reproduces our ablation studies on weights and computational performance

- NAS-Bench-201
  -- Reproduce.ipynb reproduces **epsinas** metric computation for the whole search space
  -- Graphics.ipynb reproduces figures and tables reported in the paper
  -- Ablation.ipynb reproduces our ablation studies on weights, batch size, synthetic data and initializations
  -- Integration.ipynb integrates `epsinas` into random search and ageing evoluation

- NAS-Bench-NLP
  -- Reproduce.ipynb reproduces **epsinas** metric computation for the whole search space
  -- Graphics.ipynb reproduces figures and tables reported in the paper
  -- Ablation.ipynb reproduces our ablation studies on weights, embeddings and batch size
```

For each space, *epsinas* implementation can be found in the `epsinas-utils.py` file.
