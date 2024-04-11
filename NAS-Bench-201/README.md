# Epsilon NAS for NAS-Bench-201 benchmark data

The code in this directory is a modified version of the codebase from [Mellor et al.](https://arxiv.org/abs/2006.04647).\
The metrics computation is modified to implement the epsilon metric.

## Usage 

Download the NASbench201 data (see https://github.com/D-X-Y/NAS-Bench-201)

Reproduce all of the results by running `Reproduce.ipynb` notebook.\
Reproduce all the plots by running `Plots_and_Tables.ipynb` notebook.\
Reproduce ablation studies with `Ablations.ipynb` notebook.\
Integration with other search arlgorithms can be found in the `Integration.ipynb` notebook.
