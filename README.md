# Dual Lagrangian Learning

This repository contains code to reproduce the results of the NeurIPS 2024 paper [Dual Lagrangian Learning for Conic Optimization](https://neurips.cc/virtual/2024/poster/94146).

* OpenReview: https://openreview.net/forum?id=gN1iKwxlL5
* NeurIPS poster: https://neurips.cc/virtual/2024/poster/94146

## Installation instructions

⚠️ This code requires the CUDA.jl package. It has been tested only on an NVIDIA V100 GPU, with the provided julia environment. 

1. Install julia
2. Install commercial optimization solvers
    * Gurobi: see [Gurobi.jl](https://github.com/jump-dev/Gurobi.jl) instructions. Gurobi provides free academic licenses.
    * Mosek: see [Mosek.jl](https://github.com/MOSEK/Mosek.jl) instructions. Mosek provides free academic licenses.
2. Install julia packages
    ```bash
    julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
    ```

## Running numerical experiments

### Multi-dimensional knapsack

Slurm files are located in `exp/multiknapsack/multiknapsack.sbatch`.

### Production planning

Slurm files are located in `exp/rcprod/rcprod.sbatch`.
