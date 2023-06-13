![](docs/src/images/path_skeleton.png#gh-light-mode-only)
![](docs/src/images/path_skeleton_dark.png#gh-dark-mode-only)

# StochasticAD

[![Build Status](https://github.com/gaurav-arya/StochasticAD.jl/workflows/CI/badge.svg?branch=main)](https://github.com/gaurav-arya/StochasticAD.jl/actions?query=workflow:CI)
[![](https://img.shields.io/badge/docs-main-blue.svg)](https://gaurav-arya.github.io/StochasticAD.jl/dev/)
[![arXiv article](https://img.shields.io/badge/article-arXiv%3A10.48550-B31B1B)](https://arxiv.org/abs/2210.08572)

StochasticAD is an experimental, research package for automatic differentiation (AD) of stochastic programs. It implements AD algorithms for handling programs that can contain *discrete* randomness, based on the methodology developed in [this NeurIPS 2022 paper](https://doi.org/10.48550/arXiv.2210.08572). We're still working on docs and code cleanup!

## Installation

The package can be installed with the Julia package manager:

```julia
julia> using Pkg;
julia> Pkg.add("StochasticAD");
```

## Citation

```
@inproceedings{arya2022automatic,
 author = {Arya, Gaurav and Schauer, Moritz and Sch\"{a}fer, Frank and Rackauckas, Christopher},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {10435--10447},
 publisher = {Curran Associates, Inc.},
 title = {Automatic Differentiation of Programs with Discrete Randomness},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/43d8e5fc816c692f342493331d5e98fc-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```
