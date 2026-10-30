# Diffusion-Driven High-Dimensional Variable Selection

This repository contains an implementation of the paper

- Wang, M., Shen, X., & Pan, W. (2026+). Diffusion-Driven High-Dimensional Variable Selection.

Diffusion-Driven High-Dimensional Variable Selection studies the problem of leveraging multiple synthetic data sets generated from diffusion models to improve variable selection by selecting stable features.

## Directory Summary

- scripts: contains scripts and functions to perform Diffusion-Driven High-Dimensional Variable Selection.
- data: contains the observed data.
- exp: contains base configurations for tuning TabDDPM, model outputs, and generated synthetic data sets.

## How to Use

1. Follow the installation instructions provided in the official repository to download and install TabDDPM: Modelling Tabular Data with Diffusion Models:
   https://github.com/yandex-research/tab-ddpm.
2. Download this entire repository.
3. Run example usage file:
```bash
python scripts/sim_pipeline_lasso_autoreg_p100.py
```
