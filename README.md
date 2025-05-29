# Soil Moisture Estimation via Bayesian Fusion of Forecasting and Retrieval

This repository contains the code and supplementary material for the paper:

**Fusion of Forecasting and Retrieval for Uncertainty-Aware Soil Moisture Estimation**  <br>
*G. Tsagkatakis, A. Melebari, J. D. Campbell, A. Kannan, P. Shokri, E. Hodges, M. Moghaddam, P. Tsakalides* <br>
*IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2025* 

## Overview

We present a Bayesian machine learning framework that integrates retrieval and forecasting methods for soil moisture estimation using:

- **CYGNSS GNSS-R observations**
- **SMAP L4 data assimilation forecasts**
- **NGBoost probabilistic regression**

## Repository Structure

- `soil-moisture-bayesian-fusion.py`: Main script implementing the pipeline
- `data/`: Folder for storing or linking input `.npy` files (see below)
- `results/`: Stores output plots and evaluation metrics
- `paper/`: PDF of the published manuscript

## Required packages
numpy
matplotlib
scikit-learn
ngboost
scipy

## Input Data
You will need the following .npy files per site: <br>
- SMAP_L4_<site>.npy <br>
- AUX_all_<site>.npy <br>
- MM_all_<site>.npy <br>
Sites supported: jr1, jr2, jr3, kendall, lucky_hills, z1, z4 <br>
These files are in the data/ directory.


## Citation
If you use this code or find our work useful in your research, please consider citing our paper:

```bibtex
@article{tsagkatakis2024uncertainty,
  author = {G. Tsagkatakis, A. Melebari, J. D. Campbell, A. Kannan, P. Shokri, E. Hodges, M. Moghaddam, P. Tsakalides},
  title = {Fusion of Forecasting and Retrieval for Uncertainty-Aware Soil Moisture Estimation},
  journal = {IGARSS 2025 - IEEE International Geoscience and Remote Sensing Symposium},
  year = {2025},
}


