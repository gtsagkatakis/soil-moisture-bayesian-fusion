# Soil Moisture Estimation via Bayesian Fusion of Forecasting and Retrieval

This repository contains the code and supplementary material for the paper:

**Fusion of Forecasting and Retrieval for Uncertainty-Aware Soil Moisture Estimation**  
*G. Tsagkatakis, A. Melebari, J. D. Campbell, A. Kannan, P. Shokri, E. Hodges, M. Moghaddam, P.Tsakalides*
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


## Citation
@inproceedings{tsagkatakis2025fusion,
  title={Fusion of Forecasting and Retrieval for Uncertainty-Aware Soil Moisture Estimation},
  author={Tsagkatakis, Grigorios and Melebari, Amer and Campbell, James D. and others},
  booktitle={IGARSS 2025 - IEEE International Geoscience and Remote Sensing Symposium},
  year={2025}
}

