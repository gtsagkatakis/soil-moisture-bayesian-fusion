# Soil Moisture Estimation via Bayesian Fusion of Forecasting and Retrieval

This repository contains the code and supplementary material for the paper:

**Fusion of Forecasting and Retrieval for Uncertainty-Aware Soil Moisture Estimation**  
*G. Tsagkatakis et al., IGARSS 2025*

## Overview

We present a Bayesian machine learning framework that integrates retrieval and forecasting methods for soil moisture estimation using:

- **CYGNSS GNSS-R observations**
- **SMAP L4 data assimilation forecasts**
- **NGBoost probabilistic regression**

## Repository Structure

- `BM_all_sites_testing_forecasting.py`: Main script implementing the pipeline
- `data/`: Folder for storing or linking input `.npy` files (see below)
- `results/`: Stores output plots and evaluation metrics
- `paper/`: PDF of the published manuscript

## Setup

```bash
pip install -r requirements.txt
