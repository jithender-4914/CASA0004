# Multimodal Spatio-Temporal Fusion: A Generalizable GCN-LSTM with Attention Framework for Urban Applications

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

This repository contains the source code and data for the research paper **"Multimodal Spatio-Temporal Fusion: A Generalizable GCN-LSTM with Attention Framework for Urban Applications"**.

The project develops and evaluates a general-purpose, multimodal spatio-temporal deep learning framework for urban applications. The framework integrates a Graph Convolutional Network (GCN), a Long Short-Term Memory (LSTM) network, and attention mechanisms to model diverse urban phenomena.

## Table of Contents

- [Overview](#overview)
- [Abstract](#abstract)
- [Key Contributions](#key-contributions)
- [Study Area](#study-area)
- [Framework Architecture](#framework-architecture)
- [Case Studies](#case-studies)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup and Usage](#setup-and-usage)
- [Results](#results)
- [Model Details](#model-details)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview

<div align="center">
  <img src="graph&output/Overview%20of%20the%20Research%20Flow.png" alt="Research Flow" width="800"/>
  <p><em>Figure 1: Research workflow overview</em></p>
</div>

## Abstract

Urban big data offers unprecedented opportunities for understanding cities, yet analytical methods remain fragmented and domain-specific. This research addresses this gap by developing and evaluating a **general-purpose, multimodal spatio-temporal deep learning framework** for urban applications.

### Key Innovation
The framework combines:
- **Graph Convolutional Networks (GCN)** for spatial dependency modeling
- **Long Short-Term Memory (LSTM)** networks for temporal dynamics
- **Attention mechanisms** for feature importance weighting

### Validation
Applied to two distinct urban prediction tasks in London:
1. **Crime Forecasting**: Monthly crime counts across categories
2. **Housing Price Estimation**: Quarterly median price changes

### Performance
- **34.15% MAE reduction** in crime prediction vs. LSTM baseline
- **R² = 0.910** for theft prediction (90%+ variance explained)
- **15.01% MAE improvement** in housing price estimation
- **R² = 0.801** for housing price prediction

The results validate that a single, adaptable architecture can effectively model diverse urban phenomena while providing transparency for real-world decision support.

## 🎯 Key Contributions

### 1. **Methodological Innovation**
- Novel hybrid GCN-LSTM-Attention framework engineered for cross-domain generalizability
- Integration of spatial, temporal, and multimodal data streams in unified architecture
- Scalable design applicable to diverse urban prediction tasks

### 2. **Empirical Validation** 
- First rigorous dual-domain validation of spatio-temporal framework
- Demonstrated superior performance across heterogeneous urban applications
- Comprehensive benchmarking against established baselines

### 3. **Practical Impact**
- Actionable interpretability through attention-based feature importance analysis
- Bridge between advanced predictive modeling and policy-relevant insights
- Open-source implementation for reproducible urban analytics

## 🗺️ Study Area

The framework was evaluated using data from **Greater London**, providing a robust testbed for urban analytics with:

- **Spatial Units**: 4,835 Lower Layer Super Output Areas (LSOAs)
- **Population**: ~9 million residents across diverse urban contexts  
- **Area Coverage**: 1,572 km² including central urban, suburban, and peripheral areas
- **Data Richness**: Comprehensive multimodal datasets spanning demographics, infrastructure, and socioeconomic indicators

<div align="center">
  <img src="graph&output/Map%20of%20the%20London%20Study%20Area.png" alt="Study Area" width="700"/>
  <p><em>Figure 2: Lower Layer Super Output Areas (LSOAs) in Greater London study area</em></p>
</div>

### Why London?
London serves as an ideal testbed due to its:
- **Data Availability**: Rich, open datasets across multiple urban domains
- **Spatial Heterogeneity**: Diverse neighborhoods with varying characteristics
- **Policy Relevance**: Active urban planning and public safety initiatives
- **Scale**: Large enough for robust statistical analysis, manageable for computational resources

## 🏗️ Framework Architecture

The model architecture is a **hybrid system** designed to process and fuse spatial, temporal, and external feature information through multiple interconnected components, enabling generalization across diverse urban prediction tasks.

<div align="center">
  <img src="graph&output/Detailed%20Architecture%20of%20the%20Spatio-Temporal%20Framework.png" alt="Framework Architecture" width="900"/>
  <p><em>Figure 3: Detailed architecture of the spatio-temporal GCN-LSTM framework with attention mechanisms</em></p>
</div>

<div align="center">
  <img src="graph&output/model_map.png" alt="Model Structure" width="600"/>
  <p><em>Figure 4: High-level model structure and component relationships</em></p>
</div>

### 🔧 Key Components

#### 1. **Input Processing Layer**
- **Temporal Embeddings**: Projects time-series data into high-dimensional latent space
- **Static Feature Encoding**: Processes external features (demographics, geography, etc.)
- **Feature Normalization**: Ensures consistent scales across heterogeneous data types

#### 2. **Spatio-Temporal Fusion Block**
- **Multi-Head Graph Convolution**: 
  - Captures complex spatial dependencies via geographic proximity graph
  - Multiple attention heads learn different spatial relationship patterns
  - Incorporates neighborhood effects and spatial spillovers

- **Gating Mechanism**: 
  - Adaptive fusion of spatially-aware representations with static features
  - Learns optimal combination weights for different feature types
  - Prevents information loss during feature integration

- **LSTM Temporal Modeling**:
  - Models temporal dynamics from fused spatial-feature representations  
  - Captures both short-term fluctuations and long-term trends
  - Maintains temporal context across prediction horizons

#### 3. **Attention & Prediction Layer**
- **Temporal Attention Mechanism**: 
  - Dynamically weights importance of different time steps
  - Focuses on most relevant historical periods for prediction
  - Provides interpretability for temporal dependencies

- **Multi-Layer Prediction Head**: 
  - Final MLP generates predictions for target variables
  - Dropout regularization prevents overfitting
  - Flexible output layer adapts to different prediction tasks

<div align="center">
  <img src="graph&output/Cross_Attention.png" alt="Attention Mechanism" width="500"/>
  <p><em>Figure 5: Cross-attention mechanism for temporal importance weighting</em></p>
</div>

### 📚 Theoretical Foundation

<div align="center">
  <img src="graph&output/Conceptual%20Diagram%20of%20a%20Graph%20Convolutional%20Network%20(Kipf%20and%20Welling,%202017)..png" alt="GCN Concept" width="600"/>
  <p><em>Figure 6: Conceptual diagram of Graph Convolutional Network operations (Kipf & Welling, 2017)</em></p>
</div>

The framework builds upon established deep learning paradigms:
- **Graph Neural Networks**: Spatial relationship modeling in non-Euclidean domains
- **Recurrent Networks**: Sequential pattern learning for temporal dependencies  
- **Attention Mechanisms**: Selective focus and interpretability enhancement
- **Multi-Task Learning**: Shared representations across domain applications

## Case Studies

The framework's generalizability was tested on two distinct urban prediction tasks in Greater London:

1.  **Urban Crime Forecasting (`ex1_crime/`)**: Predicting monthly crime counts for three categories: Theft, Vehicle Offences, and Violence Against the Person
2.  **Housing Price Estimation (`ex2_housing/`)**: Estimating quarterly median housing prices

### Crime Data Overview

![Crime Trends](graph&output/Crime%20Trends%20in%20London%20Over%20Time%20by%20Category.png)
*Temporal trends in crime categories across London*

### Housing Data Overview

![Housing Trends](graph&output/London%20House%20Price%20Trends%20Over%20Time.png)
*London house price trends over time*

## Dataset

A core component of this research was a principled data fusion pipeline to integrate heterogeneous data for London into a single, model-ready feature matrix at the Lower Layer Super Output Area (LSOA) level.

![Data Processing Pipeline](graph&output/The%20Multimodal%20Data%20Processing%20Pipeline.png)
*The multimodal data processing pipeline for integrating diverse urban datasets*

The integrated dataset includes 15 static external features from five categories:
- **Demographic:** Population, education levels
- **Geographic:** Area, land use diversity  
- **Transport:** Public transport accessibility (PTAL), distance to stations
- **Street Network:** Street length, density, and segment counts
- **Sentiment:** Public sentiment scores derived from Google venue reviews using a BERT model

![Feature Correlation](graph&output/Feature%20Correlation%20Heatmap.png)
*Feature correlation heatmap showing relationships between variables*

The target variables are time-series data for crime counts and housing prices. All data and preprocessing steps can be found in the `data&preprocessing/` directory.

## Repository Structure

- **`data&preprocessing/`**: Contains raw data, data processing notebooks (`data_combination_clean.ipynb`), and the final feature matrices (`gcn_feature_matrix_optimal.csv`).
- **`ex1_crime/`**: Jupyter notebooks for the crime analysis and prediction task (`crime_final.ipynb`, `DSSS_cw_24044734.ipynb`).
- **`ex2_housing/`**: Jupyter notebook for the housing price analysis and estimation task (`housing_final.ipynb`).
- **`graph&output/`**: Contains all figures, plots, and visualizations generated during the research.
- **`LICENSE`**: The license for this project.

## Setup and Usage

### Prerequisites

- Python 3.8+ with Jupyter Notebook or JupyterLab
- Required packages for deep learning, geospatial analysis, and data processing

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/IflyNY2PR/CASA0004.git
    cd CASA0004
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r data&preprocessing/data_science_core_requirements_latest.txt
    ```

3.  **Run the notebooks:**
    Navigate to the experiment directories (`ex1_crime/` or `ex2_housing/`) to explore the analysis notebooks.

### Key Notebooks

- **Crime Analysis:** `ex1_crime/crime_final.ipynb` and `ex1_crime/DSSS_cw_24044734.ipynb`
- **Housing Analysis:** `ex2_housing/housing_final.ipynb`
- **Data Processing:** `data&preprocessing/data_combination_clean.ipynb`

## Results

### Crime Forecasting Performance

The framework demonstrated exceptional performance across all crime categories:

- **34.15% reduction in MAE** for Theft prediction compared to a pure LSTM baseline
- **R² of 0.910** for the Theft category, explaining over 90% of the variance
- Superior performance across Vehicle Offences and Violence Against the Person categories

![Crime Prediction Line](graph&output/crime_prediction_line.png)
*Detailed theft prediction results over time*

### Feature Importance Analysis

The model provides interpretable insights into the drivers of different crime types:

![Feature Importance](graph&output/feature_heatmap.png)

### Housing Price Estimation Performance

- **15.01% MAE improvement** over baseline models
- **R² of 0.801**, demonstrating strong predictive capability
- Successfully captured spatial spillover effects in the housing market
