# Multimodal Spatio-Temporal Fusion: A Generalizable GCN-LSTM with Attention Framework for Urban Applications

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

This repository contains the source code and data for the research paper **"Multimodal Spatio-Temporal Fusion: A Generalizable GCN-LSTM with Attention Framework for Urban Applications"**.

The project develops and evaluates a general-purpose, multimodal spatio-temporal deep learning framework for urban applications. The framework integrates a Graph Convolutional Network (GCN), a Long Short-Term Memory (LSTM) network, and attention mechanisms to model diverse urban phenomena.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{#undo
}
```

### Institutional Affiliation
- **Institution**: University College London (UCL)
- **Department**: Centre for Advanced Spatial Analysis (CASA)
- **Research Group**: MRes Urban Spatial Science

---

**Last Updated**: September 2025  
**Version**: 0.9.9  
**Status**: Active Developmentnal Network (GCN), a Long Short-Term Memory (LSTM) network, and attention mechanisms to model diverse urban phenomena.

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

## 📊 Case Studies

The framework's **generalizability** was rigorously tested across two heterogeneous urban prediction domains, demonstrating cross-domain applicability and robust performance.

### 🚔 Case Study 1: Urban Crime Forecasting (`ex1_crime/`)

**Objective**: Predict monthly crime counts across three major categories
- **Theft**: Property crimes, burglary, shoplifting
- **Vehicle Offences**: Car theft, vehicle interference  
- **Violence Against the Person**: Assault, harassment, threats

**Data Characteristics**:
- **Temporal Resolution**: Monthly aggregation (2019-2023)
- **Spatial Coverage**: All 4,835 London LSOAs
- **Target Variable**: Crime count per LSOA per month
- **Challenge**: High variability, seasonal patterns, spatial clustering

<div align="center">
  <img src="graph&output/Crime%20Trends%20in%20London%20Over%20Time%20by%20Category.png" alt="Crime Trends" width="800"/>
  <p><em>Figure 7: Temporal trends in crime categories across London showing seasonal patterns and COVID-19 impact</em></p>
</div>

### 🏠 Case Study 2: Housing Price Estimation (`ex2_housing/`)

**Objective**: Estimate quarterly median housing prices at LSOA level
- **Market Dynamics**: Supply-demand interactions, gentrification effects
- **Spatial Spillovers**: Neighborhood price influences and contagion
- **Economic Factors**: Interest rates, policy changes, market sentiment

**Data Characteristics**:
- **Temporal Resolution**: Quarterly aggregation (2019-2023)
- **Spatial Coverage**: All 4,835 London LSOAs  
- **Target Variable**: Median house price per LSOA per quarter
- **Challenge**: Non-linear price dynamics, spatial autocorrelation, market volatility

<div align="center">
  <img src="graph&output/London%20House%20Price%20Trends%20Over%20Time.png" alt="Housing Trends" width="800"/>
  <p><em>Figure 8: London house price trends over time showing market cycles and regional variations</em></p>
</div>

### 🔄 Cross-Domain Validation Strategy

| Aspect | Crime Forecasting | Housing Estimation |
|--------|-------------------|-------------------|
| **Temporal Pattern** | High-frequency, seasonal | Low-frequency, cyclical |
| **Spatial Structure** | Clustered hotspots | Smooth gradients |  
| **Feature Importance** | Demographics, accessibility | Economics, amenities |
| **Interpretability** | Policy intervention points | Market drivers |

This dual-domain approach ensures the framework's **robustness** across different:
- **Data distributions** (count vs. continuous)
- **Temporal dynamics** (monthly vs. quarterly)  
- **Spatial patterns** (clustered vs. smooth)
- **Domain knowledge** (criminology vs. economics)

## 📊 Dataset

A core innovation of this research is the **principled data fusion pipeline** that integrates heterogeneous urban datasets into a unified, model-ready feature matrix at the LSOA level. This comprehensive dataset enables robust spatio-temporal modeling across diverse urban applications.

<div align="center">
  <img src="graph&output/The%20Multimodal%20Data%20Processing%20Pipeline.png" alt="Data Processing Pipeline" width="900"/>
  <p><em>Figure 9: Multimodal data processing pipeline for integrating diverse urban datasets</em></p>
</div>

### 🏗️ Feature Engineering

The integrated dataset comprises **15 static external features** across five thematic categories:

#### 1. **Demographics** (3 features)
- Population density and composition
- Education levels (higher education percentage)
- Age distribution characteristics

#### 2. **Geographic Context** (2 features)  
- **Area**: LSOA spatial extent (km²)
- **Land Use Diversity**: Shannon diversity index of land use types

#### 3. **Transportation Access** (4 features)
- **PTAL Score**: Public Transport Accessibility Level (0-6 scale)
- **Station Distance**: Proximity to nearest rail/tube stations
- **Transport Density**: Number of transport nodes per area
- **Connectivity Index**: Network centrality measures

#### 4. **Street Network** (4 features)
- **Street Length**: Total road network length per LSOA
- **Street Density**: Road length per unit area
- **Intersection Count**: Number of road junctions
- **Segment Density**: Road connectivity measure

#### 5. **Public Sentiment** (2 features)
- **Venue Sentiment**: Public sentiment scores from Google venue reviews
- **Sentiment Processing**: BERT-based natural language processing
- **Aggregation**: Mean sentiment per LSOA with spatial interpolation

### 🔗 Feature Relationships

<div align="center">
  <img src="graph&output/Feature%20Correlation%20Heatmap.png" alt="Feature Correlation" width="700"/>
  <p><em>Figure 10: Feature correlation heatmap revealing multi-dimensional urban relationships</em></p>
</div>

**Key Insights from Correlation Analysis**:
- **Transport-Demographics**: Strong correlation between accessibility and population density
- **Network-Economic**: Street connectivity correlates with housing prices
- **Sentiment-Safety**: Public sentiment negatively correlates with crime rates
- **Geographic Constraints**: Area size inversely related to urban intensity metrics

### 📈 Target Variables

#### **Crime Data** (`ex1_crime/`)
- **Source**: Metropolitan Police Service open data
- **Temporal Coverage**: 2019-2023 (monthly)
- **Categories**: Theft, Vehicle Offences, Violence Against Person
- **Preprocessing**: Spatial aggregation to LSOA, outlier detection, missing value imputation

#### **Housing Prices** (`ex2_housing/`)
- **Source**: UK Land Registry, Rightmove, Zoopla
- **Temporal Coverage**: 2019-2023 (quarterly)
- **Metric**: Median house prices per LSOA
- **Preprocessing**: Price normalization, seasonal adjustment, market trend removal

### 📁 Dataset Statistics

- **Spatial Units**: 4,835 LSOAs
- **Temporal Points**: 48 months (crime) / 16 quarters (housing)
- **Feature Dimensions**: 15 static + temporal targets

## ⚙️ Model Details

### 🏗️ Architecture Specifications

#### **Key Components**

<div align="center">

**Table 1: List of Symbols and Acronyms**

</div>

| Symbol | Description | Context |
|--------|-------------|---------|
| **Greek Letters** | | |
| σ | Activation function | GCN formula |
| α<sub>t</sub> | Attention weight | Temporal attention |
| ⊙ | Element-wise product | Gating mechanism |
| **Model Components** | | |
| G, V, E | Graph, Vertices, Edges | Foundational structure |
| N | Number of nodes (LSOAs) | Graph size |
| A | Adjacency matrix | Defines connections |
| Â | A+I | Self-loops |
| D̂ | Degree matrix of Â | GCN normalization |
| X<sub>temp</sub> | Time-series input | Primary model input |
| X<sub>ext</sub> | External features input | Primary model input |
| B, T, F | Batch, Time, Features | Input dimensions |
| D<sub>emb</sub> | Embedding dimension | Latent space size |
| h<sub>t</sub> | Hidden state at time t | LSTM output |
| c | Context vector | Aggregation |
| Ŷ | Predicted value | Model output |
| **Acronyms** | | |
| GCN | Graph Convolutional Network | Spatial modeling |
| LSTM | Long Short-Term Memory | Temporal modeling |
| BERT | Bidirectional Encoder Representations from Transformers | Sentiment analysis |
| PTAL | Public Transport Accessibility Level | Key input feature |
| LSOA | Lower Layer Super Output Area | Spatial unit of analysis |

#### **Computational Requirements**
- **Training Time**: 
  - Crime Model: 150 minutes (NVIDIA A100 GPU) + 600 minutes (Google Colab TPU v6e-1)
  - Housing Model: 60 minutes (NVIDIA A100 GPU)
- **Memory Usage**: 40GB(GPU) 200GB(TPU) RAM during training

### 🔬 Implementation Details

#### **Graph Construction**
- **Spatial Graph**: K-nearest neighbors (k=8) based on geographic distance
- **Edge Weights**: Inverse distance weighting with gaussian kernel
- **Graph Preprocessing**: Laplacian normalization for stable training
- **Dynamic Edges**: Optional temporal edge reweighting based on feature similarity

#### **Data Preprocessing Pipeline**
1. **Spatial Aggregation**: Point data → LSOA polygons using spatial joins
2. **Temporal Alignment**: Irregular timestamps → regular monthly/quarterly grids  
3. **Feature Scaling**: Min-max normalization with outlier clipping (±3σ)
4. **Missing Data**: Spatial interpolation followed by temporal forward-fill
5. **Graph Augmentation**: Random edge dropout (10%) during training for robustness

#### **Training Strategy**
- **Loss Weighting**: Adaptive loss balancing for multi-scale targets
- **Curriculum Learning**: Progressive increase in sequence length  
- **Data Augmentation**: Temporal jittering, spatial noise injection
- **Validation**: Time-series cross-validation with walk-forward approach

## 📁 Repository Structure

```
CASA0004/
├── 📊 data&preprocessing/           # Data pipeline and feature engineering
│   ├── data_combination_clean.ipynb     # Main data processing notebook
│   ├── gcn_feature_matrix_optimal.csv  # Final model-ready dataset
│   ├── feature_heatmap.ipynb           # Feature analysis and visualization
│   ├── create_core_requirements.py     # Environment management
│   └── [subdirectories]/               # Raw data organized by theme
│       ├── economic/                    # Economic indicators
│       ├── Infrastructure/              # Transport and utilities
│       ├── landuse/                     # Land use classifications  
│       ├── social/                      # Demographics and social data
│       └── shapefiles/                  # Spatial boundary files
│
├── 🚔 ex1_crime/                   # Crime forecasting experiment
│   ├── crime_final.ipynb               # Streamlined crime analysis  
│   ├── DSSS_cw_24044734.ipynb         # Complete research notebook
│   ├── crime_ml.ipynb                  # Model development and testing
│   ├── crime_data/                     # Crime-specific datasets
│   └── saved_models/                   # Trained model checkpoints
│
├── 🏠 ex2_housing/                 # Housing price estimation experiment  
│   ├── housing_final.ipynb             # Housing analysis and modeling
│   └── housing_test.ipynb              # Model validation experiments
│
├── 📈 graph&output/                # Visualizations and results
│   ├── [architecture_diagrams]/        # Framework visualization
│   ├── [performance_plots]/            # Results and metrics
│   └── [analysis_figures]/             # Feature analysis plots
│
├── 📋 README_files/                # Documentation assets
└── 📄 [root_files]                 # Project metadata
    ├── README.md                       # This documentation  
    ├── LICENSE                         # MIT License
    └── README.html                     # HTML version
```

### 🔧 Key Files Description

| File/Directory | Purpose | Key Outputs |
|----------------|---------|-------------|
| `data_combination_clean.ipynb` | Master data pipeline | Feature matrices, documentation |
| `crime_final.ipynb` | Crime model implementation | Performance metrics, predictions |  
| `housing_final.ipynb` | Housing model implementation | Price estimates, validation results |
| `gcn_feature_matrix_optimal.csv` | Model-ready dataset | 15 features × 4,835 LSOAs |
| `saved_models/` | Trained checkpoints | Deployable model weights |

## 🚀 Setup and Usage

### 📋 Prerequisites

#### **System Requirements**
- **Python**: 3.8+ (recommended: 3.9 or 3.10)
- **Memory**: Google Colab recommended for full dataset

#### **Software Dependencies**
- **Deep Learning**: PyTorch, PyTorch Geometric for GNN implementation  
- **Geospatial**: GeoPandas, Shapely, Folium for spatial analysis
- **Data Science**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

### ⚡ Quick Start

#### 1. **Clone Repository**
```bash
git clone https://github.com/IflyNY2PR/CASA0004.git
cd CASA0004
```

#### 2. **Environment Setup**
```bash
# Create virtual environment (recommended)
python -m venv casa0004_env
source casa0004_env/bin/activate  # On Windows: casa0004_env\Scripts\activate

# Install dependencies
pip install -r data&preprocessing/data_science_core_requirements_latest.txt

# Verify installation
python -c "import torch, torch_geometric; print('Setup successful!')"
```

#### 3. **Data Preparation** 
```bash
# Navigate to data processing
cd data&preprocessing/

# Run data pipeline (optional - processed data included)
jupyter notebook data_combination_clean.ipynb
```

#### 4. **Model Execution**
```bash
# Crime forecasting
cd ex1_crime/
jupyter notebook crime_final.ipynb

# Housing price estimation  
cd ex2_housing/
jupyter notebook housing_final.ipynb
```

## 📈 Results

### 🚔 Crime Forecasting Performance

The framework achieved **state-of-the-art performance** across all crime categories, significantly outperforming baseline models:

#### **Quantitative Results**
| Crime Category | MAE Reduction | R² Score | RMSE | MAPE |
|----------------|---------------|----------|------|------|
| **Theft** | **34.15%** | **0.910** | 2.84 | 12.3% |
| **Vehicle Offences** | **28.7%** | **0.847** | 1.92 | 15.1% |
| **Violence** | **31.2%** | **0.883** | 3.15 | 14.8% |

#### **Baseline Comparisons**
- **vs. LSTM**: 25-35% improvement across all metrics
- **vs. Simple GCN**: 15-25% enhancement from temporal modeling

<div align="center">
  <img src="graph&output/crime_prediction_line.png" alt="Crime Prediction Results" width="800"/>
  <p><em>Figure 11: Detailed theft prediction results showing model accuracy over time periods</em></p>
</div>

### 🏠 Housing Price Estimation Performance

#### **Market Prediction Results**  
- **15.01% MAE improvement** over baseline econometric models
- **R² = 0.801** explaining 80%+ of price variance
- **MAPE = 8.7%** well within acceptable forecasting bounds

### 🔍 Feature Importance Analysis

The framework provides **interpretable insights** into urban dynamics through attention-based feature importance:

<div align="center">
  <img src="graph&output/feature_heatmap.png" alt="Feature Importance" width="800"/>
  <p><em>Figure 12: Feature importance heatmap across different prediction tasks and geographic areas</em></p>
</div>

#### **Crime Drivers** (Top 5)
1. **Transport Access** (0.23): High PTAL areas show different crime patterns
2. **Population Density** (0.19): Dense areas have higher theft rates
3. **Street Connectivity** (0.17): Well-connected areas enable quick escape routes
4. **Public Sentiment** (0.15): Negative sentiment correlates with crime hotspots  
5. **Education Levels** (0.12): Higher education areas have lower crime rates