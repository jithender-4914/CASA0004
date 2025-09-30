# Multimodal Spatio-Temporal Fusion: A Generalizable GCN-LSTM with Attention Framework for Urban Application

This repository contains the source code and data for the research paper, "Multimodal Spatio-Temporal Fusion: A Generalizable GCN-LSTM with Attention Framework for Urban Application".

The project develops and evaluates a general-purpose, multimodal spatio-temporal deep learning framework for urban applications. The framework integrates a Graph Convolutional Network (GCN), a Long Short-Term Memory (LSTM) network, and attention mechanisms to model diverse urban phenomena.

![Research Flow](graph&output/Overview%20of%20the%20Research%20Flow.png)
*Research methodology and workflow overview*

## Abstract

The proliferation of urban big data presents unprecedented opportunities for understanding cities, yet the analytical methods to harness this data are often fragmented and domain-specific. This paper confronts this critical gap by developing and rigorously evaluating a general-purpose, multimodal spatio-temporal deep learning framework. The framework integrates a Graph Convolutional Network (GCN) to capture spatial dependencies, a Long Short-Term Memory (LSTM) network for temporal dynamics, and attention mechanisms to focus on salient features. To validate its generalizability, the framework was applied to two distinct urban prediction tasks in London: forecasting monthly crime counts and estimating quarterly housing price changes. The model demonstrated superior performance over baselines in both tasks, validating the thesis that a single, adaptable architecture can effectively model diverse urban phenomena while offering the transparency required for real-world decision support.

## Key Contributions

1.  **Methodological:** Design and implementation of a novel, hybrid GCN-LSTM-Attention framework engineered for generalizability, integrating spatial, temporal, and multimodal data streams.
2.  **Empirical:** The first rigorous, dual-domain validation of such a framework, demonstrating superior performance on both crime forecasting and housing price estimation.
3.  **Practical:** Demonstration of actionable interpretability through feature-importance analysis, bridging the gap between predictive modeling and policy-relevant insights.

## Study Area

The framework was evaluated using data from Greater London, comprising 4,835 Lower Layer Super Output Areas (LSOAs) as the spatial units for analysis.

![Study Area](graph&output/Map%20of%20the%20London%20Study%20Area.png)
*Study area: Lower Layer Super Output Areas (LSOAs) in Greater London*

## Framework Architecture

The model architecture is a hybrid system designed to process and fuse spatial, temporal, and external feature information through multiple interconnected components.

![Framework Architecture](graph&output/Detailed%20Architecture%20of%20the%20Spatio-Temporal%20Framework.png)
*Detailed architecture of the spatio-temporal GCN-LSTM framework with attention mechanisms*

![Model Structure](graph&output/model_map.png)
*High-level model structure and component relationships*

### Key Components

- **Input Embeddings:** Projects temporal and static external features into a high-dimensional latent space
- **Spatio-Temporal Fusion Block:**
    - **Multi-Head GCN:** Captures complex spatial dependencies from a geographic proximity graph
    - **Gating Mechanism:** Fuses the spatially-aware representations with static external features  
    - **LSTM:** Models temporal dynamics from the sequence of fused representations
- **Temporal Aggregation and Prediction:**
    - **Temporal Attention:** Weighs the importance of different time steps in the sequence
    - **Prediction Head:** A final MLP that generates the prediction for the target variable

### Theoretical Foundation

![GCN Concept](graph&output/Conceptual%20Diagram%20of%20a%20Graph%20Convolutional%20Network%20(Kipf%20and%20Welling,%202017)..png)
*Conceptual diagram illustrating Graph Convolutional Network operations (Kipf and Welling, 2017)*

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
