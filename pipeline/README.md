# Pipeline-Oriented Repository Layout

This folder reorganizes the project around the requested end-to-end workflow:

CSV Files → Load Data → Normalize Data → Create Graph (Adjacency Matrix) → Temporal Embedding → GCN (Spatial Learning) → LSTM (Time Learning) → Attention → Prediction

## Stage directories

- `01_load_data/`
- `02_normalize_data/`
- `03_create_graph_adjacency_matrix/`
- `04_temporal_embedding/`
- `05_gcn_spatial_learning/`
- `06_lstm_time_learning/`
- `07_attention/`
- `08_prediction/`

Each stage contains a short `README.md` that points to the most relevant existing notebooks/scripts/files in this repository.

## Existing raw CSV sources

Primary CSV files are retained in their original locations and referenced by stage docs:

- `ex1_crime/crime_data/`
- `ex2_housing/housing_data/`
- `ex3_transport/transport_data/`
- `data&preprocessing/`

This avoids breaking existing notebook paths while providing a clear pipeline-first structure.
## Quick access
Use the `links/` folder in this stage for direct, pipeline-organized shortcuts to original project files.

