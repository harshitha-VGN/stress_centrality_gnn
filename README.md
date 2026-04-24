# Stress Centrality GNN

## Overview
This repository contains a PyTorch and PyTorch Geometric implementation of a Graph Neural Network (GNN) designed to approximate and predict edge additions that maximally reduce the **Stress Centrality** of an overall graph. 

Because calculating exact stress centrality changes for every potential edge addition is highly computationally expensive, this solution formulates the problem as a machine learning task. It uses fundamental structural node properties combined with GraphSAGE convolutions to predict whether a candidate edge reduces stress and ranks these candidates to estimate the optimal edge addition.

## Tech Stack
- **Python 3**
- **PyTorch** & **PyTorch Geometric (PyG)**
- **NetworkX** (graph topology & feature operations)
- **Scikit-Learn** (data normalization & splitting)
- **Matplotlib / Seaborn** (data visualization tools)

## Key Features
1. **Diverse Synthetic Dataset Generation**: Automatically generates synthetic Barabási–Albert, Erdős–Rényi, Path, and Tree graphs. The code exhaustively simulates all non-edge additions to establish ground-truth stress centrality reductions.
2. **Robust Node Feature Pipeline**: Computes meaningful topological features for each node:
   - Degree Centrality
   - Eigenvector Centrality
   - Closeness Centrality
   - PageRank
   - Clustering Coefficient
   - Harmonic Centrality
3. **Graph Neural Network Architecture (`StressMinimizationGNN`)**: A dual-head architecture using `SAGEConv` layers:
   - **Classification Head**: Predicts if a given target edge successfully reduces stress (Binary Classification).
   - **Ranking/Regression Head**: Ranks target edges by estimating the magnitude of their stress reduction.
4. **Custom Ranking Loss**: Utilizes a margin-based pairwise ranking loss over edge addition reductions to capture ordering properly and handle ties (e.g., structural equivalences).

## Repository Contents
- **`diff_techniques.ipynb`**: The primary Jupyter Notebook containing the entire end-to-end pipeline. This includes:
  - Dataset and ground-truth generation
  - Multi-feature processing
  - Model definitions
  - Batched PyTorch data loaders preparation
  - Training and validation loops

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/harshitha-VGN/stress_centrality_gnn.git
   cd stress_centrality_gnn
   ```
2. Install the required dependencies:
   ```bash
   pip install torch torchvision torchaudio 
   pip install torch-geometric networkx pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the complete pipeline via the Jupyter Notebook `diff_techniques.ipynb`. By default, the notebook automatically generates synthetic graph datasets and stores them in a local `graph_data/` directory.
