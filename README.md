# 🧥 Predicting Resale Prices in Secondhand Fashion using Machine Learning and Graph Neural Networks

**Project by**: Piangpim Chancharunee, Chiao-Kai Chiang, Ruxi He, I-Hsun Lu  
**Course**: Machine Learning in Network Science — Final Project  
**Dataset**: [Vestiaire Collective Fashion Dataset on Kaggle](https://www.kaggle.com/datasets/justinpakzad/vestiaire-fashion-dataset)

## 📌 Overview

This project aims to **predict the resale price of secondhand fashion items** on platforms like Vestiaire Collective. We explore both **traditional machine learning** and **graph-based deep learning** techniques to model the complex relationships among product listings and enhance prediction accuracy.

## 🎯 Objectives

- Predict resale prices using structured product attributes.
- Compare the performance of:
  - Random Forest (RF)
  - Graph Convolutional Networks (GCNs)
  - GraphSAGE
- Investigate the impact of product-level similarities (e.g., brand, material) modeled as graphs.

## 👗 Use Case & Motivation

- Help **sellers** set competitive, data-informed prices.
- Support **buyers** in identifying good-value investments.
- Enable **platforms** to optimize pricing suggestions, recommendations, and search rankings.
- Promote **sustainability** through better resale circulation of durable fashion items.

## 🧠 Methods

### Data Preprocessing

- Cleaned and normalized categorical variables using fuzzy string matching.
- Mapped fine-grained `product type` into higher-level `broad type`.
- Selected key features:  
  `product type`, `broad type`, `gender`, `category`, `color`, `brand`, `condition`, `material`, `like count`

### Graph Construction

Edges were added between items based on:
- Shared brand, category, or material.
- Similar user engagement (via `like count`).

### Models

| Model        | Description                                                   |
|--------------|---------------------------------------------------------------|
| **Random Forest** | Baseline ML model trained on tabular data                   |
| **GCN**          | Graph model with neighborhood aggregation                   |
| **GraphSAGE**    | Scalable GNN with flexible neighborhood sampling and aggregation |

## 📈 Results

| Model       | R² Score | MAE    | MSE         |
|-------------|----------|--------|-------------|
| Random Forest | 0.13     | 250.17  | 875, 084.94  |
| GCN          | 0.05     | 355.77 | 644,539.81  |
| GraphSAGE    | **0.33** | 283.84 | 455,572.88  |

GraphSAGE significantly outperformed both GCN and Random Forest, showing strong capabilities in sparse, noisy graph data environments.

## 🔍 Key Insights

- **Product like count**, **brand**, **material**, and **condition** are key predictors.
- **GraphSAGE** is robust under graph sparsity and outperforms GCN.
- Likes are informative but only available after listing; future models for sellers should exclude this.

## ⚠️ Limitations

- Price is based on listing price, not transaction price.
- Graphs were downsampled, reducing connectivity.
- Image and text data not used.
- Likes are post-listing signals—not suitable for initial price prediction.

## 🔮 Future Work

- Try scalable GNN architectures like **LightGCN**.
- Add multimodal inputs (e.g., product images or text descriptions).
- Improve graph connectivity via better sampling strategies or user-behavior-based edges.

## 📁 Project Structure
├── RF.ipynb                 # Random Forest regression implementation
├── SAGE_n_GCN.ipynb         # GCN and GraphSAGE implementation using PyTorch Geometric
├── CHANCHARUNEE_CHIANG_HE_LU_Project Final Report.pdf  # Full project report
├── README.md                # This file

## 📚 References

1. Avogaro et al., "Dif4FF: Multimodal Diffusion + GNNs for Fashion Forecasting" (ECCV 2024)
2. Kawas et al., "Predicting Product Returns in E-Commerce with GNNs" (Springer, 2023)
