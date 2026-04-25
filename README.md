# AE & VAE on Medical MNIST



## Project Overview
This project implements:
- Autoencoder (AE)
- Variational Autoencoder (VAE)


on the Medical MNIST dataset (6 classes).

## Dataset
Medical MNIST:
- AbdomenCT
- BreastMRI
- CXR
- ChestCT
- Hand
- HeadCT

## Experiments
- Image reconstruction (AE vs VAE)
- Latent space visualization (2D)
- Sample generation (VAE)
- Denoising comparison (AE, VAE, DAE)
- MSE & MAE evaluation

## Results
AE achieved better reconstruction:
- AE MSE: 0.00235
- AE MAE: 0.02613
- VAE MSE: 0.02044
- VAE MAE: 0.08646

VAE enabled generation of new images.

Denoising Autoencoder achieved best noise removal.

## How to Run
1. Open notebook in Google Colab
2. Mount Google Drive
3. Run all cells

## Files
- Notebook: Implementation
- Report: Technical analysis
