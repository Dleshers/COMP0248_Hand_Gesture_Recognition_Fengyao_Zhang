# COMP0248 Coursework 1: Attentional RGB-D Hand Gesture Recognition

**Author:** Fengyao Zhang (Student ID: 25081046)  
**Programme:** MSc Robotics and Artificial Intelligence, University College London (UCL)  

---

## Performance Highlights
The finalized **Attentional RGB-D Two-Stream** architecture achieved exceptional performance across all three heterogeneous tasks, completely shattering the capacity bottleneck of lightweight custom networks:
* **Top-1 Classification Accuracy:** **88.00%** (Unseen Test Set) / **94.20%** (Validation Set)
* **Mean Bounding Box IoU:** **0.9345** (Near-perfect spatial encapsulation)
* **Segmentation Mask mIoU:** **0.9088**

## Project Overview
This repository contains the implementation of a multi-task learning framework trained entirely from scratch for simultaneous hand gesture **classification**, pixel-wise **segmentation**, and bounding box **detection**. 

To overcome the inherent noise found in raw depth sensor data and the gradient domination issue in multi-task learning, this project introduces a novel **Attentional RGB-D Two-Stream Late Fusion** architecture. By integrating a Squeeze-and-Excitation (SE) channel attention mechanism and a highly rigorous data preprocessing pipeline, the network autonomously mutes modality-specific noise, effectively transforming the depth stream into a robust structural enhancer.

## Key Innovations
* **Dynamic ROI Cropping & Spatial Synchronization:** Extracts the bounding box from the ground-truth mask to dynamically crop the Region of Interest (ROI). All spatial transformations (rotation, horizontal flip) are perfectly synchronized across RGB, Depth, and Mask tensors to eliminate background clutter and preserve absolute geometric consistency.
* **Absolute Physical Depth Clipping:** Restricts raw depth data to a human-interactive spatial volume (200mm - 1500mm), systematically eradicating extreme background outliers and dead pixels.
* **SE-Attentional Late Fusion:** Dynamically learns inter-channel dependencies via a Squeeze-and-Excitation block to autonomously mute corrupted depth features before the final multi-task heads.
* **Advanced Optimization & Loss Balancing:** Mitigates severe gradient domination by explicitly prioritizing classification ($\lambda_{cls}=3.0, \lambda_{seg}=0.5, \lambda_{det}=0.5$). Coupled with the **AdamW** optimizer and a **Cosine Annealing** learning rate scheduler, the custom lightweight network achieves stable, oscillation-free convergence.

---

## Environment Setup

It is recommended to use a virtual environment (e.g., Conda) to manage dependencies.

```bash
# Create and activate a new conda environment
conda create -n comp0248_cw1 python=3.9
conda activate comp0248_cw1

# Install PyTorch and Torchvision (adjust CUDA version as per your hardware)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional required packages from requirements.txt
pip install -r requirements.txt


