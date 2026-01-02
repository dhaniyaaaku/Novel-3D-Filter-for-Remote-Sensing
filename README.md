

# A Novel Spatial–Spectral Fusion Filter for Hyperspectral Image Classification

This repository presents a novel spatial–spectral filtering framework designed for hyperspectral image (HSI) analysis. The proposed filter introduces a **simultaneous, multiplicative fusion** of spatial gradients, spectral variation, and spatial–spectral correlation to generate highly discriminative feature representations.

Unlike traditional hyperspectral feature extraction methods that process spatial and spectral information sequentially, this work explicitly models their interaction within a unified mathematical formulation.



## Key Contributions

- First-of-its-kind **simultaneous spatial–spectral fusion filter**
- Explicit modeling of:
  - Spatial edge strength
  - Spectral heterogeneity
  - Local spectral–spatial consistency
- Lightweight and interpretable alternative to deep feature extractors
- Strong performance under **limited labeled data**
- Computationally efficient compared to EMP and 3D Gabor filters

---

## Experimental Setup

**Datasets**
- Indian Pines
- Pavia University
- Salinas Scene

**Baseline Filters**
- 3D Gabor Wavelet Filters
- 3D Extended Morphological Profiles (EMP)

**Classifiers**
- 3D Convolutional Neural Network (3D CNN)
- Residual Neural Network (ResNet)

---

## Results Highlights

- Indian Pines: **99.93% accuracy (ResNet)**
- PaviaU: **99.86% accuracy (3D CNN)**
- Consistently lowest computational time among compared filters
- Superior boundary preservation and reduced over-smoothing



---

## ⚙️ Setup Instructions

```bash
pip install -r requirements.txt
