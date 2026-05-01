# UTD AI Safety Lab Work

## Overview
AI safety research work associated with UTD, covering uncertainty estimation, out-of-distribution (OOD) detection, and trajectory prediction using deep learning.

## Structure
```
UTD-AI-SAFETY-LAB-WORK/
├── TaskOne/ # Trajectory prediction (nuScenes / CoverNet)
├── TaskTwo/ # Uncertainty estimation (MNIST / EMNIST, EDL)
├── DatasetCreation/ # Later CARLA dataset generation work
│ ├── docker/ # CARLA + Jupyter environment
│ ├── builderfiles/ # Supporting scripts
│ ├── *.py # Data + training pipeline
│ └── *.ipynb # Notebook workflows
```

## Dataset Pipeline (Reference)
1. Rasterization – convert collected CARLA metadata into raster inputs  
2. Dataset Construction – build temporal datasets with trajectories after rasterization  
3. Lattice Generation – create fixed trajectory sets for CoverNet  
4. Training – train and evaluate on the generated dataset

## Running the Code

Some components in this repository require specific environments and external datasets.

- TaskOne / TaskTwo can be run as standalone Python scripts after installing dependencies.
- DatasetCreation requires a Docker environment along with CARLA and additional data setup.

This repository is intended primarily for reference and experimentation.

## Additional Notes
- Dataset creation work was developed later and is included as reference  
- Some components require Docker and external datasets like CARLA or nuScenes  

## Tech Stack
Python, PyTorch, NumPy, CARLA, nuScenes, Docker