# Scientific Computing for All — Visual PDE Lab

A GPU-accelerated 2D partial differential equation (PDE) solver suite with an interactive visualization interface built in Streamlit.  
The project demonstrates classical numerical methods for Poisson, Diffusion, and Wave equations on structured grids, accelerated using CuPy (CUDA) when available.

---

## Overview

This project combines a modular solver engine with an interactive web-based interface for exploring PDEs in real time.  
It supports both CPU (NumPy) and GPU (CuPy) computation, configurable boundary conditions, and flexible domain masking for geometric constraints.

---

## Key Features

- Unified solver for Poisson, Diffusion, and Wave equations
- CPU/GPU backend (NumPy or CuPy) with automatic fallback
- Dirichlet, Neumann, and Robin boundary conditions (per-side control)
- Implicit and explicit time-stepping schemes for Diffusion
- Leapfrog integration for the Wave equation
- Geometry masking (draw or import binary PNG)
- Streamlit interface for interactive configuration and visualization
- Config export/import and field export to PNG or NPZ

## Installation


## Clone repository
```bash
git clone https://github.com/Jedi3108/scientific-computing-for-all.git
cd scientific-computing-for-all
```

## Create environment
```bash
conda create -n scfa python=3.11 -y
conda activate scfa
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## (Optional) Install CuPy for GPU acceleration
```bash
pip install cupy-cuda12x
```
## Usage

```bash
streamlit run app/app_streamlit.py
```

## Example Workflows

- Poisson: Set boundary potentials and click Solve to compute the steady-state potential field.
- Diffusion: Choose explicit or implicit (Backward Euler) scheme and observe time evolution.
- Wave: Adjust CFL fraction and wave speed, then run the simulation to observe propagation.
- Masking: Apply geometric masks via rectangles, circles, or imported binary images to restrict the computational domain.

## Technical Notes

- Finite Difference Method (FDM) on structured 2D grids (5-point stencil)
- Conjugate Gradient (CG) method for Poisson and implicit Diffusion
- CFL condition enforcement for explicit schemes
- Streamlit used as a front-end visualization and control interface
- Designed for educational, research, and rapid prototyping purposes
