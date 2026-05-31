# PINN Bachelor Research

Exploring Physics-Informed Neural Networks (PINNs) for fluid flow reconstruction, with a focus on recovering velocity fields in regions where experimental measurements are unavailable or unreliable.

---

## Overview

This project investigates the use of PINNs — neural networks trained to satisfy physical governing equations alongside data — for reconstructing flow fields in measurement blind zones. The work progresses from simple analytical benchmarks to a full experimental application on real PIV data.

The core idea: given sparse or missing measurements in a region of interest, a PINN can fill the gap by enforcing the Navier-Stokes equations as a soft constraint in the loss function, using available data outside the gap as boundary supervision.

---

## Repository Structure

```
PINN_Bachelor_Research/
├── Code/
│   ├── Poiseuille.py               # Poiseuille flow PINN
│   ├── Poiseullie_veri.py          # Poiseuille verification
│   ├── BL_PINN_FS.py               # boundary layer PINN (flat surface)
│   ├── NS_PINN.py                  # standalone Navier-Stokes PINN
│   ├── cylinder_pinn.py            # cylinder flow PINN (synthetic data)
│   ├── CFD_DATA_PINN.py            # PINN trained on CFD cylinder data
│   ├── gaptrial.py                 # gap thickness study
│   └── RealCyltFlow/
│       ├── RealCylFlowPINN.py      # main training script (real PIV data)
│       ├── posrealcylflow.py       # post-processing: Cf, BLP, plots
│       └── Model Saves/            # saved models and data per gap ratio
├── Data/
│   ├── RealCylFlow/
│   │   ├── Cylinder_D_125mm_Uinf_5ms.tp   # raw Tecplot PIV data
│   │   └── flow_data.npz
│   ├── Cylinder flow data/
│   │   ├── CYLINDER_ALL.mat               # CFD cylinder dataset
│   │   └── cylinder_flow.npz
│   └── u.npy, v.npy, x.npy, y.npy        # flat surface BL data
├── Results/                        # plots organised by case
└── README.md
```

---

## Cases

### 1. Poiseuille Flow
Analytical 1D benchmark used to validate the PINN framework. A gap is introduced in the channel and the PINN reconstructs the parabolic velocity profile using only boundary conditions and NS physics. Tested with varying numbers of data points, boundary condition configurations, noisy data, and loss weightings.

### 2. Flat Surface Boundary Layer
PINN applied to boundary layer flow over a flat surface. Used to develop and test the near-wall reconstruction approach before moving to curved geometry.

### 3. Cylinder Flow — Synthetic / CFD Data
2D PINN on numerically generated flow past a circular cylinder. Gap masking introduced to test reconstruction capability in a controlled setting where the ground truth is fully known.

### 4. Cylinder Flow — Real PIV Data
Full experimental application. PIV measurements of flow past a circular cylinder (D = 125 mm, U∞ = 5 m/s, Re ≈ 41,667) have a near-wall blind zone due to laser reflections. The PINN reconstructs the velocity field in this gap, enabling computation of:
- Skin friction coefficient C_f via automatic differentiation at the wall
- Boundary layer profiles along wall-normal rays
- Gap thickness study (g/D = 0.005 to 0.08) to find the optimal gap size

---

## General Approach

All cases follow the same PINN framework:

**Loss function:**
$$\mathcal{L} = w_{NS}\mathcal{L}_{NS} + w_{data}\mathcal{L}_{data} + w_{BC}\mathcal{L}_{BC}$$

- $\mathcal{L}_{NS}$ — residual of the governing PDE (Navier-Stokes or simplified form)
- $\mathcal{L}_{data}$ — MSE against available measurements outside the gap
- $\mathcal{L}_{BC}$ — boundary condition enforcement

**Architecture:** fully connected network, Tanh activation, normalised inputs, outputs vary by case (u only, or u/v/p).

**Framework:** PyTorch, automatic differentiation for PDE residuals, Adam optimiser.

---

## Dependencies

```
python >= 3.8
torch >= 1.12
numpy
matplotlib
scipy
```

---

*Results and validation ongoing.*
