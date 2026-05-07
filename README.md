# β-Brass Order-Disorder Transformation

A simulation of the order-disorder phase transition in β-brass (Cu-Zn) using free-energy minimization from the lattice bond model.

## Background

β-brass has a BCC structure where Cu and Zn atoms randomly occupy lattice sites at high temperature (disordered, A2). Below the critical temperature T_c ≈ 730 K, the system orders into the β′ (B2/CsCl) structure, where Cu preferentially occupies one simple-cubic sublattice and Zn occupies the other.

The transformation is a **second-order (continuous) phase transition** — the order parameter vanishes continuously at T_c.

## Model

The free energy is derived from a lattice bond model with bond energies ε_AA, ε_BB, ε_AB and interaction parameter W = ε_AA + ε_BB − 2ε_AB (W > 0 favors AB bonds → ordering):

$$\Delta F = 4W[X_B X_A - \eta^2] + k_BT\left[(X_B+\eta)\ln(X_B+\eta) + (X_B-\eta)\ln(X_B-\eta) + (X_A+\eta)\ln(X_A+\eta) + (X_A-\eta)\ln(X_A-\eta)\right]$$

The **order parameter** η is defined as:

$$\eta = \frac{1}{2}(X_B^\alpha - X_B^\beta)$$

where X_B^α and X_B^β are the mole fractions of B on each sublattice. η = 0 is fully disordered; |η| = X_B is fully ordered.

The critical temperature follows from d²ΔF/dη²|_{η=0} = 0, giving **k_B T_c = W**.

## Output

![order_disorder.png](order_disorder.png)

The figure shows:
- **Free energy curves** ΔF(η) − ΔF(0) at four temperatures, illustrating the double-well → single-well transition at T_c
- **Equilibrium order parameter** η_eq(T), showing the continuous second-order transition
- **2D lattice snapshots** of the BCC sublattice at three temperatures (ordered, partially ordered, disordered)

## Usage

```bash
pip install numpy matplotlib scipy
python order_disorder.py
```

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| X_B | 0.5 | Zn mole fraction (stoichiometric) |
| W | 1.0 | Ordering interaction energy (sets T_c scale) |
| GRID | 60 | Lattice snapshot size (60 × 60) |

## Reference

Lecture notes: *65.4 Order-Disorder Transformations*, 3.21 Kinetic Processes in Materials (MIT).
