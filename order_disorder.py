"""
β-Brass Order-Disorder Transformation
Free-energy minimization using the formula from lecture notes (65.4):

  ΔF = 4W[X_B X_A - η²]
      + kT[(X_B+η)ln(X_B+η) + (X_B-η)ln(X_B-η)
           + (X_A+η)ln(X_A+η) + (X_A-η)ln(X_A-η)]

Units: k_B = 1, W = 1  →  T_c = W = 1
Order parameter η ∈ [-X_B, +X_B]; η=0 disordered, |η|=X_B fully ordered.
Real-scale: β-brass T_c ≈ 730 K (457 °C), used to label axes in Kelvin.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.optimize import minimize_scalar

# ── Parameters ──────────────────────────────────────────────────────────────
X_B = 0.5  # Zn mole fraction (B atoms)
X_A = 1 - X_B  # Cu mole fraction (A atoms)
W = 1.0  # ordering interaction; W > 0 → AB bonds energetically favored
T_c = W  # critical temperature (k_B T_c = W, with k_B = 1)

GRID = 60  # lattice grid size for snapshots
T_c_K = 730  # β-brass critical temperature in Kelvin (≈ 457 °C)


# ── Free energy (per site) ───────────────────────────────────────────────────
def delta_F(eta, T):
    eps = 1e-15
    xbp = np.clip(X_B + eta, eps, 1.0)
    xbm = np.clip(X_B - eta, eps, 1.0)
    xap = np.clip(X_A + eta, eps, 1.0)
    xam = np.clip(X_A - eta, eps, 1.0)
    energy = 4 * W * (X_B * X_A - eta**2)
    config = T * (
        xbp * np.log(xbp) + xbm * np.log(xbm) + xap * np.log(xap) + xam * np.log(xam)
    )
    return energy + config


def eta_equilibrium(T):
    """Equilibrium η at temperature T via free-energy minimization."""
    bound = min(X_B, X_A) - 1e-9
    res = minimize_scalar(delta_F, bounds=(0.0, bound), method="bounded", args=(T,))
    # Accept ordered state only if it genuinely beats the disordered state
    return float(res.x) if res.fun < delta_F(0.0, T) - 1e-10 else 0.0


# ── 2D BCC sublattice snapshot ───────────────────────────────────────────────
def make_lattice(eta, seed=42):
    """
    Checkerboard BCC sub-lattices:
      (i+j) even → α-sublattice, P(B) = X_B + η
      (i+j) odd  → β-sublattice, P(B) = X_B - η
    """
    rng = np.random.default_rng(seed)
    ii, jj = np.indices((GRID, GRID))
    p = np.where((ii + jj) % 2 == 0, X_B + eta, X_B - eta)
    return (rng.random((GRID, GRID)) < p).astype(np.int8)


# ── Compute equilibrium η over temperature range ─────────────────────────────
T_sweep = np.linspace(0.05 * T_c, 1.8 * T_c, 500)
eta_vals = np.array([eta_equilibrium(T) for T in T_sweep])

# Temperatures for the F(η) curves and lattice snapshots (in Kelvin, then normalised)
T_curves_K = [300, 600, 730, 900]
T_snaps_K = [219, 621, 1095]
T_curves = [T / T_c_K for T in T_curves_K]
T_snaps = [T / T_c_K for T in T_snaps_K]
colors = ["#1a237e", "#1565c0", "#f57f17", "#b71c1c"]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9))
fig.suptitle(
    r"β-Brass Order-Disorder Transformation  ($X_B = 0.5$, $W = 1$)",
    fontsize=14,
    fontweight="bold",
    y=0.97,
)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.38)
ax_F = fig.add_subplot(gs[0, :2])  # ΔF vs η
ax_eta = fig.add_subplot(gs[0, 2:])  # η_eq vs T
ax_s = [fig.add_subplot(gs[1, k]) for k in range(3)]
ax_leg = fig.add_subplot(gs[1, 3])

# ── Panel 1: ΔF(η) − ΔF(0) vs η ─────────────────────────────────────────────
eta_arr = np.linspace(-0.499, 0.499, 800)

for T, T_K, c in zip(T_curves, T_curves_K, colors):
    F_ref = delta_F(0.0, T)
    F = np.array([delta_F(e, T) - F_ref for e in eta_arr])
    lbl = rf"$T = {T_K}$ K" + (r"  $(= T_c)$" if T_K == 730 else "")
    ax_F.plot(eta_arr, F, color=c, lw=2, label=lbl)

ax_F.axhline(0, color="gray", lw=0.6, ls="--")
ax_F.axvline(0, color="gray", lw=0.6, ls="--")
ax_F.set_xlabel(r"Order parameter $\eta$", fontsize=11)
ax_F.set_ylabel(r"$\Delta F(\eta) - \Delta F(0)$  [units of $W$]", fontsize=10)
ax_F.set_title("Free Energy vs. Order Parameter", fontsize=12)
ax_F.legend(fontsize=9, loc="lower center", ncol=2)
ax_F.set_xlim(-0.55, 0.55)
ax_F.set_ylim(-0.15, 0.15)

# ── Panel 2: η_eq vs T ───────────────────────────────────────────────────────
# Ordered branch (T < T_c): solid curve where η_eq > 0
ordered = eta_vals > 1e-6
ax_eta.plot(
    eta_vals[ordered],
    T_sweep[ordered] * T_c_K,
    "k-",
    lw=2.5,
    label=r"$\eta_{eq}$ (ordered)",
)

# Disordered branch (T > T_c): η_eq = 0, shown as dashed vertical segment
T_dis = T_sweep[~ordered] * T_c_K
ax_eta.plot(
    np.zeros_like(T_dis),
    T_dis,
    "k--",
    lw=1.5,
    alpha=0.5,
    label=r"$\eta_{eq} = 0$ (disordered)",
)

ax_eta.axhline(T_c_K, color="red", lw=1.3, ls="--", label=rf"$T_c = {T_c_K}$ K")

snap_colors = [colors[0], colors[1], colors[3]]
for T_s, T_K, c in zip(T_snaps, T_snaps_K, snap_colors):
    e_s = eta_equilibrium(T_s)
    ax_eta.scatter(e_s, T_K, s=90, color=c, zorder=5, label=rf"${T_K}$ K")

ax_eta.set_xlabel(r"$\eta_\mathrm{eq}$", fontsize=11)
ax_eta.set_ylabel(r"$T$ (K)", fontsize=11)
ax_eta.set_title("Equilibrium Order Parameter vs. Temperature", fontsize=12)
ax_eta.legend(fontsize=9)
ax_eta.set_xlim(-0.01, 0.53)
ax_eta.set_ylim(0, 1.85 * T_c_K)
ax_eta.set_box_aspect(1)

# ── Panels 3-5: lattice snapshots ─────────────────────────────────────────────
# Gold = Cu (A atoms, sublattice B when ordered), steel blue = Zn (B atoms)
atom_cmap = ListedColormap(["#FFD700", "#4169E1"])
snap_titles = [
    rf"Ordered  ($T = {T_snaps_K[0]}$ K)",
    rf"Partial  ($T = {T_snaps_K[1]}$ K)",
    rf"Disordered  ($T = {T_snaps_K[2]}$ K)",
]

for ax, T_s, title, seed, c in zip(
    ax_s, T_snaps, snap_titles, [None, None, None], snap_colors
):
    eta_s = eta_equilibrium(T_s)
    lat = make_lattice(eta_s, seed=seed)
    ax.imshow(
        lat, cmap=atom_cmap, vmin=0, vmax=1, interpolation="nearest", aspect="equal"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    # Mark the dot on the η_eq plot
    for spine in ax.spines.values():
        spine.set_edgecolor(c)
        spine.set_linewidth(2.5)

# ── Legend panel ──────────────────────────────────────────────────────────────
ax_leg.axis("off")
ax_leg.legend(
    handles=[
        Patch(facecolor="#FFD700", edgecolor="gray", label="A atoms (Cu)"),
        Patch(facecolor="#4169E1", edgecolor="gray", label="B atoms (Zn)"),
    ],
    loc="center",
    fontsize=12,
    title="Atom types",
    title_fontsize=12,
    frameon=True,
    framealpha=0.9,
)
ax_leg.text(
    0.5,
    0.15,
    "α-sublattice: $P(B) = X_B + \\eta$\nβ-sublattice: $P(B) = X_B - \\eta$",
    ha="center",
    va="center",
    transform=ax_leg.transAxes,
    fontsize=9,
    color="dimgray",
)

plt.savefig("order_disorder.png", dpi=150, bbox_inches="tight")
plt.show()
