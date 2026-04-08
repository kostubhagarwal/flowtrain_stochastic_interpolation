"""
Plot gravity inversion results:
  - Observed gravity map
  - True geology vs posterior samples (XY / XZ / YZ slices)
  - Gravity residuals: d_pred(true) - d_obs  and  d_pred(posterior) - d_obs
"""

import os
import sys
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from density_mapping import DENSITY_TABLE_15
from gravity_forward import GravityForward

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
density_lut = np.array(DENSITY_TABLE_15, dtype=np.float32)
CMAP = "viridis"
VMIN, VMAX = 0.0, 3.8   # g/cm³ range of DENSITY_TABLE_15


def cat_to_density(cat_vol_np):
    """(X,Y,Z) int array → (X,Y,Z) float density in g/cm³."""
    return density_lut[cat_vol_np.astype(int)]


# ── load ──────────────────────────────────────────────────────────────────────
true_model = torch.load(os.path.join(RESULTS_DIR, "true_model.pt"), map_location="cpu")
d_obs      = np.load(os.path.join(RESULTS_DIR, "d_obs.npy"))

post_paths  = sorted(glob.glob(os.path.join(RESULTS_DIR, "posterior_sample_*.pt")))
uncond_paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "unconditional_sample_*.pt")))

posterior_samples    = [torch.load(post_paths[-1],  map_location="cpu")] if post_paths  else []
unconditional_samples = [torch.load(uncond_paths[-1], map_location="cpu")] if uncond_paths else []
n_samples = len(posterior_samples)

print(f"True model shape      : {true_model.shape}")
print(f"d_obs shape           : {d_obs.shape}")
print(f"Posterior samples     : {n_samples}")
print(f"Unconditional samples : {len(unconditional_samples)}")


# ── 1. Gravity map ─────────────────────────────────────────────────────────────
n_side = int(np.sqrt(d_obs.size))
if n_side * n_side == d_obs.size:
    grav_grid = d_obs.reshape(n_side, n_side)
else:
    grav_grid = d_obs.reshape(-1, 1)   # fallback: 1-D strip

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(grav_grid, cmap="RdBu_r", origin="lower")
plt.colorbar(im, ax=ax, label="gz (mGal)")
ax.set_title("Observed gravity (synthetic)", fontsize=11)
ax.set_xlabel("East receiver")
ax.set_ylabel("North receiver")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gravity_map.png"), dpi=150)
plt.show()


# ── 2. Geology slices: true + posterior samples ────────────────────────────────
def _squeeze(vol):
    """Return a plain (X, Y, Z) numpy array regardless of leading batch dims."""
    t = vol
    while t.dim() > 3:
        t = t.squeeze(0)
    return t.numpy()


def show_3views(ax_row, vol_np, title):
    density_vol = cat_to_density(vol_np)
    mid = density_vol.shape[0] // 2
    slices = [density_vol[:, :, mid],
              np.rot90(density_vol[:, mid, :], k=3),
              np.rot90(density_vol[mid, :, :], k=3)]
    plane_labels = ["XY (mid Z)", "XZ (mid Y)", "YZ (mid X)"]
    for ax, slc, lbl in zip(ax_row, slices, plane_labels):
        ax.imshow(slc, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                  origin="lower", interpolation="nearest")
        ax.set_title(f"{title}\n{lbl}", fontsize=8)
        ax.axis("off")


rows = [("True geology", true_model)]
for i, s in enumerate(unconditional_samples):
    rows.append((f"Unconditional sample {i}", s))
for i, s in enumerate(posterior_samples):
    rows.append((f"Posterior sample {i}", s))

n_rows = len(rows)
fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]

for ax_row, (title, vol) in zip(axes, rows):
    show_3views(ax_row, _squeeze(vol), title)

# shared colorbar
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(vmin=VMIN, vmax=VMAX))
sm.set_array([])
fig.colorbar(sm, ax=axes, label="Density (g/cm³)", fraction=0.015, pad=0.02)

plt.suptitle("Gravity inversion — posterior flow sampling", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "geology_comparison.png"), dpi=150,
            bbox_inches="tight")
plt.show()

print(f"\nFigures saved to {RESULTS_DIR}")


# ── 3. Gravity residuals ───────────────────────────────────────────────────────
print("\nBuilding gravity forward operator (this may take a moment) ...")
gravity_fwd = GravityForward(
    shape=(64, 64, 64),
    bounds=((-1920, 1920), (-1920, 1920), (-1920, 1920)),
    n_receivers_per_side=16,
    receiver_height=30.0,
)

def cat_to_density_flat(cat_vol_np):
    """(X,Y,Z) int numpy array → flat density array in Fortran order for SimPEG."""
    return density_lut.astype(np.float64)[cat_vol_np.astype(int)].flatten(order="F")


print("Computing gravity for true model ...")
d_true = gravity_fwd.forward(cat_to_density_flat(_squeeze(true_model)))

residual_rows = [("True model", d_true - d_obs)]
for i, sample in enumerate(unconditional_samples):
    print(f"Computing gravity for unconditional sample {i} ...")
    d_u = gravity_fwd.forward(cat_to_density_flat(_squeeze(sample)))
    residual_rows.append((f"Unconditional {i}", d_u - d_obs))
for i, sample in enumerate(posterior_samples):
    print(f"Computing gravity for posterior sample {i} ...")
    d_post = gravity_fwd.forward(cat_to_density_flat(_squeeze(sample)))
    residual_rows.append((f"Posterior {i}", d_post - d_obs))

# reshape to grid
n_side = int(np.sqrt(d_obs.size))
d_obs_grid = d_obs.reshape(n_side, n_side)

n_cols = 1 + len(residual_rows)   # observed + one per model
fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

# shared colour scale for residuals
res_vals = np.concatenate([r for _, r in residual_rows])
vmax_res = np.percentile(np.abs(res_vals), 98)

im0 = axes[0].imshow(d_obs_grid, cmap="RdBu_r", origin="lower")
plt.colorbar(im0, ax=axes[0], label="gz (mGal)")
axes[0].set_title("Observed d_obs", fontsize=10)
axes[0].set_xlabel("East"); axes[0].set_ylabel("North")

for ax, (label, residual) in zip(axes[1:], residual_rows):
    res_grid = residual.reshape(n_side, n_side)
    im = ax.imshow(res_grid, cmap="RdBu_r", origin="lower",
                   vmin=-vmax_res, vmax=vmax_res)
    plt.colorbar(im, ax=ax, label="Δgz (mGal)")
    rms = np.sqrt(np.mean(residual ** 2))
    ax.set_title(f"Residual: {label}\n(RMS {rms:.4f} mGal)", fontsize=9)
    ax.set_xlabel("East")

plt.suptitle("Gravity residuals  (d_pred − d_obs)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gravity_residuals.png"), dpi=150,
            bbox_inches="tight")
plt.show()
print(f"Residual figure saved to {RESULTS_DIR}/gravity_residuals.png")
