# Posterior Flow Sampling for Gravity Inversion

Samples from the posterior distribution over subsurface geology conditioned on
observed surface gravity measurements, without retraining the generative model.

At each ODE step the prior velocity from a pretrained flow-matching model is
steered toward lower gravity misfit:

```
dm/dt = v_prior(m, t)  -  mu(t) * grad_m || G(rho(m)) - d_obs ||²
```

where `G` is the SimPEG gravity forward operator, `rho(m)` converts the
continuous embedding field to physical density, and `mu(t)` is a time-dependent
guidance weight.

---

## Module overview

```
gravity-inversion-unconditional/
├── main.py                   entry point — configure and run the full pipeline
├── posterior_flow_solver.py  guided ODE integrator (core algorithm)
├── gravity_forward.py        SimPEG gravity forward model and adjoint
├── density_mapping.py        embedding → density, differentiable
├── plotter.py                all visualisation
└── results/                  saved outputs (models, gravity data, figures)
```

### `main.py`

Orchestrates the pipeline end-to-end:

1. Loads a pretrained `FlowModel` from a Lightning checkpoint (downloads weights
   automatically if missing).
2. Builds the SimPEG gravity simulation once via `GravityForward`.
3. Generates a synthetic true geology by running one unconditional ODE solve,
   then computes synthetic observed gravity with instrument noise.
4. Runs unconditional samples (prior baseline) with the same seeds as the
   posterior runs for a fair comparison.
5. Runs posterior samples via `PosteriorFlowSolver`.
6. Saves all results to `results/` and, when `save_trajectory=True`, opens an
   interactive scrollable plot of the ODE trajectory.

`FlowModel` is a thin `nn.Module` wrapper around `Unet3D` plus an `Embedding`
layer. It exposes `.decode()`, which converts a continuous embedding volume
`(B, E, X, Y, Z)` to integer category labels `(B, X, Y, Z)` via cosine
similarity against the category centroids.

Configuration lives in `get_config()` as a plain dict — edit it directly to
change any hyperparameters.

---

### `posterior_flow_solver.py`

Contains the two classes that implement the guided ODE:

**`GuidanceSchedule`** — wraps a scalar function `mu(t)` that controls how
strongly the likelihood gradient influences the velocity at each time step.
Built-in schedules: `constant`, `linear_ramp`, `cosine_ramp`, `bell`. A
callable can also be passed directly.

**`PosteriorFlowSolver`** — integrates the posterior ODE from `t0` to `tf`
using either Euler or RK4. At each step:

1. The prior velocity `v_prior(m_t, t)` is queried from the frozen velocity
   network (no gradient tape needed here).
2. The gravity misfit gradient is computed via a chain-rule through three
   stages: soft-decode `m_t → rho` (PyTorch autograd), forward model
   `rho → d_pred` (SimPEG, CPU, numpy), adjoint `J^T (d_pred - d_obs)`
   (SimPEG), back into embedding space (autograd).
3. The two terms are combined: `v_posterior = v_prior - mu(t) * grad_misfit`.

Optional controls: gradient norm clipping (`grad_clip`), normalization of the
likelihood gradient relative to the prior velocity norm (`normalize_grad`), and
a static air-voxel mask to suppress guidance in known-empty regions.

When `save_trajectory=True`, every intermediate state is kept on CPU and
returned as a `(n_steps+1, B, E, X, Y, Z)` tensor.

---

### `gravity_forward.py`

`GravityForward` wraps a SimPEG `Simulation3DIntegral` on a regular
`TensorMesh`. The simulation is built once at construction and the full
sensitivity matrix is stored in RAM (`store_sensitivities="ram"`), so repeated
forward and adjoint evaluations are fast (matrix-vector products only).

Key methods:
- `forward(density)` — evaluates `G rho`, returning `(n_receivers,)` in mGal.
- `jtvec(density, v)` — evaluates `J^T v`, the adjoint needed for the gradient.
- `generate_synthetic_data(true_density, accuracy, confidence, seed)` — runs
  the forward model and adds Gaussian instrument noise scaled from a quoted
  instrument accuracy at a given confidence level (e.g. 0.1 mGal at 95%
  confidence → σ ≈ 0.051 mGal).

The default domain is a 64³ grid spanning ±1920 m in each axis (60 m cell
size), with 32×32 receivers on the surface at 30 m elevation.

---

### `density_mapping.py`

Provides two representations of the embedding-to-density mapping:

**`DENSITY_TABLE_15`** — a fixed 15-entry lookup table (g/cm³) covering air,
sediment layers, dikes, intrusions, and ore bodies. Indices are the integer
category labels produced by `FlowModel.decode()`.

**`DifferentiableDensityMapper`** — a soft, differentiable version used during
the guided ODE. Instead of a hard argmax it computes cosine similarities
between each voxel's embedding vector and the category centroids, applies a
temperature softmax, then takes the density as a probability-weighted sum of
`DENSITY_TABLE_15`. This keeps the computation graph intact so autograd can
propagate the gravity gradient back into the embedding field.

High temperature (default 10) concentrates probability on the nearest category
(near-hard assignment); lower temperature spreads mass across categories
(softer, more diffuse gradients).

---

### `plotter.py`

All figures are written to `results/`. Three functions are exposed:

- **`plot_geology_comparison`** — side-by-side XY / XZ / YZ density slices for
  the true model, unconditional samples, and posterior samples.
- **`plot_gravity_residuals`** — observed gravity alongside predicted-minus-
  observed residual maps for every model, with shared colour scale and per-panel
  RMS.
- **`plot_trajectory`** — interactive figure for inspecting an ODE trajectory.
  A slider (and left/right arrow keys) steps through all `n_steps+1` decoded
  density snapshots so you can watch the geology evolve from Gaussian noise to
  a structured geological model.

---

## Results directory

```
results/
├── true_model.pt               (X, Y, Z) integer category tensor
├── d_obs.npy                   (n_receivers,) observed gravity in mGal
├── posterior_sample_<i>.pt     final decoded posterior geology
├── unconditional_sample_<i>.pt final decoded prior geology
├── geology_comparison.png
└── gravity_residuals.png
```
