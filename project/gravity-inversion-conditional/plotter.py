import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple
from density_mapping import DENSITY_TABLE_15


RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DENSITY_LUT   = np.array(DENSITY_TABLE_15, dtype=np.float32)
CMAP          = "viridis"
VMIN, VMAX    = 0.0, 3.8   # g/cm³ range of DENSITY_TABLE_15
N_CATS        = 15          # categories 0–14 (0 = air)
CAT_CMAP      = plt.cm.get_cmap("tab20", N_CATS)
DPI           = 150


def load_results() -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[torch.Tensor], List[torch.Tensor]]:
    """Load true model, boreholes, observed gravity, and the latest conditional prior / posterior samples."""
    true_model = torch.load(os.path.join(RESULTS_DIR, "true_model.pt"), map_location="cpu")
    boreholes  = torch.load(os.path.join(RESULTS_DIR, "boreholes.pt"),  map_location="cpu")
    d_obs      = np.load(os.path.join(RESULTS_DIR, "d_obs.npy"))

    prior_paths    = sorted(glob.glob(os.path.join(RESULTS_DIR, "conditional_prior_sample_*.pt")))
    post_paths     = sorted(glob.glob(os.path.join(RESULTS_DIR, "conditional_posterior_sample_*.pt")))

    prior_samples     = [torch.load(p, map_location="cpu") for p in prior_paths]
    posterior_samples = [torch.load(p, map_location="cpu") for p in post_paths]

    print(f"True model shape         : {true_model.shape}")
    print(f"Boreholes shape          : {boreholes.shape}")
    print(f"d_obs shape              : {d_obs.shape}")
    print(f"Conditional prior samples: {len(prior_samples)}")
    print(f"Conditional post. samples: {len(posterior_samples)}")

    return true_model, boreholes, d_obs, prior_samples, posterior_samples


def squeeze(vol: torch.Tensor) -> np.ndarray:
    """(B..., X, Y, Z) tensor → plain (X, Y, Z) numpy array."""
    t = vol
    while t.dim() > 3:
        t = t.squeeze(0)
    return t.numpy()


def cat_to_density(cat_vol: np.ndarray) -> np.ndarray:
    # (X, Y, Z) int in [0, 14] → (X, Y, Z) float density in g/cm³
    return DENSITY_LUT[cat_vol.astype(int)]


def cat_to_density_flat(cat_vol: np.ndarray) -> np.ndarray:
    # (X, Y, Z) int in [0, 14] → flat float64 density in Fortran order for SimPEG
    return DENSITY_LUT.astype(np.float64)[cat_vol.astype(int)].flatten(order="F")


def _vol_to_slices(vol_np: np.ndarray, mode: str) -> Tuple[List[np.ndarray], object, float, float, str]:
    """Return (3 mid-axis slices, cmap, vmin, vmax, colorbar_label) for the given mode."""
    if mode == "categories":
        data  = vol_np.astype(int)
        cmap, vmin, vmax, label = CAT_CMAP, -0.5, N_CATS - 0.5, "Rock category"
    else:
        data  = cat_to_density(vol_np)
        cmap, vmin, vmax, label = CMAP, VMIN, VMAX, "Density (g/cm³)"

    mid    = data.shape[0] // 2
    slices = [
        data[:, :, mid],
        np.rot90(data[:, mid, :], k=3),
        np.rot90(data[mid, :, :], k=3),
    ]
    return slices, cmap, vmin, vmax, label


def _obs_grid(d_obs: np.ndarray) -> np.ndarray:
    n_side = int(np.sqrt(d_obs.size))
    return d_obs.reshape(n_side, n_side) if n_side * n_side == d_obs.size else d_obs.reshape(-1, 1)


def _show_3views(ax_row, vol_np: np.ndarray, title: str, mode: str = "density") -> None:
    slices, cmap, vmin, vmax, _ = _vol_to_slices(vol_np, mode)
    labels = ["XY (mid Z)", "XZ (mid Y)", "YZ (mid X)"]
    for ax, slc, lbl in zip(ax_row, slices, labels):
        ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        ax.set_title(f"{title}\n{lbl}", fontsize=8)
        ax.axis("off")


def plot_trajectory(
    decoded_steps: List[np.ndarray],
    t0: float = 0.001,
    tf: float = 1.0,
    sample_idx: int = 0,
    mode: str = "density",
) -> None:
    """Scrollable 3-slice view of an ODE trajectory.

    decoded_steps : list of (X, Y, Z) integer category arrays, one per time step.
    mode          : 'density' or 'categories'
    Use the slider, or left/right arrow keys, to move through the integration.
    """
    from matplotlib.widgets import Slider

    n_steps          = len(decoded_steps)
    times            = np.linspace(t0, tf, n_steps)
    _, cmap, vmin, vmax, cbar_label = _vol_to_slices(decoded_steps[0], mode)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    plt.subplots_adjust(bottom=0.22)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label=cbar_label, fraction=0.015, pad=0.02)
    if mode == "categories":
        cbar.set_ticks(np.arange(N_CATS))

    slice_imgs = []
    labels     = ["XY (mid Z)", "XZ (mid Y)", "YZ (mid X)"]
    dummy      = np.zeros((decoded_steps[0].shape[0], decoded_steps[0].shape[1]))
    for ax, lbl in zip(axes, labels):
        im = ax.imshow(dummy, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest")
        ax.set_title(lbl, fontsize=8)
        ax.axis("off")
        slice_imgs.append(im)

    title = fig.suptitle("", fontsize=11)

    def draw(step: int) -> None:
        slices, *_ = _vol_to_slices(decoded_steps[step], mode)
        for im, slc in zip(slice_imgs, slices):
            im.set_data(slc)
        title.set_text(
            f"Conditional posterior sample {sample_idx} — step {step}/{n_steps - 1}  (t = {times[step]:.4f})"
        )
        fig.canvas.draw_idle()

    draw(0)

    ax_slider = plt.axes([0.15, 0.08, 0.70, 0.04])
    slider    = Slider(ax_slider, "Step", 0, n_steps - 1, valinit=0, valstep=1)
    slider.on_changed(lambda val: draw(int(val)))

    def on_key(event) -> None:
        step = int(slider.val)
        if event.key == "right" and step < n_steps - 1:
            slider.set_val(step + 1)
        elif event.key == "left" and step > 0:
            slider.set_val(step - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def plot_geology_comparison(
    true_model: torch.Tensor,
    boreholes: torch.Tensor,
    prior_samples: List[torch.Tensor],
    posterior_samples: List[torch.Tensor],
    mode: str = "density",
) -> None:
    """XY / XZ / YZ slices for the true model, boreholes, and all samples.

    mode : 'density' or 'categories'
    """
    rows  = [("True geology", true_model), ("Boreholes (observed)", boreholes)]
    rows += [(f"Cond. prior sample {i}",     s) for i, s in enumerate(prior_samples)]
    rows += [(f"Cond. posterior sample {i}", s) for i, s in enumerate(posterior_samples)]

    _, cmap, vmin, vmax, cbar_label = _vol_to_slices(squeeze(true_model), mode)

    n_rows    = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ax_row, (title, vol) in zip(axes, rows):
        _show_3views(ax_row, squeeze(vol), title, mode=mode)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label=cbar_label, fraction=0.015, pad=0.02)
    if mode == "categories":
        cbar.set_ticks(np.arange(N_CATS))

    plt.suptitle("Gravity inversion — conditional posterior flow sampling", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "geology_comparison.png"), dpi=DPI, bbox_inches="tight")
    plt.show()


def plot_gravity_residuals(
    d_obs: np.ndarray,
    true_model: torch.Tensor,
    prior_samples: List[torch.Tensor],
    posterior_samples: List[torch.Tensor],
    gravity_fwd,
) -> None:
    """Gravity residuals (d_pred − d_obs) for the true model and all samples."""
    residual_rows: List[Tuple[str, np.ndarray]] = []

    print("Computing gravity for true model ...")
    d_true = gravity_fwd.forward(cat_to_density_flat(squeeze(true_model)))
    residual_rows.append(("True model", d_true - d_obs))

    for i, sample in enumerate(prior_samples):
        print(f"Computing gravity for conditional prior sample {i} ...")
        d_p = gravity_fwd.forward(cat_to_density_flat(squeeze(sample)))
        residual_rows.append((f"Cond. prior {i}", d_p - d_obs))

    for i, sample in enumerate(posterior_samples):
        print(f"Computing gravity for conditional posterior sample {i} ...")
        d_post = gravity_fwd.forward(cat_to_density_flat(squeeze(sample)))
        residual_rows.append((f"Cond. posterior {i}", d_post - d_obs))

    n_side    = int(np.sqrt(d_obs.size))
    n_cols    = 1 + len(residual_rows)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    res_vals = np.concatenate([r for _, r in residual_rows])
    vmax_res = np.percentile(np.abs(res_vals), 98)

    im0 = axes[0].imshow(_obs_grid(d_obs), cmap="RdBu_r", origin="lower")
    plt.colorbar(im0, ax=axes[0], label="gz (mGal)")
    axes[0].set_title("Observed d_obs", fontsize=10)
    axes[0].set_xlabel("East")
    axes[0].set_ylabel("North")

    for ax, (label, residual) in zip(axes[1:], residual_rows):
        im  = ax.imshow(residual.reshape(n_side, n_side), cmap="RdBu_r", origin="lower",
                        vmin=-vmax_res, vmax=vmax_res)
        rms = np.sqrt(np.mean(residual ** 2))
        plt.colorbar(im, ax=ax, label="Δgz (mGal)")
        ax.set_title(f"Residual: {label}\n(RMS {rms:.4f} mGal)", fontsize=9)
        ax.set_xlabel("East")

    plt.suptitle("Gravity residuals  (d_pred − d_obs)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "gravity_residuals.png"), dpi=DPI, bbox_inches="tight")
    plt.show()


def save_trajectory_gif(
    decoded_steps: List[np.ndarray],
    out_path: str,
    t0: float = 0.001,
    tf: float = 1.0,
    sample_idx: int = 0,
    fps: int = 10,
    mode: str = "density",
) -> None:
    """Save an ODE trajectory as an animated GIF (3 slices per frame).

    mode : 'density' or 'categories'
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    n_steps          = len(decoded_steps)
    times            = np.linspace(t0, tf, n_steps)
    _, cmap, vmin, vmax, cbar_label = _vol_to_slices(decoded_steps[0], mode)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    plt.subplots_adjust(right=0.88)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label=cbar_label, fraction=0.015, pad=0.02)
    if mode == "categories":
        cbar.set_ticks(np.arange(N_CATS))

    labels = ["XY (mid Z)", "XZ (mid Y)", "YZ (mid X)"]
    dummy  = np.zeros((decoded_steps[0].shape[0], decoded_steps[0].shape[1]))
    imgs   = [ax.imshow(dummy, cmap=cmap, vmin=vmin, vmax=vmax,
                        origin="lower", interpolation="nearest")
              for ax in axes]
    for ax, lbl in zip(axes, labels):
        ax.set_title(lbl, fontsize=8)
        ax.axis("off")
    title = fig.suptitle("", fontsize=11)

    def update(step: int):
        slices, *_ = _vol_to_slices(decoded_steps[step], mode)
        for im, slc in zip(imgs, slices):
            im.set_data(slc)
        title.set_text(
            f"Conditional posterior sample {sample_idx} — step {step}/{n_steps - 1}  (t = {times[step]:.4f})"
        )
        return imgs

    anim = FuncAnimation(fig, update, frames=n_steps, interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":

    MODE          = "categories"   # "density" | "categories"
    PLOT_GRAVITY  = True
    GIF_FPS       = 10
    
    from gravity_forward import GravityForward

    true_model, boreholes, d_obs, prior_samples, posterior_samples = load_results()

    plot_geology_comparison(true_model, boreholes, prior_samples, posterior_samples, mode=MODE)

    traj_paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "trajectory_*.pt")))
    for traj_path in traj_paths:
        idx           = int(os.path.splitext(os.path.basename(traj_path))[0].split("_")[-1])
        decoded_steps = [t.numpy() for t in torch.load(traj_path, map_location="cpu")]
        out_path      = os.path.join(RESULTS_DIR, f"trajectory_{idx}_{MODE}.gif")
        save_trajectory_gif(decoded_steps, out_path, sample_idx=idx, fps=GIF_FPS, mode=MODE)

    if PLOT_GRAVITY:
        print("\nBuilding gravity forward operator ...")
        gravity_fwd = GravityForward(shape=(64, 64, 64),
                                     bounds=((-1920, 1920), (-1920, 1920), (-1920, 1920)),
                                     n_receivers_per_side=32,
                                     receiver_height=30.0)
        plot_gravity_residuals(d_obs, true_model, prior_samples, posterior_samples, gravity_fwd)

    print(f"\nFigures saved to {RESULTS_DIR}")
