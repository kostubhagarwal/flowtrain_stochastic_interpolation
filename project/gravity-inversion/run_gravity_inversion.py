"""
Posterior flow sampling for gravity inversion.

Uses a pre-trained unconditional flow-matching model as the prior and guides
the ODE integration with a gravity data-fidelity gradient so that generated
geological models are consistent with observed gravity measurements.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Allow imports from the sibling unconditional project (model loading, utils)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UNCOND_DIR = os.path.join(_SCRIPT_DIR, "..", "geodata-3d-unconditional")
sys.path.insert(0, _UNCOND_DIR)

from model_train_inference import Geo3DStochInterp  # noqa: E402
from utils import download_if_missing, plot_static_views, plot_cat_view  # noqa: E402

# Local modules
from density_mapping import DifferentiableDensityMapper, DENSITY_TABLE_15  # noqa: E402
from gravity_forward import GravityForward  # noqa: E402
from posterior_flow_solver import PosteriorFlowSolver, GuidanceSchedule  # noqa: E402


# ======================================================================
# Configuration
# ======================================================================

def get_config() -> dict:
    return {
        "model": {
            "checkpoint_path": None,  # filled by CLI or default demo weights
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "domain": {
            "shape": (64, 64, 64),
            "bounds": ((-1920, 1920), (-1920, 1920), (-1920, 1920)),
        },
        "gravity": {
            "n_receivers_per_side": 16,
            "receiver_height": 30.0,
            "noise_percent": 2.0,
        },
        "inversion": {
            "n_samples": 4,
            "n_steps": 50,
            "t0": 0.001,
            "tf": 1.0,
            "mu_0": 1.0,
            "schedule": "linear_ramp",
            "method": "euler",
            "temperature": 10.0,
            "grad_clip": None,
            "normalize_grad": False,
            "seed": 42,
        },
        "output": {
            "save_dir": os.path.join(_SCRIPT_DIR, "results"),
            "save_trajectory": False,
            "save_images": True,
        },
    }


# ======================================================================
# Model loading
# ======================================================================

def load_pretrained_model(config: dict) -> Geo3DStochInterp:
    """Load the unconditional flow model from checkpoint."""
    device = config["model"]["device"]
    ckpt = config["model"]["checkpoint_path"]

    if ckpt is None:
        # Fall back to the demo weights shipped with the unconditional project
        ckpt = os.path.join(
            _UNCOND_DIR, "demo_model", "unconditional-weights.ckpt"
        )
        url = (
            "https://github.com/chipnbits/flowtrain_stochastic_interpolation"
            "/releases/download/v1.0.0/unconditional-weights.ckpt"
        )
        download_if_missing(ckpt, url)

    print(f"Loading model from: {ckpt}")
    model = Geo3DStochInterp.load_from_checkpoint(ckpt, map_location=device)
    model.to(device)
    model.eval()
    return model


# ======================================================================
# Synthetic problem generation
# ======================================================================

def create_synthetic_problem(
    model: Geo3DStochInterp,
    gravity_fwd: GravityForward,
    density_mapper: DifferentiableDensityMapper,
    config: dict,
) -> tuple:
    """Generate a synthetic gravity inverse problem.

    1. Draw one unconditional sample from the flow model.
    2. Decode to categorical indices → density.
    3. Compute gravity response + noise → d_obs.

    Returns
    -------
    true_categorical : torch.Tensor
        ``(X, Y, Z)`` categorical indices of the true model.
    true_density_np : np.ndarray
        ``(n_cells,)`` flattened density (Fortran order for SimPEG).
    d_obs : np.ndarray
        ``(n_data,)`` synthetic observed gravity data.
    """
    device = config["model"]["device"]
    shape = config["domain"]["shape"]

    print("Generating true model via unconditional sampling ...")
    from flowtrain.solvers import ODEFlowSolver

    solver = ODEFlowSolver(model=model.net, rtol=1e-6)
    X0 = torch.randn(1, model.embedding_dim, *shape, device=device)
    solution = solver.solve(X0, t0=0.001, tf=1.0, n_steps=32)
    X_final = solution[-1]  # (1, E, X, Y, Z)

    # Decode to categorical
    true_categorical = model.decode(X_final)[0].cpu()  # (X, Y, Z)

    # Map to density via the hard decode (argmax) path for a clean true model
    emb_weights = model.embedding.weight  # (K, E)
    n_cats = emb_weights.shape[0]
    density_lut = torch.tensor(DENSITY_TABLE_15[:n_cats], dtype=torch.float32)
    true_density_vol = density_lut[true_categorical.long()]  # (X, Y, Z)
    true_density_np = true_density_vol.numpy().flatten(order="F")

    # Generate observed gravity data with noise
    d_obs = gravity_fwd.generate_synthetic_data(
        true_density_np,
        noise_percent=config["gravity"]["noise_percent"],
        seed=config["inversion"]["seed"],
    )

    print(
        f"  True model categories: {int(true_categorical.min())} – {int(true_categorical.max())}"
    )
    print(f"  Density range: {true_density_np.min():.2f} – {true_density_np.max():.2f} g/cm³")
    print(f"  Gravity data: {d_obs.shape[0]} observations")
    print(f"  Gravity range: {d_obs.min():.4f} – {d_obs.max():.4f}")

    return true_categorical, true_density_np, d_obs


# ======================================================================
# Posterior sampling
# ======================================================================

def run_posterior_sampling(
    model: Geo3DStochInterp,
    gravity_fwd: GravityForward,
    d_obs: np.ndarray,
    config: dict,
) -> list:
    """Generate posterior samples conditioned on observed gravity data.

    Returns
    -------
    list[torch.Tensor]
        Decoded categorical models ``(X, Y, Z)`` per sample.
    """
    device = config["model"]["device"]
    inv = config["inversion"]
    shape = config["domain"]["shape"]

    density_mapper = DifferentiableDensityMapper(
        embedding_weights=model.embedding.weight.clone(),
        density_values=DENSITY_TABLE_15,
        temperature=inv["temperature"],
    ).to(device)

    guidance = GuidanceSchedule(mu_0=inv["mu_0"], schedule=inv["schedule"])

    solver = PosteriorFlowSolver(
        net=model.net,
        density_mapper=density_mapper,
        gravity_forward=gravity_fwd,
        d_obs=d_obs,
        guidance_schedule=guidance,
        method=inv["method"],
        grad_clip=inv["grad_clip"],
        normalize_grad=inv["normalize_grad"],
    )

    generator = torch.Generator(device="cpu").manual_seed(inv["seed"])
    results = []

    for idx in range(inv["n_samples"]):
        print(f"\n--- Posterior sample {idx + 1}/{inv['n_samples']} ---")
        m_0 = torch.randn(
            1, model.embedding_dim, *shape, generator=generator
        ).to(device)

        t_start = time.time()
        m_final = solver.solve(
            m_0,
            t0=inv["t0"],
            tf=inv["tf"],
            n_steps=inv["n_steps"],
            save_trajectory=config["output"]["save_trajectory"],
        )
        elapsed = time.time() - t_start

        # Take final state
        if config["output"]["save_trajectory"]:
            final_state = m_final[-1].to(device)
        else:
            final_state = m_final

        decoded = model.decode(final_state)[0].detach().cpu()  # (X, Y, Z)
        results.append(decoded)
        print(f"  Completed in {elapsed:.1f}s")

    return results


# ======================================================================
# Save results
# ======================================================================

def save_results(
    posterior_samples: list,
    true_categorical: torch.Tensor,
    d_obs: np.ndarray,
    save_dir: str,
    save_images: bool = True,
):
    """Persist posterior samples, the true model, and observed data."""
    os.makedirs(save_dir, exist_ok=True)

    # Save observed data
    np.save(os.path.join(save_dir, "d_obs.npy"), d_obs)

    # Save true model
    torch.save(true_categorical, os.path.join(save_dir, "true_model.pt"))
    if save_images:
        try:
            plot_cat_view(true_categorical, save_path=os.path.join(save_dir, "true_model_cat.png"))
        except Exception as e:
            warnings.warn(f"Failed to save true model image: {e}")

    # Save posterior samples
    for i, sample in enumerate(posterior_samples):
        torch.save(sample, os.path.join(save_dir, f"posterior_sample_{i}.pt"))
        if save_images:
            try:
                plot_cat_view(sample, save_path=os.path.join(save_dir, f"posterior_sample_{i}_cat.png"))
            except Exception as e:
                warnings.warn(f"Failed to save posterior sample {i} image: {e}")

    print(f"\nResults saved to {save_dir}")


# ======================================================================
# CLI
# ======================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Posterior flow sampling for gravity inversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to unconditional model checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device ('cpu' or 'cuda')")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Number of posterior samples")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="Number of ODE integration steps")
    parser.add_argument("--mu0", type=float, default=1.0,
                        help="Base guidance strength")
    parser.add_argument("--schedule", type=str, default="linear_ramp",
                        choices=["constant", "linear_ramp", "cosine_ramp", "bell"],
                        help="Guidance schedule")
    parser.add_argument("--temperature", type=float, default=10.0,
                        help="Softmax temperature for soft decode")
    parser.add_argument("--method", type=str, default="euler",
                        choices=["euler", "rk4"],
                        help="ODE integration method")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Gradient norm clipping value")
    parser.add_argument("--normalize-grad", action="store_true",
                        help="Normalise gradient to match prior velocity norm")
    parser.add_argument("--noise-percent", type=float, default=2.0,
                        help="Noise level for synthetic data (%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-trajectory", action="store_true",
                        help="Save full ODE trajectory (large!)")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip saving visualization images")
    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_arguments()
    config = get_config()

    # Override from CLI
    if args.checkpoint:
        config["model"]["checkpoint_path"] = args.checkpoint
    if args.device:
        config["model"]["device"] = args.device
    config["inversion"]["n_samples"] = args.n_samples
    config["inversion"]["n_steps"] = args.n_steps
    config["inversion"]["mu_0"] = args.mu0
    config["inversion"]["schedule"] = args.schedule
    config["inversion"]["temperature"] = args.temperature
    config["inversion"]["method"] = args.method
    config["inversion"]["grad_clip"] = args.grad_clip
    config["inversion"]["normalize_grad"] = args.normalize_grad
    config["inversion"]["seed"] = args.seed
    config["gravity"]["noise_percent"] = args.noise_percent
    config["output"]["save_trajectory"] = args.save_trajectory
    config["output"]["save_images"] = not args.no_images

    # 1. Load model
    model = load_pretrained_model(config)

    # 2. Set up gravity forward
    print("Building SimPEG gravity simulation ...")
    gravity_fwd = GravityForward(
        shape=config["domain"]["shape"],
        bounds=config["domain"]["bounds"],
        n_receivers_per_side=config["gravity"]["n_receivers_per_side"],
        receiver_height=config["gravity"]["receiver_height"],
    )
    print(f"  Mesh cells: {gravity_fwd.mesh.nC}")
    print(f"  Receivers : {gravity_fwd.survey.nD}")

    # 3. Create synthetic problem
    device = config["model"]["device"]
    density_mapper = DifferentiableDensityMapper(
        embedding_weights=model.embedding.weight.clone(),
        density_values=DENSITY_TABLE_15,
        temperature=config["inversion"]["temperature"],
    ).to(device)

    true_cat, true_density, d_obs = create_synthetic_problem(
        model, gravity_fwd, density_mapper, config
    )

    # 4. Run posterior sampling
    posterior_samples = run_posterior_sampling(model, gravity_fwd, d_obs, config)

    # 5. Save everything
    save_results(
        posterior_samples,
        true_cat,
        d_obs,
        config["output"]["save_dir"],
        save_images=config["output"]["save_images"],
    )


if __name__ == "__main__":
    main()
