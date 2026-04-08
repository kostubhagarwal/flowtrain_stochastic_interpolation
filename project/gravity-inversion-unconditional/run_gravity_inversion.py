"""
Posterior flow sampling for gravity inversion.

Uses a pre-trained unconditional flow-matching model as the prior and guides
the ODE integration with a gravity data-fidelity gradient so that generated
geological models are consistent with observed gravity measurements.

This script is self-contained — it reconstructs the model directly from a
Lightning checkpoint without importing the training-only modules that depend
on geogen.
"""

import argparse
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flowtrain.models import Unet3D
from flowtrain.solvers import ODEFlowSolver

from density_mapping import DifferentiableDensityMapper, DENSITY_TABLE_15
from gravity_forward import GravityForward
from posterior_flow_solver import PosteriorFlowSolver, GuidanceSchedule


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# Lightweight model wrapper (avoids importing geogen)
# ======================================================================

class FlowModel(nn.Module):
    """Minimal wrapper holding the Unet3D velocity network and embedding layer.

    Provides the same ``.net``, ``.embedding``, ``.embedding_dim``,
    ``.decode()`` interface as ``Geo3DStochInterp`` but without any
    training / Lightning / geogen dependencies.
    """

    def __init__(self, net: Unet3D, embedding: nn.Embedding):
        super().__init__()
        self.net = net
        self.embedding = embedding
        self.embedding_dim = embedding.embedding_dim
        self.num_categories = embedding.num_embeddings

    # ---- decode (same logic as Geo3DStochInterp.decode) ----
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode embedding-space tensor to categorical indices.

        Parameters
        ----------
        x : torch.Tensor  ``(B, E, X, Y, Z)``

        Returns
        -------
        torch.Tensor  ``(B, X, Y, Z)``  integer category indices.
        """
        emb = self.embedding.weight  # (K, E)
        B, E, X, Y, Z = x.shape

        x_norm = F.normalize(x, dim=1)
        emb_norm = F.normalize(emb, dim=1)

        x_exp = x_norm.unsqueeze(1)  # (B, 1, E, X, Y, Z)
        emb_exp = emb_norm.view(1, self.num_categories, E, 1, 1, 1)

        logits = (x_exp * emb_exp).sum(dim=2)  # (B, K, X, Y, Z)
        return torch.argmax(logits, dim=1)  # (B, X, Y, Z)


def _download_if_missing(path: str, url: str) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading weights from {url} ...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")


def _init_simplex_embedding(n_cats: int, n_dims: int) -> torch.Tensor:
    """Build the same centred-simplex embedding used during training."""
    init = torch.zeros(n_cats, n_dims)
    init[:, :n_cats] = torch.eye(n_cats)
    centroid = torch.ones(n_cats) / n_cats
    centroid = torch.cat([centroid, torch.zeros(n_dims - n_cats)])
    init[:, :n_cats] -= centroid[:n_cats].unsqueeze(0)
    init = init / init.norm(dim=1, keepdim=True)
    return init


def load_pretrained_model(config: dict) -> FlowModel:
    """Reconstruct the inference model directly from a Lightning checkpoint."""
    device = config["model"]["device"]
    ckpt_path = config["model"]["checkpoint_path"]

    if ckpt_path is None:
        ckpt_path = os.path.join(
            _SCRIPT_DIR, "..", "geodata-3d-unconditional",
            "demo_model", "unconditional-weights.ckpt",
        )
        url = (
            "https://github.com/chipnbits/flowtrain_stochastic_interpolation"
            "/releases/download/v1.0.0/unconditional-weights.ckpt"
        )
        _download_if_missing(ckpt_path, url)

    print(f"Loading model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    hp = ckpt["hyper_parameters"]
    num_categories = hp.get("num_categories", 15)
    embedding_dim = hp.get("embedding_dim", 20)

    # Build Unet3D from the saved model_params
    model_params = {k: v for k, v in hp.items()
                    if k not in ("data_shape", "time_range", "num_categories",
                                 "embedding_dim", "lambda_angle",
                                 "learning_rate", "lr_decay")}
    model_params["data_channels"] = embedding_dim
    net = Unet3D(**model_params)

    # Build embedding
    embedding = nn.Embedding(num_categories, embedding_dim)
    embedding.weight.data.copy_(_init_simplex_embedding(num_categories, embedding_dim))
    embedding.weight.requires_grad = False

    # Load weights from state_dict
    sd = ckpt["state_dict"]
    net_sd = {k.replace("net.", "", 1): v for k, v in sd.items() if k.startswith("net.")}
    emb_sd = {k.replace("embedding.", "", 1): v for k, v in sd.items() if k.startswith("embedding.")}

    net.load_state_dict(net_sd)
    embedding.load_state_dict(emb_sd)

    model = FlowModel(net, embedding)

    # Load EMA weights if available
    ema_shadow = ckpt.get("ema_shadow", {})
    if ema_shadow:
        print("  Applying EMA weights ...")
        for name, param in model.net.named_parameters():
            full_key = f"net.{name}"
            if full_key in ema_shadow:
                param.data.copy_(ema_shadow[full_key])

    model.to(device)
    model.eval()
    return model


# ======================================================================
# Configuration
# ======================================================================

def get_config() -> dict:
    return {
        "model": {
            "checkpoint_path": None,
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
            "air_mask_threshold": 0.0,
            "air_z_frac": 0.0,
            "seed": 42,
        },
        "output": {
            "save_dir": os.path.join(_SCRIPT_DIR, "results"),
            "save_trajectory": False,
        },
    }


# ======================================================================
# Synthetic problem generation
# ======================================================================

def create_synthetic_problem(
    model: FlowModel,
    gravity_fwd: GravityForward,
    config: dict,
) -> tuple:
    """Generate a synthetic gravity inverse problem.

    1. Draw one unconditional sample from the flow model.
    2. Decode to categorical indices -> density.
    3. Compute gravity response + noise -> d_obs.
    """
    device = config["model"]["device"]
    shape = config["domain"]["shape"]

    print("Generating true model via unconditional sampling ...")
    solver = ODEFlowSolver(model=model.net, rtol=1e-5)
    X0 = torch.randn(1, model.embedding_dim, *shape, device=device)
    solution = solver.solve(X0, t0=0.001, tf=1.0, n_steps=32)
    X_final = solution[-1]  # (1, E, X, Y, Z)

    true_categorical = model.decode(X_final)[0].cpu()  # (X, Y, Z)

    n_cats = model.num_categories
    density_lut = torch.tensor(DENSITY_TABLE_15[:n_cats], dtype=torch.float32)
    true_density_vol = density_lut[true_categorical.long()]  # (X, Y, Z)
    true_density_np = true_density_vol.numpy().flatten(order="F")

    d_obs = gravity_fwd.generate_synthetic_data(
        true_density_np,
        noise_percent=config["gravity"]["noise_percent"],
        seed=config["inversion"]["seed"],
    )

    print(f"  Categories: {int(true_categorical.min())} – {int(true_categorical.max())}")
    print(f"  Density: {true_density_np.min():.2f} – {true_density_np.max():.2f} g/cm³")
    print(f"  Gravity: {d_obs.shape[0]} obs, range {d_obs.min():.4f} – {d_obs.max():.4f}")

    return true_categorical, true_density_np, d_obs


# ======================================================================
# Posterior sampling
# ======================================================================

def run_unconditional_sampling(
    model: FlowModel,
    config: dict,
) -> list:
    """Generate unconditional samples using the same seeds as posterior sampling.

    These serve as a baseline — no gravity guidance, pure prior samples.
    """
    device = config["model"]["device"]
    inv = config["inversion"]
    shape = config["domain"]["shape"]

    from tqdm import tqdm

    n_steps = inv["n_steps"]
    t0, tf  = inv["t0"], inv["tf"]
    dt      = (tf - t0) / n_steps
    times   = torch.linspace(t0, tf, n_steps + 1)
    step_fn = inv["method"]  # "euler" or "rk4"

    generator = torch.Generator(device="cpu").manual_seed(inv["seed"])
    results = []

    for idx in range(inv["n_samples"]):
        print(f"\n--- Unconditional sample {idx + 1}/{inv['n_samples']} ---")
        m_t = torch.randn(1, model.embedding_dim, *shape,
                          generator=generator).to(device)

        t_start = time.time()
        pbar = tqdm(range(n_steps), desc="Unconditional ODE", unit="step")
        for i in pbar:
            t = times[i].item()
            T = torch.full((m_t.size(0),), t, device=device)
            with torch.no_grad():
                if step_fn == "rk4":
                    k1 = model.net(m_t, T)
                    k2 = model.net(m_t + 0.5 * dt * k1,
                                   torch.full_like(T, t + 0.5 * dt))
                    k3 = model.net(m_t + 0.5 * dt * k2,
                                   torch.full_like(T, t + 0.5 * dt))
                    k4 = model.net(m_t + dt * k3,
                                   torch.full_like(T, t + dt))
                    v = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                else:
                    v = model.net(m_t, T)
            m_t = m_t + dt * v
            pbar.set_postfix(t=f"{t:.3f}")

        elapsed = time.time() - t_start
        decoded = model.decode(m_t)[0].detach().cpu()
        results.append(decoded)
        print(f"  Completed in {elapsed:.1f}s")

    return results


def run_posterior_sampling(
    model: FlowModel,
    gravity_fwd: GravityForward,
    d_obs: np.ndarray,
    config: dict,
    true_categorical: torch.Tensor = None,
) -> list:
    """Generate posterior samples conditioned on observed gravity data."""
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
        air_mask_threshold=inv["air_mask_threshold"],
        air_z_frac=inv["air_z_frac"],
        static_air_mask=(true_categorical == 0) if true_categorical is not None else None,
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

        if config["output"]["save_trajectory"]:
            final_state = m_final[-1].to(device)
        else:
            final_state = m_final

        decoded = model.decode(final_state)[0].detach().cpu()
        results.append(decoded)
        print(f"  Completed in {elapsed:.1f}s")

    return results


# ======================================================================
# Save results
# ======================================================================

def save_results(
    posterior_samples: list,
    unconditional_samples: list,
    true_categorical: torch.Tensor,
    d_obs: np.ndarray,
    save_dir: str,
):
    """Persist posterior samples, unconditional samples, the true model, and observed data."""
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "d_obs.npy"), d_obs)
    torch.save(true_categorical, os.path.join(save_dir, "true_model.pt"))

    for i, sample in enumerate(posterior_samples):
        torch.save(sample, os.path.join(save_dir, f"posterior_sample_{i}.pt"))

    for i, sample in enumerate(unconditional_samples):
        torch.save(sample, os.path.join(save_dir, f"unconditional_sample_{i}.pt"))

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
    parser.add_argument("--air-mask-threshold", type=float, default=0.5,
                        help="Zero gradient where air probability > threshold (0 to disable)")
    parser.add_argument("--air-z-frac", type=float, default=0.0,
                        help="Hard-mask top fraction of z-axis from gradient, e.g. 0.3")
    parser.add_argument("--noise-percent", type=float, default=2.0,
                        help="Noise level for synthetic data (%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-trajectory", action="store_true",
                        help="Save full ODE trajectory (large!)")
    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_arguments()
    config = get_config()

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
    config["inversion"]["air_mask_threshold"] = args.air_mask_threshold
    config["inversion"]["air_z_frac"] = args.air_z_frac
    config["inversion"]["seed"] = args.seed
    config["gravity"]["noise_percent"] = args.noise_percent
    config["output"]["save_trajectory"] = args.save_trajectory

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
    true_cat, true_density, d_obs = create_synthetic_problem(
        model, gravity_fwd, config
    )

    # 4. Run unconditional sampling (same seeds — baseline for comparison)
    unconditional_samples = run_unconditional_sampling(model, config)

    # 5. Run posterior sampling
    posterior_samples = run_posterior_sampling(model, gravity_fwd, d_obs, config, true_cat)

    # 6. Save everything
    save_results(posterior_samples, unconditional_samples, true_cat, d_obs,
                 config["output"]["save_dir"])


if __name__ == "__main__":
    main()
