import os
import time
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from flowtrain.models import Unet3D
from flowtrain.solvers import ODEFlowSolver
from density_mapping import DifferentiableDensityMapper, DENSITY_TABLE_15
from gravity_forward import GravityForward
from posterior_flow_solver import PosteriorFlowSolver, GuidanceSchedule


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class FlowModel(nn.Module):
    """
    Minimal wrapper holding the Unet3D velocity network and embedding layer.
    """

    def __init__(self, net: Unet3D, embedding: nn.Embedding):
        super().__init__()
        self.net            = net
        self.embedding      = embedding
        self.embedding_dim  = embedding.embedding_dim
        self.num_categories = embedding.num_embeddings

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # cosine similarity against each category centroid → argmax → (B, X, Y, Z)
        emb      = self.embedding.weight              # (K, E)
        B, E, X, Y, Z = x.shape
        x_norm   = F.normalize(x,   dim=1)
        emb_norm = F.normalize(emb, dim=1)
        logits   = (x_norm.unsqueeze(1) * emb_norm.view(1, self.num_categories, E, 1, 1, 1)).sum(dim=2)
        return torch.argmax(logits, dim=1)            # (B, X, Y, Z)


def _download_if_missing(path: str, url: str) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading weights from {url} ...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")


def _init_simplex_embedding(n_cats: int, n_dims: int) -> torch.Tensor:
    # centred-simplex embedding of categories used during training
    init     = torch.zeros(n_cats, n_dims)
    init[:, :n_cats] = torch.eye(n_cats)
    centroid = torch.cat([torch.ones(n_cats) / n_cats, torch.zeros(n_dims - n_cats)])
    init[:, :n_cats] -= centroid[:n_cats].unsqueeze(0)
    return init / init.norm(dim=1, keepdim=True)


def load_pretrained_model(config: dict) -> FlowModel:
    # reconstruct the inference model directly from a Lightning checkpoint
    device    = config["model"]["device"]
    ckpt_path = config["model"]["checkpoint_path"]

    if ckpt_path is None:
        ckpt_path = os.path.join(_SCRIPT_DIR, "..", "geodata-3d-unconditional", "demo_model", "unconditional-weights.ckpt")
        _download_if_missing(
            ckpt_path,
            "https://github.com/chipnbits/flowtrain_stochastic_interpolation"
            "/releases/download/v1.0.0/unconditional-weights.ckpt",
        )

    print(f"Loading model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    hp             = ckpt["hyper_parameters"]
    num_categories = hp.get("num_categories", 15)
    embedding_dim  = hp.get("embedding_dim", 20)

    model_params = {k: v for k, v in hp.items()
                    if k not in ("data_shape", "time_range", "num_categories",
                                 "embedding_dim", "lambda_angle", "learning_rate", "lr_decay")}
    model_params["data_channels"] = embedding_dim
    net = Unet3D(**model_params)

    embedding = nn.Embedding(num_categories, embedding_dim)
    embedding.weight.data.copy_(_init_simplex_embedding(num_categories, embedding_dim))
    embedding.weight.requires_grad = False

    sd      = ckpt["state_dict"]
    net_sd  = {k.replace("net.", "", 1): v for k, v in sd.items() if k.startswith("net.")}
    emb_sd  = {k.replace("embedding.", "", 1): v for k, v in sd.items() if k.startswith("embedding.")}
    net.load_state_dict(net_sd)
    embedding.load_state_dict(emb_sd)

    model = FlowModel(net, embedding)

    ema_shadow = ckpt.get("ema_shadow", {})
    if ema_shadow:
        print("  Applying EMA weights ...")
        for name, param in model.net.named_parameters():
            if f"net.{name}" in ema_shadow:
                param.data.copy_(ema_shadow[f"net.{name}"])

    model.to(device)
    model.eval()
    return model


def get_config() -> dict:
    return {
        "model": {
            "checkpoint_path": None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "domain": {
            "shape":  (64, 64, 64),
            "bounds": ((-1920, 1920), (-1920, 1920), (-1920, 1920)),
        },
        "gravity": {
            "n_receivers_per_side": 32,
            "receiver_height":      30.0,
            "accuracy":             0.1,
            "confidence":           0.95,
        },
        "inversion": {
            "n_samples":     1,
            "n_steps":       50,
            "t0":            0.001,
            "tf":            1.0,
            "mu_0":          5.0,
            "schedule":      "linear_ramp",
            "method":        "euler",
            "temperature":   10.0,
            "grad_clip":     None,
            "normalize_grad": False,
            "seed":          42,
        },
        "output": {
            "save_dir":        os.path.join(_SCRIPT_DIR, "results"),
            "save_trajectory": True,
            "mode":            "density",   # "density" | "categories"
        },
    }


def create_synthetic_problem(model: FlowModel, gravity_fwd: GravityForward, config: dict) -> tuple:
    # sample one unconditional geology model → density → gravity + noise
    device = config["model"]["device"]
    shape  = config["domain"]["shape"]

    print("Generating true model via unconditional sampling ...")
    solver   = ODEFlowSolver(model=model.net, rtol=1e-5)
    X0       = torch.randn(1, model.embedding_dim, *shape, device=device)
    X_final  = solver.solve(X0, t0=0.001, tf=1.0, n_steps=32)[-1]  # (1, E, X, Y, Z)

    true_categorical  = model.decode(X_final)[0].cpu()              # (X, Y, Z)
    density_lut       = torch.tensor(DENSITY_TABLE_15[:model.num_categories], dtype=torch.float32)
    true_density_np   = density_lut[true_categorical.long()].numpy().flatten(order="F")

    d_obs = gravity_fwd.generate_synthetic_data(
        true_density_np,
        accuracy=config["gravity"]["accuracy"],
        confidence=config["gravity"]["confidence"],
        seed=config["inversion"]["seed"],
    )

    print(f"  Categories: {int(true_categorical.min())} – {int(true_categorical.max())}")
    print(f"  Density:    {true_density_np.min():.2f} – {true_density_np.max():.2f} g/cm³")
    print(f"  Gravity:    {d_obs.shape[0]} obs, range {d_obs.min():.4f} – {d_obs.max():.4f}")
    return true_categorical, true_density_np, d_obs


def run_unconditional_sampling(model: FlowModel, config: dict) -> list:
    # prior samples with same seeds as posterior — baseline for comparison
    device  = config["model"]["device"]
    inv     = config["inversion"]
    shape   = config["domain"]["shape"]
    n_steps = inv["n_steps"]
    t0, tf  = inv["t0"], inv["tf"]
    dt      = (tf - t0) / n_steps
    times   = torch.linspace(t0, tf, n_steps + 1)

    generator = torch.Generator(device="cpu").manual_seed(inv["seed"])
    results   = []

    for idx in range(inv["n_samples"]):
        print(f"\n--- Unconditional sample {idx + 1}/{inv['n_samples']} ---")
        m_t     = torch.randn(1, model.embedding_dim, *shape, generator=generator).to(device)
        t_start = time.time()

        pbar = tqdm(range(n_steps), desc="Unconditional ODE", unit="step")
        for i in pbar:
            t = times[i].item()
            T = torch.full((m_t.size(0),), t, device=device)
            with torch.no_grad():
                if inv["method"] == "rk4":
                    k1 = model.net(m_t, T)
                    k2 = model.net(m_t + 0.5 * dt * k1, torch.full_like(T, t + 0.5 * dt))
                    k3 = model.net(m_t + 0.5 * dt * k2, torch.full_like(T, t + 0.5 * dt))
                    k4 = model.net(m_t + dt * k3,        torch.full_like(T, t + dt))
                    v  = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                else:
                    v = model.net(m_t, T)
            m_t = m_t + dt * v
            pbar.set_postfix(t=f"{t:.3f}")

        results.append(model.decode(m_t)[0].detach().cpu())
        print(f"  Completed in {time.time() - t_start:.1f}s")

    return results


def run_posterior_sampling(
    # Posterior Flow Sampling = Unconditional sampling + gravity gradient guidance.
    model: FlowModel,
    gravity_fwd: GravityForward,
    d_obs: np.ndarray,
    config: dict,
    true_categorical: torch.Tensor = None,
) -> list:
    device = config["model"]["device"]
    inv    = config["inversion"]
    shape  = config["domain"]["shape"]

    density_mapper = DifferentiableDensityMapper(
        embedding_weights=model.embedding.weight.clone(),
        density_values=DENSITY_TABLE_15,
        temperature=inv["temperature"],
    ).to(device)

    solver = PosteriorFlowSolver(
        net=model.net,
        density_mapper=density_mapper,
        gravity_forward=gravity_fwd,
        d_obs=d_obs,
        guidance_schedule=GuidanceSchedule(mu_0=inv["mu_0"], schedule=inv["schedule"]),
        method=inv["method"],
        grad_clip=inv["grad_clip"],
        normalize_grad=inv["normalize_grad"],
        static_air_mask=(true_categorical == 0) if true_categorical is not None else None,
    )

    generator    = torch.Generator(device="cpu").manual_seed(inv["seed"])
    results      = []
    trajectories = []   # one decoded-step list per sample (only when save_trajectory)

    for idx in range(inv["n_samples"]):
        print(f"\n--- Posterior sample {idx + 1}/{inv['n_samples']} ---")
        m_0     = torch.randn(1, model.embedding_dim, *shape, generator=generator).to(device)
        t_start = time.time()
        m_final = solver.solve(m_0, t0=inv["t0"], tf=inv["tf"], n_steps=inv["n_steps"],
                               save_trajectory=config["output"]["save_trajectory"])

        if config["output"]["save_trajectory"]:
            # m_final: (n_steps+1, B, E, X, Y, Z) on CPU
            decoded_steps = [
                model.decode(m_final[i].to(device))[0].detach().cpu().numpy()
                for i in range(len(m_final))
            ]
            trajectories.append(decoded_steps)
            final_state = m_final[-1].to(device)
        else:
            final_state = m_final

        results.append(model.decode(final_state)[0].detach().cpu())
        print(f"  Completed in {time.time() - t_start:.1f}s")

    return results, trajectories


def save_results(
    posterior_samples: list,
    unconditional_samples: list,
    true_categorical: torch.Tensor,
    d_obs: np.ndarray,
    save_dir: str,
    trajectories: list = [],
):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "d_obs.npy"), d_obs)
    torch.save(true_categorical, os.path.join(save_dir, "true_model.pt"))
    for i, s in enumerate(posterior_samples):
        torch.save(s, os.path.join(save_dir, f"posterior_sample_{i}.pt"))
    for i, s in enumerate(unconditional_samples):
        torch.save(s, os.path.join(save_dir, f"unconditional_sample_{i}.pt"))
    for i, traj in enumerate(trajectories):
        # traj: list of (X, Y, Z) numpy arrays, one per step
        torch.save([torch.from_numpy(s) for s in traj],
                   os.path.join(save_dir, f"trajectory_{i}.pt"))
    print(f"\nResults saved to {save_dir}")


def main():
    config = get_config()

    model = load_pretrained_model(config)

    print("Building SimPEG gravity simulation ...")
    gravity_fwd = GravityForward(
        shape=config["domain"]["shape"],
        bounds=config["domain"]["bounds"],
        n_receivers_per_side=config["gravity"]["n_receivers_per_side"],
        receiver_height=config["gravity"]["receiver_height"],
    )
    print(f"  Mesh cells: {gravity_fwd.mesh.nC}")
    print(f"  Receivers : {gravity_fwd.survey.nD}")

    true_cat, true_density, d_obs = create_synthetic_problem(model, gravity_fwd, config)
    unconditional_samples = run_unconditional_sampling(model, config)
    posterior_samples, trajectories = run_posterior_sampling(model, gravity_fwd, d_obs, config, true_cat)
    save_results(posterior_samples, unconditional_samples, true_cat, d_obs,
                 config["output"]["save_dir"], trajectories)

    if config["output"]["save_trajectory"]:
        from plotter import plot_trajectory
        for idx, traj in enumerate(trajectories):
            plot_trajectory(traj, t0=config["inversion"]["t0"], tf=config["inversion"]["tf"],
                            sample_idx=idx, mode=config["output"]["mode"])


if __name__ == "__main__":
    main()
