import os
import time
import urllib.request
import numpy as np
import torch
from tqdm import tqdm
from geogen.dataset import GeoData3DStreamingDataset
from density_mapping import DifferentiableDensityMapper, DENSITY_TABLE_15
from gravity_forward import GravityForward
from boreholes import make_combined_mask
from training.model_train_sh_inference_cond import Geo3DStochInterp
from conditional_posterior_flow_solver import ConditionalPosteriorFlowSolver, GuidanceSchedule


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _download_if_missing(path: str, url: str) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading weights from {url} ...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")


def load_pretrained_model(config: dict) -> Geo3DStochInterp:
    # load conditional model from Lightning checkpoint, apply EMA weights if available
    device    = config["model"]["device"]
    ckpt_path = config["model"]["checkpoint_path"]

    if ckpt_path is None:
        ckpt_path = os.path.join(_SCRIPT_DIR, "demo_model", "conditional-weights.ckpt")
        _download_if_missing(
            ckpt_path,
            "https://github.com/chipnbits/flowtrain_stochastic_interpolation"
            "/releases/download/v1.0.0/conditional-weights.ckpt",
        )

    print(f"Loading model from: {ckpt_path}")
    model = Geo3DStochInterp.load_from_checkpoint(ckpt_path, map_location=device)

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    ema_shadow = ckpt.get("ema_shadow", {})
    if ema_shadow:
        print("  Applying EMA weights ...")
        for name, param in model.named_parameters():
            if name in ema_shadow:
                param.data.copy_(ema_shadow[name].to(device))

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
        "boreholes": {
            "n_bores_min": 8,
            "n_bores_max": 16,
        },
        "gravity": {
            "n_receivers_per_side": 32,
            "receiver_height":      30.0,
            "accuracy":             0.1,
            "confidence":           0.95,
        },
        "inversion": {
            "n_samples":      4,
            "n_steps":        50,
            "t0":             0.001,
            "tf":             1.0,
            "mu_0":           20.0,
            "schedule":       "bell",
            "method":         "euler",
            "temperature":    5.0,
            "grad_clip":      None,
            "normalize_grad": False,
            "seed":           42,
        },
        "output": {
            "save_dir":        os.path.join(_SCRIPT_DIR, "results"),
            "save_trajectory": True,
            "mode":            "density",   # "density" | "categories"
        },
    }


def create_synthetic_problem(
    model: Geo3DStochInterp, gravity_fwd: GravityForward, config: dict
) -> tuple:
    # draw one geology from the generative dataset → apply boreholes → embed → compute gravity
    device = config["model"]["device"]
    shape  = config["domain"]["shape"]

    print("Generating true model from GeoData3DStreamingDataset ...")
    dataset = GeoData3DStreamingDataset(
        model_resolution=shape,
        model_bounds=config["domain"]["bounds"],
        dataset_size=100_000,
        device=device,
    )

    # raw_data: (1, 1, X, Y, Z) with values in [-1, 13]  (air = -1, rock = 0-13)
    raw_data = dataset[0].unsqueeze(0).to(device)

    # borehole mask: observed voxels keep their category, unobserved → -1 (air sentinel)
    boreholes_mask = make_combined_mask(
        raw_data,
        n_bores_min=config["boreholes"]["n_bores_min"],
        n_bores_max=config["boreholes"]["n_bores_max"],
    )   # (1, 1, X, Y, Z) bool
    boreholes      = raw_data.clone()
    boreholes[~boreholes_mask] = -1

    # embed true model and zero out unobserved voxels → ATb conditioning tensor for the ODE
    with torch.no_grad():
        X1         = model.embed(raw_data)                                  # (1, E, X, Y, Z)
        mask_embed = boreholes_mask.expand(-1, X1.shape[1], -1, -1, -1)
        atb        = X1 * mask_embed                                        # (1, E, X, Y, Z)

    # shift from [-1, 13] → [0, 14] for density lookup and saving (air = 0, rock = 1-14)
    true_categorical = (raw_data.squeeze() + 1).long().cpu()                # (X, Y, Z)
    boreholes_cat    = (boreholes.squeeze() + 1).clamp(min=0).long().cpu()  # (X, Y, Z), 0 = unobserved/air

    density_lut     = torch.tensor(DENSITY_TABLE_15, dtype=torch.float32)
    true_density_np = density_lut[true_categorical].numpy().flatten(order="F")

    d_obs = gravity_fwd.generate_synthetic_data(
        true_density_np,
        accuracy=config["gravity"]["accuracy"],
        confidence=config["gravity"]["confidence"],
        seed=config["inversion"]["seed"],
    )

    print(f"  Categories:      {int(true_categorical.min())} – {int(true_categorical.max())}")
    print(f"  Borehole voxels: {boreholes_mask.sum().item()} / {boreholes_mask.numel()}")
    print(f"  Density:         {true_density_np.min():.2f} – {true_density_np.max():.2f} g/cm³")
    print(f"  Gravity:         {d_obs.shape[0]} obs, range {d_obs.min():.4f} – {d_obs.max():.4f}")
    return true_categorical, boreholes_cat, atb, d_obs, boreholes_mask


def run_conditional_prior_sampling(
    # borehole-conditioned ODE samples without gravity guidance — baseline comparison
    model: Geo3DStochInterp,
    atb: torch.Tensor,
    config: dict,
) -> list:
    device  = config["model"]["device"]
    inv     = config["inversion"]
    shape   = config["domain"]["shape"]
    n_steps = inv["n_steps"]
    t0, tf  = inv["t0"], inv["tf"]
    dt      = (tf - t0) / n_steps
    times   = torch.linspace(t0, tf, n_steps + 1)

    atb_dev   = atb.to(device)
    generator = torch.Generator(device="cpu").manual_seed(inv["seed"])
    results   = []

    for idx in range(inv["n_samples"]):
        print(f"\n--- Conditional prior sample {idx + 1}/{inv['n_samples']} ---")
        m_t     = torch.randn(1, model.embedding_dim, *shape, generator=generator).to(device)
        t_start = time.time()

        pbar = tqdm(range(n_steps), desc="Conditional prior ODE", unit="step")
        for i in pbar:
            t = times[i].item()
            T = torch.full((m_t.size(0),), t, device=device)
            A = atb_dev.expand(m_t.size(0), -1, -1, -1, -1)
            with torch.no_grad():
                if inv["method"] == "rk4":
                    k1 = model.net(m_t,                A, T)
                    k2 = model.net(m_t + 0.5*dt*k1,   A, torch.full_like(T, t + 0.5*dt))
                    k3 = model.net(m_t + 0.5*dt*k2,   A, torch.full_like(T, t + 0.5*dt))
                    k4 = model.net(m_t + dt*k3,        A, torch.full_like(T, t + dt))
                    v  = (k1 + 2*k2 + 2*k3 + k4) / 6.0
                else:
                    v = model.net(m_t, A, T)
            m_t = m_t + dt * v
            pbar.set_postfix(t=f"{t:.3f}")

        results.append(model.decode(m_t)[0].detach().cpu())
        print(f"  Completed in {time.time() - t_start:.1f}s")

    return results


def run_conditional_posterior_sampling(
    # Conditional Posterior Flow Sampling = borehole conditioning + gravity gradient guidance.
    model: Geo3DStochInterp,
    atb: torch.Tensor,
    gravity_fwd: GravityForward,
    d_obs: np.ndarray,
    config: dict,
    true_categorical: torch.Tensor = None,
    boreholes_mask: torch.Tensor = None,
) -> list:
    device = config["model"]["device"]
    inv    = config["inversion"]
    shape  = config["domain"]["shape"]

    density_mapper = DifferentiableDensityMapper(
        embedding_weights=model.embedding.weight.clone(),
        density_values=DENSITY_TABLE_15,
        temperature=inv["temperature"],
    ).to(device)

    solver = ConditionalPosteriorFlowSolver(
        net=model.net,
        density_mapper=density_mapper,
        gravity_forward=gravity_fwd,
        d_obs=d_obs,
        guidance_schedule=GuidanceSchedule(mu_0=inv["mu_0"], schedule=inv["schedule"]),
        method=inv["method"],
        grad_clip=inv["grad_clip"],
        normalize_grad=inv["normalize_grad"],
        static_air_mask=(true_categorical == 0) if true_categorical is not None else None,
        atb=atb,
        borehole_mask=boreholes_mask,
    )

    generator    = torch.Generator(device="cpu").manual_seed(inv["seed"])
    results      = []
    trajectories = []   # one decoded-step list per sample (only when save_trajectory)

    for idx in range(inv["n_samples"]):
        print(f"\n--- Conditional posterior sample {idx + 1}/{inv['n_samples']} ---")
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
    prior_samples: list,
    true_categorical: torch.Tensor,
    boreholes_cat: torch.Tensor,
    d_obs: np.ndarray,
    save_dir: str,
    trajectories: list = [],
):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "d_obs.npy"), d_obs)
    torch.save(true_categorical, os.path.join(save_dir, "true_model.pt"))
    torch.save(boreholes_cat,    os.path.join(save_dir, "boreholes.pt"))
    for i, s in enumerate(posterior_samples):
        torch.save(s, os.path.join(save_dir, f"conditional_posterior_sample_{i}.pt"))
    for i, s in enumerate(prior_samples):
        torch.save(s, os.path.join(save_dir, f"conditional_prior_sample_{i}.pt"))
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

    true_cat, boreholes_cat, atb, d_obs, boreholes_mask = create_synthetic_problem(model, gravity_fwd, config)
    prior_samples     = run_conditional_prior_sampling(model, atb, config)
    posterior_samples, trajectories = run_conditional_posterior_sampling(
        model, atb, gravity_fwd, d_obs, config, true_cat, boreholes_mask
    )
    save_results(posterior_samples, prior_samples, true_cat, boreholes_cat, d_obs,
                 config["output"]["save_dir"], trajectories)

    if config["output"]["save_trajectory"]:
        from plotter import plot_trajectory
        for idx, traj in enumerate(trajectories):
            plot_trajectory(traj, t0=config["inversion"]["t0"], tf=config["inversion"]["tf"],
                            sample_idx=idx, mode=config["output"]["mode"])


if __name__ == "__main__":
    main()
