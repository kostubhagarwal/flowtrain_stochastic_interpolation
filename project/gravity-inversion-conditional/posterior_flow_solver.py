import numpy as np
import torch
from typing import Callable, Optional, Union
from tqdm import tqdm
from density_mapping import DifferentiableDensityMapper
from gravity_forward import GravityForward


class GuidanceSchedule:
    """Time-dependent guidance strength mu(t) - This controls how much the likelihood gradient from gravity misfit influences the posterior velocity at each time step.

    Args:
        mu_0:     base guidance strength
        schedule: 'constant' | 'linear_ramp' | 'cosine_ramp' | 'bell', or a callable t -> float
    """

    _BUILTINS = {
        "constant":    lambda t: 1.0,
        "linear_ramp": lambda t: t,
        "cosine_ramp": lambda t: (1.0 - np.cos(np.pi * t)) / 2.0,
        "bell":        lambda t: 4.0 * t * (1.0 - t),
    }

    def __init__(
        self,
        mu_0: float = 1.0,
        schedule: Union[str, Callable[[float], float]] = "linear_ramp",
    ):
        self.mu_0 = mu_0
        self._fn  = schedule if callable(schedule) else self._BUILTINS[schedule]

    def __call__(self, t: float) -> float:
        return self.mu_0 * self._fn(t)


class PosteriorFlowSolver:
    """ODE solver that augments the prior velocity with a likelihood gradient from gravity forward modelling.

    Args:
        net:                trained velocity-prediction network (Unet3D)
        density_mapper:     soft-decode from embedding space to physical density
        gravity_forward:    SimPEG gravity wrapper (forward + Jtvec)
        d_obs:              observed gravity data (n_data,)
        guidance_schedule:  mu(t) controlling guidance strength
        method:          'euler' | 'rk4'
        grad_clip:       if set, clip the norm of the likelihood gradient to this value
        normalize_grad:  scale gradient so its norm equals the prior velocity norm (makes mu_0 a relative weight)
        static_air_mask: boolean tensor (X, Y, Z) marking known air voxels to exclude from guidance
    """

    def __init__(
        self,
        net: torch.nn.Module,
        density_mapper: DifferentiableDensityMapper,
        gravity_forward: GravityForward,
        d_obs: np.ndarray,
        guidance_schedule: GuidanceSchedule,
        method: str = "euler",
        grad_clip: Optional[float] = None,
        normalize_grad: bool = False,
        static_air_mask: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net               = net.to(self.device)
        self.density_mapper    = density_mapper.to(self.device)
        self.gravity_forward   = gravity_forward
        self.d_obs             = d_obs.copy()
        self.guidance_schedule = guidance_schedule
        self.method            = method
        self.grad_clip      = grad_clip
        self.normalize_grad = normalize_grad

        self._mesh_shape = gravity_forward.shape  # (nx, ny, nz)

        # static voxel mask from true model: (1, 1, X, Y, Z) bool on device
        self._static_air_mask = (
            static_air_mask.bool().to(self.device).view(1, 1, *static_air_mask.shape[-3:])
            if static_air_mask is not None else None
        )

        self._last_residual_norm: float = float("nan")

    def compute_prior(self, m_t: torch.Tensor, t: float) -> torch.Tensor:
        with torch.no_grad():
            T = torch.full((m_t.size(0),), t, device=m_t.device)
            return self.net(m_t, T)

    def gravity_misfit_gradient(self, m_t: torch.Tensor) -> torch.Tensor:
        """
        Gradient of || G(rho(m)) - d_obs ||^2 w.r.t. m in embedding space.
        Chain rule: SimPEG J^T(d_pred - d_obs) -> density -> soft-decode -> m.
        """
        B      = m_t.shape[0]
        device = m_t.device
        grads  = []

        for b in range(B):
            m_single    = m_t[b : b + 1].detach().clone().requires_grad_(True)
            density_vol = self.density_mapper.soft_decode_to_density(m_single)  # (1, 1, X, Y, Z)

            density_np  = density_vol[0, 0].detach().cpu().numpy().astype(np.float64).flatten(order="F")
            d_pred      = self.gravity_forward.forward(density_np)
            residual    = d_pred - self.d_obs
            self._last_residual_norm = float(np.linalg.norm(residual))
            grad_density_np = self.gravity_forward.jtvec(density_np, residual)

            grad_density_vol = (
                torch.from_numpy(grad_density_np.reshape(self._mesh_shape, order="F").astype(np.float32))
                .unsqueeze(0).unsqueeze(0).to(device)
            )  # (1, 1, X, Y, Z)

            grad_m = torch.autograd.grad(
                outputs=density_vol,
                inputs=m_single,
                grad_outputs=grad_density_vol,
                retain_graph=False,
            )[0]  # (1, E, X, Y, Z)

            if self._static_air_mask is not None:
                grad_m = grad_m.masked_fill(self._static_air_mask, 0.0)

            grads.append(grad_m.detach())

        return torch.cat(grads, dim=0)  # (B, E, X, Y, Z)

    def compute_posterior(
        self, m_t: torch.Tensor, t: float, v_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Steer the velocity towards reducing gravity misfit. Posterior velocity = prior velocity - mu(t) * grad_misfit.
        """
        if v_prior is None:
            v_prior = self.compute_prior(m_t, t)

        grad_m = self.gravity_misfit_gradient(m_t)

        if self.grad_clip is not None:
            g_norm = grad_m.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            grad_m = grad_m * torch.clamp(self.grad_clip / (g_norm + 1e-8), max=1.0)

        if self.normalize_grad:
            v_norm = v_prior.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            g_norm = grad_m.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            grad_m = grad_m * (v_norm / (g_norm + 1e-8))

        return v_prior - self.guidance_schedule(t) * grad_m

    def _step_euler(self, m_t: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        return m_t + dt * self.compute_posterior(m_t, t)

    def _step_rk4(self, m_t: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """
        determine m_t+dt
        """
        k1 = self.compute_posterior(m_t, t)
        k2 = self.compute_posterior(m_t + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.compute_posterior(m_t + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.compute_posterior(m_t + dt * k3, t + dt)
        return m_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def solve(
        self,
        m_0: torch.Tensor,
        t0: float = 0.001,
        tf: float = 1.0,
        n_steps: int = 10,
        save_trajectory: bool = False,
    ) -> torch.Tensor:
        """Integrate the posterior ODE from t0 to tf.

        Args:
            m_0:              (B, E, X, Y, Z) initial state (Gaussian noise)
            t0, tf:           start / end time
            n_steps:          number of uniform time steps
            save_trajectory:  if True return (n_steps+1, B, E, X, Y, Z), else just the final (B, E, X, Y, Z)
        """
        dt      = (tf - t0) / n_steps
        times   = torch.linspace(t0, tf, n_steps + 1)
        step_fn = self._step_rk4 if self.method == "rk4" else self._step_euler
        m_t     = m_0.to(self.device)

        trajectory = [m_t.clone().cpu()] if save_trajectory else None

        pbar = tqdm(range(n_steps), desc="Posterior ODE", unit="step")
        for i in pbar:
            t   = times[i].item()
            m_t = step_fn(m_t, t, dt)
            pbar.set_postfix(t=f"{t:.3f}", mu=f"{self.guidance_schedule(t):.3f}", res=f"{self._last_residual_norm:.4g}")
            if trajectory is not None:
                trajectory.append(m_t.clone().cpu())

        return torch.stack(trajectory, dim=0) if trajectory is not None else m_t
