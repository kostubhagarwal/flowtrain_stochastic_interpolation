"""
Posterior flow ODE solver with likelihood guidance for gravity inversion.

Combines the unconditional prior velocity from a trained flow-matching model
with a data-fidelity gradient computed via SimPEG, enabling posterior sampling
without retraining.

At each step:

    dm/dt = v_prior(m, t) - mu(t) * grad_m || G(rho(m)) - d_obs ||^2
"""

import numpy as np
import torch
from typing import Callable, Optional, Union

from tqdm import tqdm

from density_mapping import DifferentiableDensityMapper
from gravity_forward import GravityForward


# ======================================================================
# Guidance schedule
# ======================================================================

class GuidanceSchedule:
    """Time-dependent guidance strength mu(t).

    Built-in schedules
    ------------------
    ``'constant'``     : mu(t) = mu_0
    ``'linear_ramp'``  : mu(t) = mu_0 * t            (0 at t=0, mu_0 at t=1)
    ``'cosine_ramp'``  : mu(t) = mu_0 * (1 - cos(pi*t)) / 2
    ``'bell'``         : mu(t) = mu_0 * 4*t*(1 - t)  (peak at t=0.5)

    A callable ``schedule(t) -> float`` can be passed instead.

    Parameters
    ----------
    mu_0 : float
        Base guidance strength.
    schedule : str or callable
        One of the built-in names or a user function.
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
        if callable(schedule):
            self._fn = schedule
        elif schedule in self._BUILTINS:
            self._fn = self._BUILTINS[schedule]
        else:
            raise ValueError(
                f"Unknown schedule '{schedule}'. "
                f"Choose from {list(self._BUILTINS)} or pass a callable."
            )

    def __call__(self, t: float) -> float:
        return self.mu_0 * self._fn(t)


# ======================================================================
# Posterior flow solver
# ======================================================================

class PosteriorFlowSolver:
    """Manual-stepping ODE solver that augments the prior velocity with a
    likelihood gradient from gravity forward modelling.

    Parameters
    ----------
    net : torch.nn.Module
        Trained velocity-prediction network (Unet3D).
    density_mapper : DifferentiableDensityMapper
        Soft-decode from embedding space to physical density.
    gravity_forward : GravityForward
        SimPEG gravity wrapper (forward + Jtvec).
    d_obs : np.ndarray
        Observed gravity data ``(n_data,)``.
    guidance_schedule : GuidanceSchedule
        mu(t) controlling guidance strength.
    method : ``'euler'`` | ``'rk4'``
        ODE integration method.
    grad_clip : float or None
        If set, clip the norm of the likelihood gradient to this value.
    normalize_grad : bool
        If *True*, scale the gradient so its norm equals the prior velocity norm
        (makes ``mu_0`` act as a relative weight).
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
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = net.to(self.device)
        self.density_mapper = density_mapper.to(self.device)
        self.gravity_forward = gravity_forward
        self.d_obs = d_obs.copy()
        self.guidance_schedule = guidance_schedule
        self.method = method
        self.grad_clip = grad_clip
        self.normalize_grad = normalize_grad

        # Pre-computed mesh shape for numpy ↔ torch reshaping
        self._mesh_shape = gravity_forward.shape  # (nx, ny, nz)

        # Updated each step as a side-effect of compute_likelihood_gradient
        self._last_residual_norm: float = float("nan")

    # ------------------------------------------------------------------
    # Prior velocity
    # ------------------------------------------------------------------
    def compute_prior_velocity(
        self, m_t: torch.Tensor, t: float
    ) -> torch.Tensor:
        """Evaluate the unconditional velocity field (no grad)."""
        with torch.no_grad():
            T = torch.full((m_t.size(0),), t, device=m_t.device)
            return self.net(m_t, T)

    # ------------------------------------------------------------------
    # Likelihood gradient
    # ------------------------------------------------------------------
    def compute_likelihood_gradient(
        self, m_t: torch.Tensor
    ) -> torch.Tensor:
        """Gradient of || G(rho(m)) - d_obs ||^2 w.r.t. m in embedding space.

        Uses a two-stage chain rule:

        1. **SimPEG** (numpy): ``J^T (d_pred - d_obs)``  →  grad w.r.t. density.
        2. **torch autograd**: backprop through the soft-decode to get grad w.r.t. m.
        """
        B = m_t.shape[0]
        device = m_t.device
        grads = []

        for b in range(B):
            # --- torch: differentiable soft-decode ------------------
            m_single = m_t[b : b + 1].detach().clone().requires_grad_(True)
            density_vol = self.density_mapper.soft_decode_to_density(m_single)
            # (1, 1, X, Y, Z)

            # --- numpy: SimPEG forward + Jtvec ----------------------
            # Flatten with Fortran order so x varies fastest (SimPEG convention)
            density_np = (
                density_vol[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .flatten(order="F")
            )

            d_pred = self.gravity_forward.forward(density_np)
            residual = d_pred - self.d_obs  # (n_data,)
            self._last_residual_norm = float(np.linalg.norm(residual))
            grad_density_np = self.gravity_forward.jtvec(density_np, residual)

            # --- back to torch: chain-rule through soft-decode ------
            grad_density_vol = (
                torch.from_numpy(
                    grad_density_np.reshape(self._mesh_shape, order="F")
                    .astype(np.float32)
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )  # (1, 1, X, Y, Z)

            grad_m = torch.autograd.grad(
                outputs=density_vol,
                inputs=m_single,
                grad_outputs=grad_density_vol,
                retain_graph=False,
            )[0]  # (1, E, X, Y, Z)

            grads.append(grad_m.detach())

        return torch.cat(grads, dim=0)  # (B, E, X, Y, Z)

    # ------------------------------------------------------------------
    # Posterior velocity
    # ------------------------------------------------------------------
    def _posterior_velocity(
        self,
        m_t: torch.Tensor,
        t: float,
        v_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine prior velocity and likelihood gradient."""
        if v_prior is None:
            v_prior = self.compute_prior_velocity(m_t, t)

        grad_m = self.compute_likelihood_gradient(m_t)

        # Optional gradient processing
        if self.grad_clip is not None:
            g_norm = grad_m.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            grad_m = grad_m * torch.clamp(self.grad_clip / (g_norm + 1e-8), max=1.0)

        if self.normalize_grad:
            v_norm = v_prior.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            g_norm = grad_m.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            grad_m = grad_m * (v_norm / (g_norm + 1e-8))

        mu_t = self.guidance_schedule(t)
        return v_prior - mu_t * grad_m

    # ------------------------------------------------------------------
    # Stepping methods
    # ------------------------------------------------------------------
    def _step_euler(
        self, m_t: torch.Tensor, t: float, dt: float
    ) -> torch.Tensor:
        dm_dt = self._posterior_velocity(m_t, t)
        return m_t + dt * dm_dt

    def _step_rk4(
        self, m_t: torch.Tensor, t: float, dt: float
    ) -> torch.Tensor:
        k1 = self._posterior_velocity(m_t, t)
        k2 = self._posterior_velocity(m_t + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._posterior_velocity(m_t + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._posterior_velocity(m_t + dt * k3, t + dt)
        return m_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------
    def solve(
        self,
        m_0: torch.Tensor,
        t0: float = 0.001,
        tf: float = 1.0,
        n_steps: int = 10,
        save_trajectory: bool = False,
    ) -> torch.Tensor:
        """Integrate the posterior ODE from *t0* to *tf*.

        Parameters
        ----------
        m_0 : torch.Tensor
            ``(B, E, X, Y, Z)`` initial state (Gaussian noise).
        t0, tf : float
            Start / end time.
        n_steps : int
            Number of uniform time steps. Default is 10 (light; use 50 for
            production runs).
        save_trajectory : bool
            If *True* return ``(n_steps+1, B, E, X, Y, Z)``, otherwise just
            the final state ``(B, E, X, Y, Z)``.

        Returns
        -------
        torch.Tensor
        """
        dt = (tf - t0) / n_steps
        times = torch.linspace(t0, tf, n_steps + 1)

        step_fn = self._step_rk4 if self.method == "rk4" else self._step_euler

        m_t = m_0.to(self.device)

        if save_trajectory:
            trajectory = [m_t.clone().cpu()]
        pbar = tqdm(range(n_steps), desc="Posterior ODE", unit="step")
        for i in pbar:
            t = times[i].item()
            m_t = step_fn(m_t, t, dt)
            # _last_residual_norm is set as a side-effect inside compute_likelihood_gradient
            pbar.set_postfix(
                t=f"{t:.3f}",
                mu=f"{self.guidance_schedule(t):.3f}",
                res=f"{self._last_residual_norm:.4g}",
            )
            if save_trajectory:
                trajectory.append(m_t.clone().cpu())

        if save_trajectory:
            return torch.stack(trajectory, dim=0)
        return m_t
