import numpy as np
import torch
from typing import Optional, Union
from density_mapping import DifferentiableDensityMapper
from gravity_forward import GravityForward
from posterior_flow_solver import PosteriorFlowSolver, GuidanceSchedule


class ConditionalPosteriorFlowSolver(PosteriorFlowSolver):
    """ODE solver that augments a borehole-conditional prior velocity with gravity guidance.

    Identical to PosteriorFlowSolver in every respect except that the prior velocity is
    computed by passing borehole conditioning (ATb) as a second positional argument to
    the underlying Unet3DCondV3 network:

        v_prior(m, t) = net(m, ATb, t)          # conditional network call

    The gravity misfit gradient, RK4/Euler stepping, trajectory saving, gradient clipping,
    and all other logic are inherited unchanged from PosteriorFlowSolver.

    Args:
        atb:   (1, E, X, Y, Z) borehole conditioning tensor (embedded + masked).
               Broadcast across the batch dimension at each ODE step.
               If None, falls back to the unconditional call net(m, t) — same as base class.
        All other args: see PosteriorFlowSolver.
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
        atb: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            net=net,
            density_mapper=density_mapper,
            gravity_forward=gravity_forward,
            d_obs=d_obs,
            guidance_schedule=guidance_schedule,
            method=method,
            grad_clip=grad_clip,
            normalize_grad=normalize_grad,
            static_air_mask=static_air_mask,
            device=device,
        )
        # borehole conditioning kept on the solver's device, broadcast to batch at call time
        self.atb = atb.to(self.device) if atb is not None else None

    def compute_prior(self, m_t: torch.Tensor, t: float) -> torch.Tensor:
        """Conditional prior velocity: net(m_t, ATb, t) instead of net(m_t, t)."""
        with torch.no_grad():
            T = torch.full((m_t.size(0),), t, device=m_t.device)
            if self.atb is not None:
                atb = self.atb.expand(m_t.size(0), -1, -1, -1, -1)
                return self.net(m_t, atb, T)
            else:
                return self.net(m_t, T)
