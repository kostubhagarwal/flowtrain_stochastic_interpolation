"""
SimPEG gravity forward modelling wrapper for the 64^3 geological domain.

Provides forward operator G, adjoint G^T (via Jtvec), and synthetic data
generation for posterior flow sampling experiments.
"""

import numpy as np
from typing import Tuple

from discretize import TensorMesh
from SimPEG import maps
from SimPEG.potential_fields import gravity


class GravityForward:
    """Wrap SimPEG gravity simulation on a regular 3-D mesh.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Number of cells ``(nx, ny, nz)``.  Default ``(64, 64, 64)``.
    bounds : tuple[tuple[float, float], ...]
        ``((x_min, x_max), (y_min, y_max), (z_min, z_max))`` in metres.
    n_receivers_per_side : int
        Receivers are placed on an ``n × n`` grid on the top surface.
    receiver_height : float
        Height (m) above the top surface for the receiver plane.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int] = (64, 64, 64),
        bounds: Tuple[Tuple[float, float], ...] = (
            (-1920, 1920),
            (-1920, 1920),
            (-1920, 1920),
        ),
        n_receivers_per_side: int = 16,
        receiver_height: float = 30.0,
    ):
        self.shape = shape
        self.bounds = bounds
        self.n_receivers_per_side = n_receivers_per_side
        self.receiver_height = receiver_height

        self.mesh = self._build_mesh()
        self.survey = self._build_survey()
        self.simulation = self._build_simulation()

    # ------------------------------------------------------------------
    def _build_mesh(self) -> TensorMesh:
        """Create a ``discretize.TensorMesh`` matching the domain."""
        hx = np.ones(self.shape[0]) * (self.bounds[0][1] - self.bounds[0][0]) / self.shape[0]
        hy = np.ones(self.shape[1]) * (self.bounds[1][1] - self.bounds[1][0]) / self.shape[1]
        hz = np.ones(self.shape[2]) * (self.bounds[2][1] - self.bounds[2][0]) / self.shape[2]
        origin = np.array([b[0] for b in self.bounds])
        return TensorMesh([hx, hy, hz], origin=origin)

    # ------------------------------------------------------------------
    def _build_survey(self) -> gravity.survey.Survey:
        """Regular receiver grid on the top surface."""
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        z_top = self.bounds[2][1] + self.receiver_height

        n = self.n_receivers_per_side
        xs = np.linspace(x_min, x_max, n + 2)[1:-1]  # avoid edges
        ys = np.linspace(y_min, y_max, n + 2)[1:-1]
        xr, yr = np.meshgrid(xs, ys)
        xr, yr = xr.ravel(), yr.ravel()
        zr = np.full_like(xr, z_top)

        receiver_locations = np.column_stack([xr, yr, zr])
        receiver_list = gravity.receivers.Point(receiver_locations, components=["gz"])
        source_field = gravity.sources.SourceField(receiver_list=[receiver_list])
        return gravity.survey.Survey(source_field)

    # ------------------------------------------------------------------
    def _build_simulation(self) -> gravity.simulation.Simulation3DIntegral:
        return gravity.simulation.Simulation3DIntegral(
            survey=self.survey,
            mesh=self.mesh,
            rhoMap=maps.IdentityMap(nP=self.mesh.nC),
            store_sensitivities="ram",
        )

    # ------------------------------------------------------------------
    def forward(self, density: np.ndarray) -> np.ndarray:
        """Compute predicted gravity from a density model.

        Parameters
        ----------
        density : np.ndarray
            ``(n_cells,)`` density values ordered consistently with ``self.mesh``.

        Returns
        -------
        np.ndarray
            ``(n_data,)`` predicted gravity data.
        """
        return self.simulation.dpred(density)

    # ------------------------------------------------------------------
    def jtvec(self, density: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute *J*^T @ *v*  (adjoint of the Jacobian times a vector).

        Parameters
        ----------
        density : np.ndarray
            ``(n_cells,)`` density model at which to evaluate the Jacobian.
        v : np.ndarray
            ``(n_data,)`` vector (typically the residual ``d_pred - d_obs``).

        Returns
        -------
        np.ndarray
            ``(n_cells,)`` gradient of the misfit w.r.t. density.
        """
        return self.simulation.Jtvec(density, v)

    # ------------------------------------------------------------------
    def generate_synthetic_data(
        self,
        true_density: np.ndarray,
        noise_percent: float = 2.0,
        seed: int = 42,
    ) -> np.ndarray:
        """Forward-model + additive Gaussian noise.

        Parameters
        ----------
        true_density : np.ndarray
            ``(n_cells,)`` ground-truth density.
        noise_percent : float
            Percentage of |d_clean| used as noise std-dev.
        seed : int
            Random seed.

        Returns
        -------
        np.ndarray
            ``(n_data,)`` observed data with noise.
        """
        rng = np.random.default_rng(seed)
        d_clean = self.forward(true_density)
        noise_std = noise_percent / 100.0 * np.abs(d_clean)
        noise_std = np.maximum(noise_std, 1e-10)  # floor
        return d_clean + rng.normal(0, noise_std)
