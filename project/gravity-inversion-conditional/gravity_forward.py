import numpy as np
from scipy.stats import norm
from typing import Tuple
from discretize import TensorMesh
from SimPEG import maps
from SimPEG.potential_fields import gravity


class GravityForward:
    """SimPEG gravity simulation on a regular 3D mesh (64³ by default).

    Builds the mesh, receiver survey, and simulation once at init.
    Then exposes forward (G) and adjoint (G^T) for inversion.

    Args:
        shape:                (nx, ny, nz) number of cells. default (64, 64, 64)
        bounds:               ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in metres
        n_receivers_per_side: receivers placed on an n×n grid on the top surface
        receiver_height:      height (m) above the top surface for the receiver plane
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
        reference_density: float = 2.67,  # g/cm³ — subtracted before forward to remove DC edge signal
    ):
        self.shape = shape
        self.bounds = bounds
        self.n_receivers_per_side = n_receivers_per_side
        self.receiver_height = receiver_height
        self.reference_density = reference_density

        self.mesh       = self._build_mesh()
        self.survey     = self._build_survey()
        self.simulation = self._build_simulation()

    def _build_mesh(self) -> TensorMesh:
        hx = np.ones(self.shape[0]) * (self.bounds[0][1] - self.bounds[0][0]) / self.shape[0]
        hy = np.ones(self.shape[1]) * (self.bounds[1][1] - self.bounds[1][0]) / self.shape[1]
        hz = np.ones(self.shape[2]) * (self.bounds[2][1] - self.bounds[2][0]) / self.shape[2]
        return TensorMesh([hx, hy, hz], origin=np.array([b[0] for b in self.bounds]))

    def _build_survey(self) -> gravity.survey.Survey:
        x_min, x_max = self.bounds[0] # inset from edges to avoid boundary effects
        y_min, y_max = self.bounds[1] # inset from edges to avoid boundary effects
        z_top = self.bounds[2][1] + self.receiver_height

        n = self.n_receivers_per_side
        xs = np.linspace(x_min, x_max, n + 2)[1:-1]  # trim edges
        ys = np.linspace(y_min, y_max, n + 2)[1:-1]
        xr, yr = np.meshgrid(xs, ys)
        zr = np.full_like(xr.ravel(), z_top)

        receiver_locations = np.column_stack([xr.ravel(), yr.ravel(), zr])
        receiver_list = gravity.receivers.Point(receiver_locations, components=["gz"])
        source_field  = gravity.sources.SourceField(receiver_list=[receiver_list])
        return gravity.survey.Survey(source_field)

    def _build_simulation(self) -> gravity.simulation.Simulation3DIntegral:
        return gravity.simulation.Simulation3DIntegral(
            survey=self.survey,
            mesh=self.mesh,
            rhoMap=maps.IdentityMap(nP=self.mesh.nC),
            store_sensitivities="ram",
        )

    def forward(self, density: np.ndarray) -> np.ndarray:
        # Forward on density anomaly δρ = ρ - ρ_ref to suppress the DC edge signal
        return self.simulation.dpred(density - self.reference_density)

    def jtvec(self, density: np.ndarray, v: np.ndarray) -> np.ndarray:
        # Jacobian transpose is linear so reference cancels — pass anomaly for consistency
        return self.simulation.Jtvec(density - self.reference_density, v)

    def generate_synthetic_data(
        self,
        true_density: np.ndarray,
        accuracy: float = 0.1,      # instrument accuracy in mGal
        confidence: float = 0.95,   # confidence level the accuracy is quoted at
        seed: int = 42,
    ) -> np.ndarray:
        # forward model + instrument noise (e.g. accuracy=0.1 mGal at 95% confidence → std.dev = 0.1 / ppf(0.975) ≈ 0.051 mGal)
        rng     = np.random.default_rng(seed)
        d_clean = self.forward(true_density)
        sigma   = accuracy / norm.ppf((1.0 + confidence) / 2.0)
        return d_clean + rng.normal(0.0, sigma, size=d_clean.shape)
