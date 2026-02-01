"""
NSC-M3L DISC (Discrete Numerics) Model Types

Defines discrete types for finite difference/FEM/lattice discretizations
per specifications/nsc_m3l_v1.md section 4.4.

Semantic Domain Objects:
- Grids, stencils, FEM spaces, lattice links
- Discrete operators with stability constraints

Denotation:
- ⟦Δ(u)⟧_DISC → LaplacianStencil(u, dx, scheme_id)
- Integrals → quadrature
- Covariant derivatives → discrete connection transport
"""

from dataclasses import dataclass, field
from typing import Union, List, Optional, Tuple, Dict, Set
from enum import Enum
import numpy as np


class BoundaryConditionType(Enum):
    """Types of boundary conditions for discrete operators."""
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"
    ROBIN = "robin"
    OUTFLOW = "outflow"


class StencilType(Enum):
    """Types of finite difference stencils."""
    CENTRAL = "central"
    FORWARD = "forward"
    BACKWARD = "backward"
    UPWIND = "upwind"
    LAX_WENDROFF = "lax_wendroff"
    WENO = "weno"


class FEMElementType(Enum):
    """Types of finite elements."""
    LAGRANGE = "lagrange"
    SERendipity = "serendipity"
    DISCONTINUOUS_GALERKIN = "dg"
    NEDELEC = "nedelec"  # H(curl) elements
    RAVIART_THOMAS = "raviart_thomas"  # H(div) elements


class LatticeType(Enum):
    """Types of lattice discretizations."""
    CUBIC = "cubic"
    BODY_CENTERED_CUBIC = "bcc"
    FACE_CENTERED_CUBIC = "fcc"
    HONEYCOMB = "honeycomb"


# === Grid Types ===

@dataclass
class Grid:
    """
    Structured Cartesian grid for finite difference discretizations.
    
    Attributes:
        dim: Spatial dimension (1, 2, or 3)
        shape: Number of points in each dimension
        spacing: Grid spacing (dx, dy, dz)
        origin: Physical coordinate of origin
        boundary_conditions: Boundary condition specification
    """
    dim: int
    shape: Tuple[int, ...]
    spacing: Tuple[float, ...]
    origin: Tuple[float, ...] = field(default_factory=tuple)
    boundary_conditions: Dict[str, BoundaryConditionType] = field(default_factory=dict)
    
    def __post_init__(self):
        if len(self.shape) != self.dim:
            raise ValueError(f"Shape length {len(self.shape)} must match dim {self.dim}")
        if len(self.spacing) != self.dim:
            raise ValueError(f"Spacing length {len(self.spacing)} must match dim {self.dim}")
        if len(self.origin) != self.dim:
            self.origin = tuple(0.0 for _ in range(self.dim))
    
    @property
    def num_points(self) -> int:
        """Total number of grid points."""
        result = 1
        for n in self.shape:
            result *= n
        return result
    
    def index_to_coord(self, idx: Tuple[int, ...]) -> Tuple[float, ...]:
        """Convert grid index to physical coordinate."""
        if len(idx) != self.dim:
            raise ValueError(f"Index length {len(idx)} must match dim {self.dim}")
        return tuple(
            self.origin[i] + idx[i] * self.spacing[i]
            for i in range(self.dim)
        )
    
    def coord_to_index(self, coord: Tuple[float, ...]) -> Tuple[int, ...]:
        """Convert physical coordinate to grid index."""
        if len(coord) != self.dim:
            raise ValueError(f"Coordinate length {len(coord)} must match dim {self.dim}")
        return tuple(
            int(round((coord[i] - self.origin[i]) / self.spacing[i]))
            for i in range(self.dim)
        )
    
    def get_staggered_grid(self, component: int) -> 'Grid':
        """Get staggered grid for component (collocated at faces)."""
        if component < 0 or component >= self.dim:
            raise ValueError(f"Component {component} out of range for dim {self.dim}")
        
        # Staggered grid has one fewer point in the component direction
        new_shape = list(self.shape)
        new_shape[component] -= 1
        
        # Origin is shifted by half spacing
        new_origin = list(self.origin)
        new_origin[component] += self.spacing[component] / 2
        
        return Grid(
            dim=self.dim,
            shape=tuple(new_shape),
            spacing=self.spacing,
            origin=tuple(new_origin),
            boundary_conditions=self.boundary_conditions
        )
    
    def compute_metric_factors(self) -> Dict:
        """Compute grid metric factors (Jacobian, etc.)."""
        # For Cartesian grid, Jacobian is simply the product of spacings
        jacobian = 1.0
        for dx in self.spacing:
            jacobian *= dx
        
        return {
            "jacobian": jacobian,
            "inverse_jacobian": 1.0 / jacobian if jacobian != 0 else float('inf'),
            "cell_volume": jacobian,
            "cell_area_2d": jacobian  # For 2D, same as volume
        }


@dataclass
class UnstructuredGrid:
    """
    Unstructured mesh for FEM discretizations.
    
    Attributes:
        points: Node coordinates (N x dim array)
        cells: Cell connectivity (M x (k+1) array for k-simplices)
        cell_type: Type of cells (tri, quad, tet, hex, etc.)
        boundary_faces: Boundary face identification
    """
    points: np.ndarray  # Shape: (N, dim)
    cells: np.ndarray  # Shape: (M, n_verts_per_cell)
    cell_type: str  # 'tri', 'quad', 'tet', 'hex', 'prism', 'pyramid'
    boundary_faces: Optional[np.ndarray] = None  # Shape: (K, dim)
    
    @property
    def num_points(self) -> int:
        """Number of nodes."""
        return self.points.shape[0]
    
    @property
    def num_cells(self) -> int:
        """Number of cells."""
        return self.cells.shape[0]
    
    @property
    def dim(self) -> int:
        """Spatial dimension."""
        return self.points.shape[1]


# === Discrete Function Spaces ===

@dataclass
class FEMSpace:
    """
    Finite element function space.
    
    Attributes:
        element_type: Type of finite element
        degree: Polynomial degree
        components: Number of components (1 for scalar, >1 for vector)
        dim: Dimension of the function space (number of DOFs)
        continuity: Continuity type ('C0', 'C1', ' discontinuous')
    """
    element_type: FEMElementType
    degree: int
    components: int = 1
    dim: Optional[int] = None  # Total DOFs, computed if not provided
    continuity: str = "C0"
    
    def __post_init__(self):
        if self.dim is None:
            # Estimate DOFs (will be refined based on actual mesh)
            self.dim = 0  # Placeholder


@dataclass
class FDSpace:
    """
    Finite difference function space.
    
    Attributes:
        stencil: Stencil pattern for derivatives
        order: Order of accuracy
        grid: Associated grid
    """
    stencil: 'Stencil'
    order: int
    grid: Grid
    
    @property
    def dim(self) -> int:
        """Dimension of the space (number of grid points)."""
        return self.grid.num_points


@dataclass
class LatticeSpace:
    """
    Lattice gauge theory space.
    
    Attributes:
        lattice_type: Type of lattice
        basis: Lattice basis vectors
        link_count: Number of links per site
        group: Gauge group (e.g., 'SU(N)', 'U(1)')
    """
    lattice_type: LatticeType
    basis: np.ndarray  # Basis vectors
    link_count: int  # Links per site (2*dim for hypercubic)
    group: str = "SU(2)"
    
    @property
    def dim(self) -> int:
        """Spatial dimension from basis."""
        return self.basis.shape[0]


# === Discrete Operators ===

@dataclass
class Stencil:
    """
    Finite difference stencil.
    
    Attributes:
        pattern: Relative offsets where coefficients are applied
        coefficients: Stencil coefficients
        accuracy: Order of accuracy
        stencil_type: Type of stencil
        boundary_stencil: Optional stencil for boundaries
    """
    pattern: List[Tuple[int, ...]]  # Relative offsets
    coefficients: np.ndarray  # Coefficients for each offset
    accuracy: int
    stencil_type: StencilType = StencilType.CENTRAL
    boundary_stencil: Optional['Stencil'] = None
    
    def __post_init__(self):
        if len(self.pattern) != len(self.coefficients):
            raise ValueError("Pattern and coefficients must have same length")
    
    @property
    def size(self) -> int:
        """Number of points in stencil."""
        return len(self.pattern)
    
    def apply(self, values: np.ndarray, idx: Tuple[int, ...]) -> float:
        """Apply stencil at grid point."""
        result = 0.0
        for offset, coef in zip(self.pattern, self.coefficients):
            neighbor_idx = tuple(idx[i] + offset[i] for i in range(len(idx)))
            # Clamp to grid bounds (simplified)
            clamped_idx = tuple(
                max(0, min(values.shape[i] - 1, neighbor_idx[i]))
                for i in range(values.ndim)
            )
            result += coef * values[clamped_idx]
        return result


@dataclass
class QuadratureRule:
    """
    Quadrature rule for numerical integration.
    
    Attributes:
        nodes: Integration points in reference element
        weights: Quadrature weights
        degree: Degree of exactness
        domain: Domain type ('interval', 'triangle', 'tetrahedron', etc.)
    """
    nodes: np.ndarray  # Shape: (n_points, dim)
    weights: np.ndarray  # Shape: (n_points,)
    degree: int
    domain: str = "interval"
    
    @property
    def num_points(self) -> int:
        """Number of integration points."""
        return len(self.weights)


@dataclass
class InterpolationMatrix:
    """
    Interpolation operator between discrete spaces.
    
    Attributes:
        ops: List of stencil operations
        method: Interpolation method ('linear', 'lagrange', 'spectral')
        source_space: Source function space
        target_space: Target function space
    """
    ops: List[Stencil]
    method: str
    source_space: Union[FEMSpace, FDSpace]
    target_space: Union[FEMSpace, FDSpace]


# === Discrete Fields ===

@dataclass
class DiscreteField:
    """
    Field defined on a discrete function space.
    
    Attributes:
        space: The discrete function space
        values: Field values at DOF locations
        dof_indices: Index mapping for DOFs
    """
    space: Union[FEMSpace, FDSpace, LatticeSpace]
    values: np.ndarray
    dof_indices: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.values.shape != (self.space.dim,):
            raise ValueError(f"Values shape {self.values.shape} must match space dim {self.space.dim}")
    
    def norm(self, p: int = 2) -> float:
        """Compute p-norm of the field."""
        return float(np.sum(np.abs(self.values) ** p) ** (1/p))


# === Stability Properties ===

@dataclass
class StabilityInfo:
    """
    Stability properties of a discrete operator.
    
    Attributes:
        cond_number: Condition number (if applicable)
        spectral_radius: Spectral radius of operator
        CFL_limit: CFL stability limit (for time-dependent problems)
        is_stable: Whether operator is unconditionally stable
    """
    cond_number: Optional[float] = None
    spectral_radius: Optional[float] = None
    CFL_limit: Optional[float] = None
    is_stable: bool = True


# === Discrete Operators ===

@dataclass
class DiscreteOperator:
    """
    Abstract discrete operator.
    
    Attributes:
        stencil: Stencil or matrix representation
        domain: Input discrete space
        codomain: Output discrete space
        stability: Stability information
    """
    stencil: Union[Stencil, np.ndarray]  # Matrix form for linear operators
    domain: Union[FEMSpace, FDSpace, LatticeSpace]
    codomain: Union[FEMSpace, FDSpace, LatticeSpace]
    stability: StabilityInfo = field(default_factory=StabilityInfo)


# === Connection/Lattice Links ===

@dataclass
class LatticeLink:
    """
    Lattice link variable (for gauge theories).
    
    Attributes:
        site: Origin site index
        direction: Direction of link (0=+x, 1=-x, 2=+y, etc.)
        group_element: Group element on link (SU(N) matrix)
    """
    site: Tuple[int, ...]
    direction: int
    group_element: np.ndarray  # SU(N) matrix


@dataclass
class ConnectionTransport:
    """
    Discrete connection transport operator.
    
    Attributes:
        from_site: Source site
        to_site: Target site
        path: Path through lattice links
        transport_matrix: Parallel transport matrix
    """
    from_site: Tuple[int, ...]
    to_site: Tuple[int, ...]
    path: List[LatticeLink]
    transport_matrix: np.ndarray


# === Union Type ===

DiscreteType = Union[
    Grid, UnstructuredGrid,
    FEMSpace, FDSpace, LatticeSpace,
    Stencil, QuadratureRule, InterpolationMatrix,
    DiscreteField, DiscreteOperator
]


# === Type Utilities ===

def is_grid_type(t: DiscreteType) -> bool:
    """Check if type is a grid type."""
    return isinstance(t, (Grid, UnstructuredGrid))


def is_function_space(t: DiscreteType) -> bool:
    """Check if type is a discrete function space."""
    return isinstance(t, (FEMSpace, FDSpace, LatticeSpace))


def get_space_dim(t: DiscreteType) -> int:
    """Get dimension of discrete space."""
    if isinstance(t, Grid):
        return t.dim
    elif isinstance(t, UnstructuredGrid):
        return t.dim
    elif isinstance(t, FEMSpace):
        return t.dim or 0
    elif isinstance(t, FDSpace):
        return t.dim
    elif isinstance(t, LatticeSpace):
        return t.dim
    elif isinstance(t, DiscreteField):
        return get_space_dim(t.space)
    elif isinstance(t, DiscreteOperator):
        return get_space_dim(t.domain)
    return 0


def compatible_discrete_spaces(space1: DiscreteType, space2: DiscreteType) -> bool:
    """Check if two discrete spaces are compatible for operations."""
    return get_space_dim(space1) == get_space_dim(space2)
