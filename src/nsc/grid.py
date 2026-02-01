"""
NSC-M3L Grid Management

Implements grid types for finite difference/FEM discretizations.

Grid types:
- Structured Cartesian grids
- Unstructured meshes
- Staggered grids
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Callable
import numpy as np

from .disc_types import (
    Grid as GridType, UnstructuredGrid as UnstructuredGridType,
    BoundaryConditionType, Stencil, FEMSpace, FEMElementType,
    LatticeType, LatticeSpace
)


class Grid:
    """
    Structured Cartesian grid for finite difference discretizations.
    
    Provides:
    - Index to coordinate conversion
    - Staggered grid access
    - Metric factor computation
    - Boundary handling
    """
    
    def __init__(self, shape: Tuple[int, ...], 
                 spacing: Tuple[float, ...],
                 origin: Tuple[float, ...] = None,
                 boundary_conditions: Dict[str, BoundaryConditionType] = None):
        """
        Initialize structured grid.
        
        Args:
            shape: Number of points in each dimension
            spacing: Grid spacing in each dimension
            origin: Physical coordinate of origin (default: (0,0,...))
            boundary_conditions: Boundary condition specification
        """
        self.dim = len(shape)
        self.shape = shape
        self.spacing = spacing
        
        if origin is None:
            self.origin = tuple(0.0 for _ in range(self.dim))
        else:
            self.origin = origin
        
        if boundary_conditions is None:
            self.boundary_conditions = {}
        else:
            self.boundary_conditions = boundary_conditions
        
        if len(spacing) != self.dim:
            raise ValueError(f"Spacing length {len(spacing)} must match dim {self.dim}")
        
        # Validate origin if provided
        if origin is not None and len(origin) != self.dim:
            raise ValueError(f"Origin length {len(origin)} must match dim {self.dim}")
    
    @property
    def num_points(self) -> int:
        """Total number of grid points."""
        result = 1
        for n in self.shape:
            result *= n
        return result
    
    @property
    def cell_volume(self) -> float:
        """Volume of a single cell (for uniform grids)."""
        vol = 1.0
        for dx in self.spacing:
            vol *= dx
        return vol
    
    @property
    def total_volume(self) -> float:
        """Total volume of the computational domain."""
        vol = self.cell_volume
        for n in self.shape:
            vol *= n
        return vol
    
    def index_to_coord(self, idx: Tuple[int, ...]) -> Tuple[float, ...]:
        """Convert grid index to physical coordinate."""
        if len(idx) != self.dim:
            raise ValueError(f"Index length {len(idx)} must match dim {self.dim}")
        
        return tuple(
            self.origin[i] + idx[i] * self.spacing[i]
            for i in range(self.dim)
        )
    
    def coord_to_index(self, coord: Tuple[float, ...]) -> Tuple[int, ...]:
        """Convert physical coordinate to nearest grid index."""
        if len(coord) != self.dim:
            raise ValueError(f"Coordinate length {len(coord)} must match dim {self.dim}")
        
        return tuple(
            int(round((coord[i] - self.origin[i]) / self.spacing[i]))
            for i in range(self.dim)
        )
    
    def coord_to_physical(self, coord: Tuple[float, ...]) -> Tuple[float, ...]:
        """Get exact physical coordinate from normalized coordinate."""
        if len(coord) != self.dim:
            raise ValueError(f"Coordinate length {len(coord)} must match dim {self.dim}")
        
        return tuple(
            self.origin[i] + coord[i] * (self.shape[i] - 1) * self.spacing[i]
            for i in range(self.dim)
        )
    
    def physical_to_coord(self, coord: Tuple[float, ...]) -> Tuple[float, ...]:
        """Convert physical coordinate to normalized [0,1] coordinate."""
        if len(coord) != self.dim:
            raise ValueError(f"Coordinate length {len(coord)} must match dim {self.dim}")
        
        return tuple(
            (coord[i] - self.origin[i]) / ((self.shape[i] - 1) * self.spacing[i])
            if self.shape[i] > 1 else 0.0
            for i in range(self.dim)
        )
    
    def get_staggered_grid(self, direction: int, 
                           location: str = "face") -> 'Grid':
        """
        Get staggered grid for a specific direction.
        
        Args:
            direction: Direction to stagger (0=x, 1=y, 2=z)
            location: "face" for face-centered, "edge" for edge-centered
        
        Returns:
            Staggered grid
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"Direction {direction} out of range for dim {self.dim}")
        
        new_shape = list(self.shape)
        new_origin = list(self.origin)
        new_spacing = list(self.spacing)
        
        if location == "face":
            # Face-centered: one fewer point in staggered direction
            new_shape[direction] = max(1, self.shape[direction] - 1)
            # Origin shifted by half spacing
            new_origin[direction] += self.spacing[direction] / 2
        elif location == "edge":
            # Edge-centered: same number of points
            new_origin[direction] += self.spacing[direction] / 2
        
        return Grid(
            shape=tuple(new_shape),
            spacing=tuple(new_spacing),
            origin=tuple(new_origin),
            boundary_conditions=self.boundary_conditions
        )
    
    def get_component_grid(self, component: int) -> 'Grid':
        """
        Get grid for velocity component (staggered in that direction).
        
        For collocated grids, this is the same grid shifted to faces.
        """
        return self.get_staggered_grid(component, "face")
    
    def compute_metric_factors(self) -> Dict:
        """
        Compute grid metric factors.
        
        Returns dictionary with:
        - jacobian: Jacobian determinant
        - inverse_jacobian: Inverse Jacobian
        - cell_volume: Cell volume
        - jacobian_matrix: Jacobian matrix (for curvilinear grids)
        """
        # For Cartesian grid, metrics are simple
        jacobian = self.cell_volume
        jacobian_matrix = np.diag(self.spacing)
        
        return {
            "jacobian": jacobian,
            "inverse_jacobian": 1.0 / jacobian if jacobian > 0 else float('inf'),
            "cell_volume": jacobian,
            "jacobian_matrix": jacobian_matrix,
            "inverse_jacobian_matrix": np.diag([1.0/dx for dx in self.spacing])
        }
    
    def compute_christoffel_symbols(self) -> np.ndarray:
        """
        Compute Christoffel symbols for this grid.
        
        For Cartesian grids with uniform spacing, Christoffel symbols are zero.
        
        Returns:
            Christoffel symbols Γ^i_jk as 3D array
        """
        dim = self.dim
        Christoffel = np.zeros((dim, dim, dim))
        
        # For uniform Cartesian grid, all Christoffel symbols are zero
        return Christoffel
    
    def boundary_indices(self, face: str) -> List[Tuple[int, ...]]:
        """
        Get indices of points on a specific boundary face.
        
        Args:
            face: Face identifier ("x_min", "x_max", "y_min", "y_max", etc.)
        
        Returns:
            List of grid indices on the boundary
        """
        indices = []
        
        if face == "x_min":
            for idx in np.ndindex(self.shape[1:]):
                indices.append((0,) + idx)
        elif face == "x_max":
            for idx in np.ndindex(self.shape[1:]):
                indices.append((self.shape[0]-1,) + idx)
        elif face == "y_min" and self.dim >= 2:
            for idx in np.ndindex((self.shape[0],) + self.shape[2:]):
                indices.append(idx[:1] + (0,) + idx[1:])
        elif face == "y_max" and self.dim >= 2:
            for idx in np.ndindex((self.shape[0],) + self.shape[2:]):
                indices.append(idx[:1] + (self.shape[1]-1,) + idx[1:])
        elif face == "z_min" and self.dim >= 3:
            for idx in np.ndindex(self.shape[:2]):
                indices.append(idx + (0,))
        elif face == "z_max" and self.dim >= 3:
            for idx in np.ndindex(self.shape[:2]):
                indices.append(idx + (self.shape[2]-1,))
        
        return indices
    
    def interior_indices(self) -> List[Tuple[int, ...]]:
        """
        Get indices of interior points (excluding all boundaries).
        
        Returns:
            List of interior grid indices
        """
        # Create slice for interior
        slices = tuple(
            slice(1, n-1) if n > 2 else slice(0, n)
            for n in self.shape
        )
        
        indices = []
        for idx in np.ndindex(*[s.stop - s.start for s in slices]):
            full_idx = tuple(
                idx[i] + slices[i].start if slices[i].start > 0 else idx[i]
                for i in range(self.dim)
            )
            indices.append(full_idx)
        
        return indices
    
    def is_boundary(self, idx: Tuple[int, ...]) -> bool:
        """Check if index is on a boundary."""
        for i, n in enumerate(self.shape):
            if idx[i] == 0 or idx[i] == n - 1:
                return True
        return False
    
    def get_neighbor(self, idx: Tuple[int, ...], 
                     direction: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get neighbor index with boundary handling.
        
        Args:
            idx: Current index
            direction: Direction offset
        
        Returns:
            Neighbor index (clamped to boundaries)
        """
        neighbor = list(idx)
        for i, d in enumerate(direction):
            neighbor[i] = min(max(0, neighbor[i] + d), self.shape[i] - 1)
        return tuple(neighbor)
    
    def apply_boundary_condition(self, field: np.ndarray,
                                  bc_type: BoundaryConditionType,
                                  bc_value: float = 0.0,
                                  face: str = "all") -> np.ndarray:
        """
        Apply boundary condition to field.
        
        Args:
            field: Field values on grid
            bc_type: Type of boundary condition
            bc_value: Boundary value (for Dirichlet) or derivative (for Neumann)
            face: Which face to apply to ("all" or specific face)
        
        Returns:
            Field with boundary conditions applied
        """
        result = field.copy()
        shape = field.shape
        
        if face == "all" or face == "x_min":
            result[0, ...] = _apply_bc_1d(result[0, ...], bc_type, bc_value, "min")
        if face == "all" or face == "x_max":
            result[-1, ...] = _apply_bc_1d(result[-1, ...], bc_type, bc_value, "max")
        if self.dim >= 2 and (face == "all" or face == "y_min"):
            result[:, 0, ...] = _apply_bc_1d(result[:, 0, ...], bc_type, bc_value, "min")
        if self.dim >= 2 and (face == "all" or face == "y_max"):
            result[:, -1, ...] = _apply_bc_1d(result[:, -1, ...], bc_type, bc_value, "max")
        if self.dim >= 3 and (face == "all" or face == "z_min"):
            result[..., 0] = _apply_bc_1d(result[..., 0], bc_type, bc_value, "min")
        if self.dim >= 3 and (face == "all" or face == "z_max"):
            result[..., -1] = _apply_bc_1d(result[..., -1], bc_type, bc_value, "max")
        
        return result


def _apply_bc_1d(line: np.ndarray, bc_type: BoundaryConditionType,
                 bc_value: float, location: str) -> np.ndarray:
    """Apply boundary condition to 1D array."""
    if bc_type == BoundaryConditionType.DIRICHLET:
        return np.full_like(line, bc_value)
    elif bc_type == BoundaryConditionType.NEUMANN:
        # Derivative condition: use one-sided difference
        if location == "min":
            line[0] = line[1] - bc_value
        else:
            line[-1] = line[-2] + bc_value
        return line
    elif bc_type == BoundaryConditionType.PERIODIC:
        # Periodic: copy opposite end
        if location == "min":
            line[0] = line[-2]
        else:
            line[-1] = line[1]
        return line
    else:
        return line


class UnstructuredGrid:
    """
    Unstructured mesh for FEM discretizations.
    
    Supports arbitrary cell types (triangles, quads, tets, hexes).
    """
    
    def __init__(self, points: np.ndarray, cells: np.ndarray, 
                 cell_type: str = 'tri'):
        """
        Initialize unstructured grid.
        
        Args:
            points: Node coordinates (N x dim array)
            cells: Cell connectivity (M x n_verts array)
            cell_type: Cell type ('tri', 'quad', 'tet', 'hex')
        """
        self.points = points
        self.cells = cells
        self.cell_type = cell_type
        self.dim = points.shape[1]
        self.num_points = points.shape[0]
        self.num_cells = cells.shape[0]
    
    @property
    def n_verts_per_cell(self) -> int:
        """Number of vertices per cell."""
        return self.cells.shape[1]
    
    def cell_volume(self, cell_idx: int) -> float:
        """Compute volume of a single cell."""
        if self.cell_type == 'tri' and self.dim == 2:
            return self._tri_area(cell_idx)
        elif self.cell_type == 'tet' and self.dim == 3:
            return self._tet_volume(cell_idx)
        else:
            return 0.0
    
    def _tri_area(self, cell_idx: int) -> float:
        """Compute area of triangle."""
        verts = self.points[self.cells[cell_idx]]
        v1 = verts[1] - verts[0]
        v2 = verts[2] - verts[0]
        return 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
    
    def _tet_volume(self, cell_idx: int) -> float:
        """Compute volume of tetrahedron."""
        verts = self.points[self.cells[cell_idx]]
        v1 = verts[1] - verts[0]
        v2 = verts[2] - verts[0]
        v3 = verts[3] - verts[0]
        return abs(np.dot(v1, np.cross(v2, v3))) / 6.0
    
    def total_volume(self) -> float:
        """Compute total volume of mesh."""
        return sum(self.cell_volume(i) for i in range(self.num_cells))
    
    def cell_centroid(self, cell_idx: int) -> np.ndarray:
        """Compute centroid of a cell."""
        verts = self.points[self.cells[cell_idx]]
        return verts.mean(axis=0)
    
    def face_connectivity(self) -> List:
        """Build face-to-cell connectivity."""
        # Simplified: return list of faces
        faces = []
        if self.cell_type == 'tri':
            for cell in self.cells:
                faces.append([cell[0], cell[1]])
                faces.append([cell[1], cell[2]])
                faces.append([cell[2], cell[0]])
        return faces
    
    def boundary_faces(self) -> np.ndarray:
        """Identify boundary faces."""
        # Simplified: return faces that appear only once
        all_faces = self.face_connectivity()
        face_count = {}
        
        for face in all_faces:
            key = tuple(sorted(face))
            face_count[key] = face_count.get(key, 0) + 1
        
        boundary = [f for f, count in face_count.items() if count == 1]
        return np.array(boundary)


class Lattice:
    """
    Lattice for lattice gauge theory discretizations.
    
    Provides:
    - Site and link indexing
    - Gauge group operations
    - Parallel transport
    """
    
    def __init__(self, shape: Tuple[int, ...], 
                 lattice_type: LatticeType = LatticeType.CUBIC,
                 group: str = "SU(2)"):
        """
        Initialize lattice.
        
        Args:
            shape: Number of sites in each dimension
            lattice_type: Type of lattice
            group: Gauge group ("SU(2)", "SU(3)", "U(1)")
        """
        self.shape = shape
        self.lattice_type = lattice_type
        self.group = group
        self.dim = len(shape)
        
        # Build basis vectors
        self.basis = self._build_basis()
        
        # Number of links per site
        self.link_count = 2 * self.dim
    
    def _build_basis(self) -> np.ndarray:
        """Build lattice basis vectors."""
        basis = np.eye(self.dim)
        if self.lattice_type == LatticeType.CUBIC:
            return basis
        elif self.lattice_type == LatticeType.BODY_CENTERED_CUBIC:
            # BCC has additional point at body center
            return np.vstack([np.eye(self.dim), 0.5 * np.ones(self.dim)])
        else:
            return basis
    
    @property
    def num_sites(self) -> int:
        """Total number of sites."""
        result = 1
        for n in self.shape:
            result *= n
        return result
    
    @property
    def num_links(self) -> int:
        """Total number of directed links."""
        return self.num_sites * self.dim
    
    def site_index(self, coords: Tuple[int, ...]) -> int:
        """Convert site coordinates to linear index."""
        strides = tuple(
            int(np.prod(self.shape[i+1:])) 
            for i in range(self.dim)
        )
        return sum(coords[i] * strides[i] for i in range(self.dim))
    
    def linear_to_coords(self, idx: int) -> Tuple[int, ...]:
        """Convert linear index to site coordinates."""
        coords = []
        for i in range(self.dim - 1, -1, -1):
            stride = int(np.prod(self.shape[i+1:])) if i < self.dim - 1 else 1
            coords.insert(0, idx // stride)
            idx %= stride
        return tuple(coords)
    
    def link_index(self, site: Tuple[int, ...], direction: int) -> int:
        """Get link index from site and direction."""
        site_idx = self.site_index(site)
        return site_idx * self.dim + direction
    
    def neighbor(self, site: Tuple[int, ...], direction: int,
                 sign: int = 1) -> Tuple[int, ...]:
        """
        Get neighboring site.
        
        Args:
            site: Current site coordinates
            direction: Direction (0=x, 1=y, 2=z)
            sign: +1 for positive, -1 for negative direction
        
        Returns:
            Neighbor site coordinates
        """
        neighbor = list(site)
        neighbor[direction] += sign
        
        # Apply periodic boundary conditions
        neighbor[direction] = neighbor[direction] % self.shape[direction]
        
        return tuple(neighbor)
    
    def plaquette(self, site: Tuple[int, ...], 
                  plane: Tuple[int, int]) -> List:
        """
        Get links forming a plaquette in a given plane.
        
        Args:
            site: Origin site
            plane: Two directions defining the plane
        
        Returns:
            List of link indices forming the plaquette
        """
        mu, nu = plane
        
        links = [
            self.link_index(site, mu),
            self.link_index(self.neighbor(site, mu), nu),
            self.link_index(self.neighbor(site, nu), mu),
            self.link_index(site, nu)
        ]
        
        return links


def create_uniform_grid(shape: Tuple[int, ...],
                        spacing: float = 1.0,
                        origin: Tuple[float, ...] = None) -> Grid:
    """
    Create uniform grid with same spacing in all directions.
    
    Args:
        shape: Number of points in each dimension
        spacing: Uniform spacing
        origin: Origin coordinates
    
    Returns:
        Uniform Grid instance
    """
    dim = len(shape)
    spacing_tuple = tuple(spacing for _ in range(dim))
    
    if origin is None:
        origin = tuple(0.0 for _ in range(dim))
    
    return Grid(shape, spacing_tuple, origin)


def create_staggered_grid(base_grid: Grid, 
                          variables: List[str]) -> Dict[str, Grid]:
    """
    Create staggered grids for incompressible flow variables.
    
    Args:
        base_grid: Base collocated grid
        variables: List of variables ("u", "v", "w", "p")
    
    Returns:
        Dictionary mapping variable names to their grids
    """
    grids = {}
    
    # Cell centers (collocated)
    grids["p"] = base_grid
    
    # Face centers for velocities
    for i, var in enumerate(["u", "v", "w"][:base_grid.dim]):
        grids[var] = base_grid.get_component_grid(i)
    
    return grids
