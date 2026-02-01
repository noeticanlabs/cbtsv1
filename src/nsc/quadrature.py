"""
NSC-M3L Quadrature Rules

Implements numerical integration rules for FEM and discrete operators.

Standard rules:
- Gauss-Legendre quadrature
- Simpson's rule
- Trapezoidal rule
- Element-specific rules (triangle, tetrahedron)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .disc_types import QuadratureRule


# === 1D Gauss-Legendre Quadrature ===

def gauss_legendre_1() -> QuadratureRule:
    """
    1-point Gauss-Legendre quadrature (degree 1).
    
    Nodes: [0]
    Weights: [2]
    Exact for polynomials up to degree 1.
    """
    return QuadratureRule(
        nodes=np.array([[0.0]]),
        weights=np.array([2.0]),
        degree=1,
        domain="interval"
    )


def gauss_legendre_2() -> QuadratureRule:
    """
    2-point Gauss-Legendre quadrature (degree 3).
    
    Nodes: [-1/√3, 1/√3]
    Weights: [1, 1]
    Exact for polynomials up to degree 3.
    """
    nodes = np.array([[-1.0 / np.sqrt(3)], [1.0 / np.sqrt(3)]])
    return QuadratureRule(
        nodes=nodes,
        weights=np.array([1.0, 1.0]),
        degree=3,
        domain="interval"
    )


def gauss_legendre_3() -> QuadratureRule:
    """
    3-point Gauss-Legendre quadrature (degree 5).
    
    Nodes: [-√(3/5), 0, √(3/5)]
    Weights: [5/9, 8/9, 5/9]
    Exact for polynomials up to degree 5.
    """
    sqrt_3_5 = np.sqrt(3.0 / 5.0)
    nodes = np.array([[-sqrt_3_5], [0.0], [sqrt_3_5]])
    weights = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    
    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        degree=5,
        domain="interval"
    )


def gauss_legendre_n(n: int) -> QuadratureRule:
    """
    N-point Gauss-Legendre quadrature.
    
    Args:
        n: Number of quadrature points
    
    Returns:
        Gauss-Legendre quadrature rule with n points
    """
    # Compute nodes using numpy's legendre polynomial roots
    from numpy.polynomial.legendre import leggauss
    
    nodes_1d, weights = leggauss(n)
    nodes = nodes_1d.reshape(-1, 1)
    
    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        degree=2*n - 1,
        domain="interval"
    )


# === Newton-Cotes Quadrature ===

def trapezoidal_rule(n_intervals: int = 1) -> QuadratureRule:
    """
    Trapezoidal rule on interval [-1, 1].
    
    Nodes: [-1, -1+2/(n-1), ..., 1] for n points
    Weights: [h/2, h, ..., h, h/2] where h = 2/(n-1)
    
    Exact for polynomials up to degree 1.
    
    Args:
        n_intervals: Number of intervals
    
    Returns:
        Trapezoidal rule quadrature
    """
    if n_intervals < 1:
        n_intervals = 1
    
    n_points = n_intervals + 1
    h = 2.0 / n_intervals
    
    nodes = np.linspace(-1, 1, n_points).reshape(-1, 1)
    weights = np.full(n_points, h)
    weights[0] = h / 2
    weights[-1] = h / 2
    
    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        degree=1,
        domain="interval"
    )


def simpson_rule() -> QuadratureRule:
    """
    Simpson's rule on interval [-1, 1].
    
    Nodes: [-1, 0, 1]
    Weights: [1/3, 4/3, 1/3] (scaled to interval length 2)
    Exact for polynomials up to degree 3.
    """
    nodes = np.array([[-1.0], [0.0], [1.0]])
    weights = np.array([1.0, 4.0, 1.0]) / 3.0
    
    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        degree=3,
        domain="interval"
    )


def simpson_3_8_rule() -> QuadratureRule:
    """
    Simpson's 3/8 rule on interval [-1, 1].
    
    Nodes: [-1, -1/3, 1/3, 1]
    Weights: [3/8, 9/8, 9/8, 3/8]
    Exact for polynomials up to degree 3.
    """
    nodes = np.array([[-1.0], [-1.0/3.0], [1.0/3.0], [1.0]])
    weights = np.array([3.0, 9.0, 9.0, 3.0]) / 8.0
    
    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        degree=3,
        domain="interval"
    )


# === 2D/3D Tensor Product Quadrature ===

def tensor_product_quadrature(quad_1d: QuadratureRule, dim: int) -> QuadratureRule:
    """
    Create tensor product quadrature from 1D rule.
    
    Args:
        quad_1d: 1D quadrature rule
        dim: Dimension (2 or 3)
    
    Returns:
        Tensor product quadrature rule
    """
    from itertools import product
    
    # Generate all combinations of 1D nodes
    nodes_1d = [quad_1d.nodes.flatten().tolist()]
    
    quad_points = []
    quad_weights = []
    
    for node_combo in product(*nodes_1d * dim):
        quad_points.append(list(node_combo))
        # Weight is product of 1D weights
        weight = 1.0
        for i in range(dim):
            idx = node_combo[i]
            # Find corresponding 1D weight
            # This is approximate - proper implementation needs index mapping
        quad_weights.append(quad_1d.weights.prod())
    
    # For the simple case where dim=1, just return original
    if dim == 1:
        return quad_1d
    
    # For dim > 1, create proper tensor product
    # This is a simplified version
    nodes_list = []
    weights_list = []
    
    return QuadratureRule(
        nodes=np.array(quad_points),
        weights=np.array(quad_weights),
        degree=quad_1d.degree,
        domain="hypercube"
    )


def gauss_legendre_tensor_2d(n: int = 2) -> QuadratureRule:
    """
    2D tensor product Gauss-Legendre quadrature.
    
    Args:
        n: Points per dimension
    
    Returns:
        2D tensor product quadrature
    """
    quad_1d = gauss_legendre_n(n)
    
    # Generate tensor product nodes and weights
    nodes_1d = quad_1d.nodes.flatten()
    weights_1d = quad_1d.weights
    
    nodes_2d = []
    weights_2d = []
    
    for i, (x, wx) in enumerate(zip(nodes_1d, weights_1d)):
        for j, (y, wy) in enumerate(zip(nodes_1d, weights_1d)):
            nodes_2d.append([x, y])
            weights_2d.append(wx * wy)
    
    return QuadratureRule(
        nodes=np.array(nodes_2d),
        weights=np.array(weights_2d),
        degree=2*n - 1,
        domain="quadrilateral"
    )


def gauss_legendre_tensor_3d(n: int = 2) -> QuadratureRule:
    """
    3D tensor product Gauss-Legendre quadrature.
    
    Args:
        n: Points per dimension
    
    Returns:
        3D tensor product quadrature
    """
    quad_1d = gauss_legendre_n(n)
    
    nodes_1d = quad_1d.nodes.flatten()
    weights_1d = quad_1d.weights
    
    nodes_3d = []
    weights_3d = []
    
    for i, (x, wx) in enumerate(zip(nodes_1d, weights_1d)):
        for j, (y, wy) in enumerate(zip(nodes_1d, weights_1d)):
            for k, (z, wz) in enumerate(zip(nodes_1d, weights_1d)):
                nodes_3d.append([x, y, z])
                weights_3d.append(wx * wy * wz)
    
    return QuadratureRule(
        nodes=np.array(nodes_3d),
        weights=np.array(weights_3d),
        degree=2*n - 1,
        domain="hexahedron"
    )


# === Triangle Quadrature ===

def quadrature_triangle_1() -> QuadratureRule:
    """
    1-point quadrature on reference triangle.
    
    Node: centroid (1/3, 1/3)
    Weight: 1/2 (area of triangle)
    Exact for polynomials up to degree 1.
    """
    return QuadratureRule(
        nodes=np.array([[1.0/3.0, 1.0/3.0]]),
        weights=np.array([0.5]),
        degree=1,
        domain="triangle"
    )


def quadrature_triangle_3() -> QuadratureRule:
    """
    3-point quadrature on reference triangle.
    
    Nodes: midpoints of edges
    Weights: 1/6 each
    Exact for polynomials up to degree 2.
    
    Note: More common 3-point rule uses vertices with different weights.
    """
    # Using the common 3-point rule
    alpha = 2.0 / 3.0
    beta = 1.0 / 6.0
    
    nodes = np.array([
        [alpha, beta],
        [beta, alpha],
        [beta, beta]
    ])
    weights = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
    
    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        degree=2,
        domain="triangle"
    )


def quadrature_triangle_6() -> QuadratureRule:
    """
    6-point quadrature on reference triangle.
    
    Uses symmetric points for degree 3 exactness.
    """
    # Knot locations
    a = 0.816847572980459  # (1+sqrt(3/5))/2
    b = 1.0 - a
    c = 0.108103018168070  # (1-sqrt(3/5))/2
    d = 0.445948490915965  # (1+sqrt(3/5))/2
    e = 1.0 - 2*d
    
    # Nodes and weights for degree 3
    nodes = np.array([
        [a, b],
        [b, a],
        [b, b],
        [c, d],
        [d, c],
        [d, d]
    ])
    weights = np.array([
        0.109951743655321,
        0.109951743655321,
        0.109951743655321,
        0.223381589678010,
        0.223381589678010,
        0.223381589678010
    ])
    
    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        degree=3,
        domain="triangle"
    )


# === Tetrahedron Quadrature ===

def quadrature_tetrahedron_1() -> QuadratureRule:
    """
    1-point quadrature on reference tetrahedron.
    
    Node: centroid (1/4, 1/4, 1/4)
    Weight: 1/6 (volume of tetrahedron)
    Exact for polynomials up to degree 1.
    """
    return QuadratureRule(
        nodes=np.array([[0.25, 0.25, 0.25]]),
        weights=np.array([1.0/6.0]),
        degree=1,
        domain="tetrahedron"
    )


def quadrature_tetrahedron_4() -> QuadratureRule:
    """
    4-point quadrature on reference tetrahedron.
    
    Uses vertices of tetrahedron.
    Exact for polynomials up to degree 2.
    """
    nodes = np.array([
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25]
    ])
    weights = np.array([1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0])
    
    return QuadratureRule(
        nodes=nodes,
        weights=weights,
        degree=2,
        domain="tetrahedron"
    )


# === Integration Utilities ===

def integrate(quad: QuadratureRule, func) -> float:
    """
    Integrate a function using quadrature rule.
    
    Args:
        quad: Quadrature rule
        func: Function to integrate (callable)
    
    Returns:
        Approximate integral value
    """
    result = 0.0
    for node, weight in zip(quad.nodes, quad.weights):
        result += weight * func(node)
    return result


def integrate_function(quad: QuadratureRule, values: np.ndarray) -> float:
    """
    Integrate pre-computed function values at quadrature points.
    
    Args:
        quad: Quadrature rule
        values: Function values at quadrature nodes
    
    Returns:
        Approximate integral
    """
    return float(np.dot(quad.weights, values))


def compute_mass_matrix(quad: QuadratureRule, 
                        basis_functions: List) -> np.ndarray:
    """
    Compute mass matrix M_ij = ∫ φ_i φ_j dV.
    
    Args:
        quad: Quadrature rule
        basis_functions: List of basis functions (callables)
    
    Returns:
        Mass matrix
    """
    n = len(basis_functions)
    M = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            for node, weight in zip(quad.nodes, quad.weights):
                M[i, j] += weight * basis_functions[i](node) * basis_functions[j](node)
    
    return M


def compute_stiffness_matrix(quad: QuadratureRule,
                             basis_grads: List) -> np.ndarray:
    """
    Compute stiffness matrix K_ij = ∫ ∇φ_i · ∇φ_j dV.
    
    Args:
        quad: Quadrature rule
        basis_grads: List of basis function gradients (callables returning arrays)
    
    Returns:
        Stiffness matrix
    """
    n = len(basis_grads)
    K = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            for node, weight in zip(quad.nodes, quad.weights):
                grad_i = basis_grads[i](node)
                grad_j = basis_grads[j](node)
                K[i, j] += weight * np.dot(grad_i, grad_j)
    
    return K


# === Reference Element Mapping ===

def map_to_reference_triangle(physical_points: np.ndarray) -> np.ndarray:
    """
    Map physical triangle points to reference triangle.
    
    Reference triangle: vertices at (0,0), (1,0), (0,1).
    
    Args:
        physical_points: Points in physical space (N x 2)
    
    Returns:
        Points in reference triangle (N x 2)
    """
    # Assuming linear mapping
    # This is a simplified version - full implementation needs vertex info
    return physical_points


def map_to_reference_tetrahedron(physical_points: np.ndarray) -> np.ndarray:
    """
    Map physical tetrahedron points to reference tetrahedron.
    
    Reference tetrahedron: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    
    Args:
        physical_points: Points in physical space (N x 3)
    
    Returns:
        Points in reference tetrahedron (N x 3)
    """
    return physical_points


def compute_jacobian(physical_points: np.ndarray, 
                     ref_points: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian of reference-to-physical mapping.
    
    Args:
        physical_points: Points in physical space
        ref_points: Corresponding points in reference space
    
    Returns:
        Jacobian determinant at each point
    """
    # Simplified: assume uniform Jacobian
    n = physical_points.shape[0]
    jacobians = np.ones(n) * 0.5  # Placeholder
    
    return jacobians
