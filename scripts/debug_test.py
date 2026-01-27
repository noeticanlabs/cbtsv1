#!/usr/bin/env python3
from src.core.gr_stepper import GRStepper
from src.core.gr_core_fields import GRCoreFields
from src.core.gr_geometry import GRGeometry
from src.core.gr_constraints import GRConstraints
from src.core.gr_gauge import GRGauge
import numpy as np

# Create a simple 4x4x4 grid
Nx, Ny, Nz = 4, 4, 4
dx, dy, dz = 1.0, 1.0, 1.0
fields = GRCoreFields(Nx, Ny, Nz, dx, dy, dz)
fields.init_minkowski()

geometry = GRGeometry(fields)
constraints = GRConstraints(fields, geometry)
gauge = GRGauge(fields, geometry)

stepper = GRStepper(fields, geometry, constraints, gauge)

print("Stepper created")

# Call compute_rhs
stepper.rhs_computer.compute_rhs(0.0, slow_update=False)

print("compute_rhs called successfully")
print("rhs_gamma_sym6 shape:", stepper.rhs_computer.rhs_gamma_sym6.shape)
print("rhs_gamma_sym6 sum:", np.sum(stepper.rhs_computer.rhs_gamma_sym6))