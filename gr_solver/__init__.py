# gr_solver - Backward Compatibility Package
# This package provides backward compatibility for imports from the old gr_solver/ directory
# All modules are re-exported from src/core and other src/ subdirectories

# Core GR solver modules from src/core/
from src.core.gr_solver import GRSolver
from src.core.gr_stepper import GRStepper
from src.core.gr_constraints import GRConstraints
from src.core.gr_geometry import GRGeometry
from src.core.gr_geometry_nsc import compute_christoffels_compiled
from src.core.gr_gauge import GRGauge
from src.core.gr_scheduler import GRScheduler
from src.core.gr_ledger import GRLedger
from src.core.gr_clock import UnifiedClock, UnifiedClockState
from src.core.gr_clocks import MultiRateClockSystem, BandClockConfig
from src.core.gr_core_fields import (
    GRCoreFields, SYM6_IDX, aligned_zeros, 
    sym6_to_mat33, mat33_to_sym6, det_sym6, inv_sym6,
    trace_sym6, norm2_sym6, eigenvalues_sym6, cond_sym6,
    repair_spd_eigen_clamp, symmetry_error
)
from src.core.gr_rhs import GRRhs
from src.core.gr_loc import GRLoC
from src.core.gr_sem import SEMDomain
from src.core.gr_gates import GateChecker, GateKind, should_hard_fail
from src.core.gr_ttl_calculator import TTLCalculator, AdaptiveTTLs, compute_adaptive_ttls
from src.core.gr_coherence import CoherenceOperator
from src.phaseloom.phaseloom_gr_orchestrator import GRPhaseLoomOrchestrator
from src.core.gr_receipts import ReceiptEmitter
from src.core.gr_loc import GRLoC

# Re-export moved modules for backward compatibility
# These modules were moved to src/phaseloom/, src/spectral/, src/elliptic/
try:
    from src.phaseloom.phaseloom_memory import PhaseLoomMemory
    from src.phaseloom.phaseloom_rails_gr import GRPhaseLoomRails
    from src.phaseloom.phaseloom_octaves import PhaseLoomOctaves
    from src.phaseloom.phaseloom_gr_adapter import PhaseLoomGRAdapter
    from src.phaseloom.phaseloom_gr_controller import PhaseLoomGRController
    from src.phaseloom.phaseloom_receipts_gr import PhaseLoomReceiptsGR
    from src.phaseloom.phaseloom_render_gr import PhaseLoomRenderGR
    from src.phaseloom.phaseloom_threads_gr import PhaseLoomThreadsGR
except ImportError as e:
    # Log warning if phaseloom modules not available
    import warnings
    warnings.warn(f"Could not import phaseloom modules: {e}")

try:
    from src.spectral.cache import SpectralCache, _phi1, _phi2
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import spectral module: {e}")

try:
    from src.elliptic.solver import EllipticSolver, apply_poisson
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import elliptic module: {e}")

# Host API
try:
    from src.host_api import GRHostAPI
except ImportError:
    # Fallback for legacy location
    from gr_solver.host_api import GRHostAPI

# AEONIC modules (if available)
try:
    from src.core.aeonic_clocks import AeonicClockPack
    from src.core.aeonic_memory_bank import AeonicMemoryBank
    from src.core.aeonic_memory_contract import AeonicMemoryContract
    from src.core.aeonic_receipts import AeonicReceipts
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import aeonic modules: {e}")

# NIR/LLCC modules (if available)
try:
    from src.nllc.ast import NIRNode, NIRProgram
    from src.nllc.nir import NIR
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import nllc modules: {e}")

__all__ = [
    # Core modules
    'GRSolver',
    'GRStepper',
    'GRConstraints',
    'GRGeometry',
    'GRGeometryNSC',
    'compute_christoffels_compiled',
    'GRGauge',
    'GRScheduler',
    'GRLedger',
    'GRClock',
    'UnifiedClockState',
    'MultiRateClockSystem',
    'BandClockConfig',
    'GRCoreFields',
    'SYM6_IDX',
    'aligned_zeros',
    'sym6_to_mat33',
    'mat33_to_sym6',
    'det_sym6',
    'inv_sym6',
    'trace_sym6',
    'norm2_sym6',
    'eigenvalues_sym6',
    'cond_sym6',
    'repair_spd_eigen_clamp',
    'symmetry_error',
    'GRRhs',
    'GRLoC',
    'GRSEM',
    'GateChecker',
    'GateKind',
    'should_hard_fail',
    'TTLCalculator',
    'AdaptiveTTLs',
    'compute_adaptive_ttls',
    'CoherenceOperator',
    'GRPhaseLoomOrchestrator',
    'Phases',
    'hpc_kernels',
    'GRReceipts',
    
    # Phaseloom modules
    'PhaseLoomMemory',
    'GRPhaseLoomRails',
    'PhaseLoomOctaves',
    'PhaseLoomGRAdapter',
    'PhaseLoomGRController',
    'PhaseLoomReceiptsGR',
    'PhaseLoomRenderGR',
    'PhaseLoomThreadsGR',
    
    # Spectral modules
    'SpectralCache',
    '_phi1',
    '_phi2',
    
    # Elliptic modules
    'EllipticSolver',
    'apply_poisson',
    
    # Host API
    'GRHostAPI',
    
    # AEONIC modules
    'AeonicClockPack',
    'AeonicMemoryBank',
    'AeonicMemoryContract',
    'AeonicReceipts',
    
    # NIR/LLCC modules
    'NIRNode',
    'NIRProgram',
    'NIR',
]
