"""Apex Industrial Solver: Zero-Dependency Industrial Operations Research Kernels.

Pure Python 3.10+ standard library.
"""

from apex_industrial_solver.core.bin_packing_3d import BinPacking3DSolver
from apex_industrial_solver.core.branch_and_bound import (
    AdaptiveLargeNeighborhoodSearch,
    BranchAndBoundSolver,
    MILPResult,
)
from apex_industrial_solver.core.facility_location import FacilityLocationSolver
from apex_industrial_solver.core.jssp import FlexibleJobShopSolver
from apex_industrial_solver.core.models import (
    BinPacking3DResult,
    Box3D,
    Container3D,
    FacilityLocationResult,
    FleetStop,
    JSSPResult,
    ScheduledTask,
    SolverStatus,
    TaskOperation,
    VehicleRoute,
    VRPTWResult,
)
from apex_industrial_solver.core.simplex import LPResult, SimplexSolver
from apex_industrial_solver.core.vrptw import VRPTWSolver
from apex_industrial_solver.hypervisor import IndustrialSolverHypervisor

__version__ = "0.1.0"

__all__ = [
    "AdaptiveLargeNeighborhoodSearch",
    "BinPacking3DResult",
    "BinPacking3DSolver",
    "Box3D",
    "BranchAndBoundSolver",
    "Container3D",
    "FacilityLocationResult",
    "FacilityLocationSolver",
    "FleetStop",
    "FlexibleJobShopSolver",
    "IndustrialSolverHypervisor",
    "JSSPResult",
    "LPResult",
    "MILPResult",
    "ScheduledTask",
    "SimplexSolver",
    "SolverStatus",
    "TaskOperation",
    "VehicleRoute",
    "VRPTWResult",
    "VRPTWSolver",
    "__version__",
]
