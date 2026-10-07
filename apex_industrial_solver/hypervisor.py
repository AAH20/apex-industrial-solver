"""Apex Industrial Solver Unified Hypervisor.

Unified orchestrator facade delivering sub-millisecond NP-hard optimization across
3D bin packing, fleet routing (VRPTW), job shop scheduling (JSSP), facility siting,
and mixed-integer linear programming (MILP).
Zero external dependencies: 100% pure Python standard library.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from apex_industrial_solver.core.bin_packing_3d import BinPacking3DSolver
from apex_industrial_solver.core.branch_and_bound import BranchAndBoundSolver, MILPResult
from apex_industrial_solver.core.facility_location import FacilityLocationSolver
from apex_industrial_solver.core.jssp import FlexibleJobShopSolver
from apex_industrial_solver.core.models import (
    BinPacking3DResult,
    Box3D,
    Container3D,
    FacilityLocationResult,
    FleetStop,
    JSSPResult,
    TaskOperation,
    VehicleRoute,
    VRPTWResult,
)
from apex_industrial_solver.core.simplex import LPResult, SimplexSolver
from apex_industrial_solver.core.vrptw import VRPTWSolver


class IndustrialSolverHypervisor:
    """Master orchestrator unifying discrete and continuous combinatorial solvers."""

    def __init__(self):
        self.bin_packer = BinPacking3DSolver()
        self.vrp_solver = VRPTWSolver()
        self.jssp_solver = FlexibleJobShopSolver()
        self.cflp_solver = FacilityLocationSolver()
        self.simplex_solver = SimplexSolver()
        self.milp_solver = BranchAndBoundSolver()

    def pack_3d(
        self,
        container: Container3D,
        items: List[Box3D],
        sort_strategy: str = "volume_desc",
    ) -> BinPacking3DResult:
        """Execute 3D container cargo packing with CoG stability."""
        return self.bin_packer.pack(container, items, sort_strategy=sort_strategy)

    def route_fleet(
        self,
        depot: FleetStop,
        stops: List[FleetStop],
        vehicle_capacities: List[float],
        vehicle_max_durations: Optional[List[float]] = None,
        speed: float = 1.0,
    ) -> VRPTWResult:
        """Execute Capacitated Vehicle Routing Problem with Time Windows."""
        return self.vrp_solver.solve(
            depot=depot,
            stops=stops,
            vehicle_capacities=vehicle_capacities,
            vehicle_max_durations=vehicle_max_durations,
            speed=speed,
        )

    def schedule_jobs(
        self,
        operations: List[TaskOperation],
        all_machines: Optional[List[str]] = None,
    ) -> JSSPResult:
        """Execute Flexible Job Shop Scheduling with precedence graphs."""
        return self.jssp_solver.solve(operations=operations, all_machines=all_machines)

    def locate_facilities(
        self,
        facilities: Dict[str, Dict[str, float]],
        customers: Dict[str, Dict[str, float]],
        transport_cost_per_unit_distance: float = 1.0,
    ) -> FacilityLocationResult:
        """Execute Capacitated Facility Location siting and customer allocation."""
        return self.cflp_solver.solve(
            facilities=facilities,
            customers=customers,
            transport_cost_per_unit_distance=transport_cost_per_unit_distance,
        )

    def solve_lp(
        self,
        c: List[float],
        A_ub: Optional[List[List[float]]] = None,
        b_ub: Optional[List[float]] = None,
        A_eq: Optional[List[List[float]]] = None,
        b_eq: Optional[List[float]] = None,
        maximize: bool = False,
    ) -> LPResult:
        """Execute continuous Two-Phase Linear Programming."""
        return self.simplex_solver.solve(
            c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, maximize=maximize
        )

    def solve_milp(
        self,
        c: List[float],
        A_ub: Optional[List[List[float]]] = None,
        b_ub: Optional[List[float]] = None,
        integer_indices: Optional[List[int]] = None,
        maximize: bool = False,
    ) -> MILPResult:
        """Execute Mixed-Integer Linear Programming via Branch-and-Bound."""
        return self.milp_solver.solve(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            integer_indices=integer_indices,
            maximize=maximize,
        )
