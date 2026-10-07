"""Core mathematical and operations research domain models for Apex Industrial Solver.

Zero external dependencies: 100% pure Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SolverStatus(str, Enum):
    """Optimization outcome status."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"


# ==============================================================================
# 1. 3D Container Bin Packing Models
# ==============================================================================

@dataclass
class Box3D:
    """A 3D cuboid cargo item to be packed into a container."""
    item_id: str
    width: float   # X-dimension
    length: float  # Y-dimension
    height: float  # Z-dimension
    weight: float
    can_rotate: bool = True
    max_stack_weight: float = float("inf")
    # Placement results populated by solver
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    placed_width: float = 0.0
    placed_length: float = 0.0
    placed_height: float = 0.0
    is_packed: bool = False

    @property
    def volume(self) -> float:
        return self.width * self.length * self.height


@dataclass
class Container3D:
    """A 3D physical container or vehicle cargo bed."""
    container_id: str
    width: float
    length: float
    height: float
    max_weight: float
    # Center of Gravity support envelope bounds [min_ratio, max_ratio]
    cog_x_bounds: Tuple[float, float] = (0.40, 0.60)
    cog_y_bounds: Tuple[float, float] = (0.40, 0.60)

    @property
    def volume(self) -> float:
        return self.width * self.length * self.height


@dataclass
class BinPacking3DResult:
    """Results of a 3D container bin packing optimization."""
    status: SolverStatus
    packed_items: List[Box3D]
    unpacked_items: List[Box3D]
    volume_utilization: float
    weight_utilization: float
    total_packed_weight: float
    center_of_gravity: Tuple[float, float, float]
    cog_is_stable: bool
    elapsed_microseconds: float


# ==============================================================================
# 2. Capacitated Vehicle Routing Problem with Time Windows (VRPTW)
# ==============================================================================

@dataclass
class FleetStop:
    """A customer delivery or pickup stop in VRPTW."""
    stop_id: str
    x: float
    y: float
    demand: float
    ready_time: float   # e_i
    due_time: float     # l_i
    service_time: float # s_i


@dataclass
class VehicleRoute:
    """An optimized delivery route for a single vehicle."""
    vehicle_id: str
    stops: List[FleetStop]
    total_distance: float
    total_travel_time: float
    total_demand: float
    arrival_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class VRPTWResult:
    """Result of fleet routing optimization."""
    status: SolverStatus
    routes: List[VehicleRoute]
    total_distance: float
    total_duration: float
    vehicles_used: int
    unassigned_stops: List[FleetStop]
    elapsed_microseconds: float


# ==============================================================================
# 3. Flexible Job Shop Scheduling (JSSP)
# ==============================================================================

@dataclass
class TaskOperation:
    """An operation within a manufacturing job."""
    task_id: str
    job_id: str
    sequence_index: int
    # Machine options: mapping of machine_id -> processing_duration
    machine_durations: Dict[str, float]


@dataclass
class ScheduledTask:
    """A task assigned to a specific machine with start and completion times."""
    task_id: str
    job_id: str
    machine_id: str
    start_time: float
    completion_time: float


@dataclass
class JSSPResult:
    """Result of manufacturing job shop scheduling."""
    status: SolverStatus
    makespan: float  # C_max
    scheduled_tasks: List[ScheduledTask]
    machine_schedules: Dict[str, List[ScheduledTask]]
    average_machine_utilization: float
    elapsed_microseconds: float


# ==============================================================================
# 4. Capacitated Facility Location Problem (CFLP)
# ==============================================================================

@dataclass
class FacilityLocationResult:
    """Result of facility siting optimization."""
    status: SolverStatus
    opened_facility_ids: List[str]
    demand_assignments: Dict[str, str]  # customer_id -> facility_id
    total_cost: float
    fixed_capital_cost: float
    transportation_cost: float
    elapsed_microseconds: float
