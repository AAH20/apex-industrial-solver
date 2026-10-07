"""Capacitated Facility Location Problem (CFLP) Engine.

Optimizes facility siting, warehouse location, and customer allocation
under fixed opening costs, capacity limits, and transportation distances.
Zero external dependencies: 100% pure Python standard library.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set, Tuple
from apex_industrial_solver.core.models import (
    FacilityLocationResult,
    SolverStatus,
)


class FacilityLocationSolver:
    """Solves Capacitated and Uncapacitated Facility Location Problems (CFLP/UFLP).
    
    Uses an iterative ADD-DROP-SWAP neighborhood search combined with
    greedy capacity-constrained minimum-cost assignment.
    """

    def __init__(self, max_iterations: int = 500, time_limit_seconds: float = 2.0):
        self.max_iterations = max_iterations
        self.time_limit = time_limit_seconds

    def solve(
        self,
        facilities: Dict[str, Dict[str, float]],
        customers: Dict[str, Dict[str, float]],
        transport_cost_per_unit_distance: float = 1.0,
    ) -> FacilityLocationResult:
        """Solve facility location problem.
        
        Args:
            facilities: Dict mapping facility_id -> {
                'x': float,
                'y': float,
                'fixed_cost': float,
                'capacity': float (inf for uncapacitated)
            }
            customers: Dict mapping customer_id -> {
                'x': float,
                'y': float,
                'demand': float
            }
            transport_cost_per_unit_distance: Multiplier for Euclidean distance * demand.
            
        Returns:
            FacilityLocationResult with opened facilities and customer assignments.
        """
        start_time = time.perf_counter()

        if not facilities or not customers:
            return FacilityLocationResult(
                status=SolverStatus.FEASIBLE,
                opened_facility_ids=[],
                demand_assignments={},
                total_cost=0.0,
                fixed_capital_cost=0.0,
                transportation_cost=0.0,
                elapsed_microseconds=0.0,
            )

        total_demand = sum(c.get("demand", 1.0) for c in customers.values())
        total_capacity = sum(f.get("capacity", float("inf")) for f in facilities.values())

        if total_capacity < total_demand:
            elapsed_us = (time.perf_counter() - start_time) * 1_000_000.0
            return FacilityLocationResult(
                status=SolverStatus.INFEASIBLE,
                opened_facility_ids=[],
                demand_assignments={},
                total_cost=float("inf"),
                fixed_capital_cost=0.0,
                transportation_cost=0.0,
                elapsed_microseconds=round(elapsed_us, 2),
            )

        # Precompute Euclidean transportation cost matrix
        # cost_matrix[facility_id][customer_id]
        cost_matrix: Dict[str, Dict[str, float]] = {}
        for fid, f in facilities.items():
            cost_matrix[fid] = {}
            for cid, c in customers.items():
                dx = f["x"] - c["x"]
                dy = f["y"] - c["y"]
                dist = math.hypot(dx, dy)
                demand = c.get("demand", 1.0)
                cost_matrix[fid][cid] = dist * demand * transport_cost_per_unit_distance

        facility_ids = list(facilities.keys())

        # Phase 1: Greedy Constructive ADD heuristic
        # Start with facility that provides lowest cost if opened alone, then add greedy best
        best_open: Set[str] = set()
        best_cost = float("inf")
        best_assignments: Dict[str, str] = {}
        best_fixed_cost = 0.0
        best_trans_cost = 0.0

        # Try opening each facility individually as a seed
        for seed_fid in facility_ids:
            cur_open = {seed_fid}
            # Check capacity sufficiency; if insufficient, add next closest until sufficient
            cur_cap = facilities[seed_fid].get("capacity", float("inf"))
            while cur_cap < total_demand and len(cur_open) < len(facility_ids):
                # Add facility with largest remaining capacity or lowest fixed cost
                remaining = [f for f in facility_ids if f not in cur_open]
                best_add = min(
                    remaining,
                    key=lambda f: facilities[f]["fixed_cost"] / max(1.0, facilities[f].get("capacity", 1.0)),
                )
                cur_open.add(best_add)
                cur_cap += facilities[best_add].get("capacity", float("inf"))

            assigned, trans_cost, feasible = self._evaluate_open_set(cur_open, facilities, customers, cost_matrix)
            if feasible:
                fixed_cost = sum(facilities[f]["fixed_cost"] for f in cur_open)
                total = fixed_cost + trans_cost
                if total < best_cost:
                    best_cost = total
                    best_open = set(cur_open)
                    best_assignments = assigned
                    best_fixed_cost = fixed_cost
                    best_trans_cost = trans_cost

        # Phase 2: Local Search (ADD, DROP, SWAP moves)
        improved = True
        iterations = 0
        deadline = time.perf_counter() + self.time_limit

        while improved and iterations < self.max_iterations and time.perf_counter() < deadline:
            improved = False
            iterations += 1

            # Try ADD moves
            for fid in facility_ids:
                if fid not in best_open:
                    candidate = best_open | {fid}
                    assigned, trans_cost, feasible = self._evaluate_open_set(candidate, facilities, customers, cost_matrix)
                    if feasible:
                        fixed_cost = sum(facilities[f]["fixed_cost"] for f in candidate)
                        tot = fixed_cost + trans_cost
                        if tot < best_cost - 1e-4:
                            best_cost = tot
                            best_open = candidate
                            best_assignments = assigned
                            best_fixed_cost = fixed_cost
                            best_trans_cost = trans_cost
                            improved = True
                            break

            if improved:
                continue

            # Try DROP moves
            if len(best_open) > 1:
                for fid in list(best_open):
                    candidate = best_open - {fid}
                    assigned, trans_cost, feasible = self._evaluate_open_set(candidate, facilities, customers, cost_matrix)
                    if feasible:
                        fixed_cost = sum(facilities[f]["fixed_cost"] for f in candidate)
                        tot = fixed_cost + trans_cost
                        if tot < best_cost - 1e-4:
                            best_cost = tot
                            best_open = candidate
                            best_assignments = assigned
                            best_fixed_cost = fixed_cost
                            best_trans_cost = trans_cost
                            improved = True
                            break

            if improved:
                continue

            # Try SWAP moves (replace one open facility with one closed)
            for f_open in list(best_open):
                for f_closed in facility_ids:
                    if f_closed not in best_open:
                        candidate = (best_open - {f_open}) | {f_closed}
                        assigned, trans_cost, feasible = self._evaluate_open_set(candidate, facilities, customers, cost_matrix)
                        if feasible:
                            fixed_cost = sum(facilities[f]["fixed_cost"] for f in candidate)
                            tot = fixed_cost + trans_cost
                            if tot < best_cost - 1e-4:
                                best_cost = tot
                                best_open = candidate
                                best_assignments = assigned
                                best_fixed_cost = fixed_cost
                                best_trans_cost = trans_cost
                                improved = True
                                break
                if improved:
                    break

        elapsed_us = (time.perf_counter() - start_time) * 1_000_000.0

        return FacilityLocationResult(
            status=SolverStatus.OPTIMAL if not improved else SolverStatus.FEASIBLE,
            opened_facility_ids=sorted(list(best_open)),
            demand_assignments=best_assignments,
            total_cost=round(best_cost, 4),
            fixed_capital_cost=round(best_fixed_cost, 4),
            transportation_cost=round(best_trans_cost, 4),
            elapsed_microseconds=round(elapsed_us, 2),
        )

    def _evaluate_open_set(
        self,
        open_set: Set[str],
        facilities: Dict[str, Dict[str, float]],
        customers: Dict[str, Dict[str, float]],
        cost_matrix: Dict[str, Dict[str, float]],
    ) -> Tuple[Dict[str, str], float, bool]:
        """Assign customers to open facilities greedily respecting capacity."""
        total_open_capacity = sum(facilities[f].get("capacity", float("inf")) for f in open_set)
        total_demand = sum(c.get("demand", 1.0) for c in customers.values())
        if total_open_capacity < total_demand:
            return {}, float("inf"), False

        # Remaining capacity per open facility
        rem_cap: Dict[str, float] = {f: facilities[f].get("capacity", float("inf")) for f in open_set}
        assignments: Dict[str, str] = {}
        total_transport_cost = 0.0

        # Sort customers by regret (difference between cheapest and 2nd cheapest open facility)
        # to ensure constrained customers get priority for their closest facility
        customer_regret: List[Tuple[str, float]] = []
        for cid, c in customers.items():
            costs = sorted([cost_matrix[f][cid] for f in open_set])
            regret = costs[1] - costs[0] if len(costs) > 1 else 0.0
            customer_regret.append((cid, regret))

        # Sort descending by regret
        customer_regret.sort(key=lambda x: x[1], reverse=True)

        for cid, _ in customer_regret:
            demand = customers[cid].get("demand", 1.0)
            # Find cheapest facility in open_set that has sufficient capacity
            viable = [f for f in open_set if rem_cap[f] >= demand]
            if not viable:
                return {}, float("inf"), False

            best_f = min(viable, key=lambda f: cost_matrix[f][cid])
            assignments[cid] = best_f
            rem_cap[best_f] -= demand
            total_transport_cost += cost_matrix[best_f][cid]

        return assignments, total_transport_cost, True
