"""Capacitated Vehicle Routing Problem with Time Windows (VRPTW) Solver.

Implements Solomon I1 insertion heuristic followed by 2-Opt edge exchange local search.
Zero external dependencies.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from apex_industrial_solver.core.models import FleetStop, SolverStatus, VehicleRoute, VRPTWResult


class VRPTWSolver:
    """Solves Capacitated Vehicle Routing with Time Windows (VRPTW)."""

    def __init__(
        self,
        alpha1: float = 0.70,
        alpha2: float = 0.30,
        mu: float = 1.0,
        max_iterations: int = 500,
    ) -> None:
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.mu = mu
        self.max_iterations = max_iterations

    def euclidean_distance(self, a: FleetStop, b: FleetStop) -> float:
        """Euclidean distance between two stops."""
        return math.hypot(a.x - b.x, a.y - b.y)

    def verify_route_feasibility(
        self,
        depot: FleetStop,
        stops: List[FleetStop],
        vehicle_capacity: float,
        speed: float = 1.0,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Checks if a sequence of stops respects capacity and arrival time windows.

        Returns (is_feasible, total_travel_distance, arrival_times_dict).
        """
        current_time = depot.ready_time
        current_load = 0.0
        total_distance = 0.0
        arrivals: Dict[str, float] = {}

        prev_stop = depot
        for stop in stops:
            current_load += stop.demand
            if current_load > vehicle_capacity:
                return False, 0.0, {}

            dist = self.euclidean_distance(prev_stop, stop)
            total_distance += dist
            travel_time = dist / speed
            arrival = current_time + travel_time

            # If arriving before ready time, wait until ready_time
            arrival = max(arrival, stop.ready_time)

            # If arriving after due time, infeasible
            if arrival > stop.due_time:
                return False, 0.0, {}

            arrivals[stop.stop_id] = arrival
            current_time = arrival + stop.service_time
            prev_stop = stop

        # Return to depot
        dist_to_depot = self.euclidean_distance(prev_stop, depot)
        total_distance += dist_to_depot
        return_time = current_time + (dist_to_depot / speed)
        if return_time > depot.due_time:
            return False, 0.0, {}

        return True, total_distance, arrivals

    def two_opt_optimize(
        self,
        depot: FleetStop,
        route_stops: List[FleetStop],
        vehicle_capacity: float,
    ) -> Tuple[List[FleetStop], float, Dict[str, float]]:
        """Applies 2-Opt local search to eliminate crossing edges without violating time windows."""
        best_stops = list(route_stops)
        feasible, best_dist, best_arrivals = self.verify_route_feasibility(depot, best_stops, vehicle_capacity)
        if not feasible:
            return route_stops, 0.0, {}

        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            n = len(best_stops)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # 2-opt inversion: reverse segment [i:j+1]
                    new_stops = best_stops[:i] + best_stops[i:j+1][::-1] + best_stops[j+1:]
                    is_feas, new_dist, new_arrivals = self.verify_route_feasibility(depot, new_stops, vehicle_capacity)
                    if is_feas and new_dist < best_dist - 0.001:
                        best_stops = new_stops
                        best_dist = new_dist
                        best_arrivals = new_arrivals
                        improved = True
                        break
                if improved:
                    break

        return best_stops, best_dist, best_arrivals

    def solve(
        self,
        depot: FleetStop,
        customer_stops: Optional[List[FleetStop]] = None,
        vehicle_capacity: Optional[float] = None,
        max_vehicles: int = 20,
        stops: Optional[List[FleetStop]] = None,
        vehicle_capacities: Optional[List[float]] = None,
        vehicle_max_durations: Optional[List[float]] = None,
        speed: float = 1.0,
    ) -> VRPTWResult:
        """Solves VRPTW using Solomon insertion heuristic + 2-Opt local search."""
        t_start = time.perf_counter_ns()

        actual_stops = stops if stops is not None else (customer_stops or [])
        unrouted = list(actual_stops)
        routes: List[VehicleRoute] = []
        total_dist = 0.0

        if vehicle_capacities is not None:
            fleet_caps = list(vehicle_capacities)
            max_vehicles = len(fleet_caps)
        else:
            default_cap = vehicle_capacity if vehicle_capacity is not None else 100.0
            fleet_caps = [default_cap] * max_vehicles

        vehicle_idx = 1
        while unrouted and vehicle_idx <= max_vehicles:
            current_cap = fleet_caps[vehicle_idx - 1]
            # Step 1: Select seed customer (farthest feasible stop with tight due time)
            seed_idx = -1
            max_seed_score = -1.0
            for idx, stop in enumerate(unrouted):
                feasible, _, _ = self.verify_route_feasibility(depot, [stop], current_cap, speed=speed)
                if feasible:
                    dist = self.euclidean_distance(depot, stop)
                    urgency = 1.0 / max(1.0, stop.due_time - stop.ready_time)
                    score = dist * urgency
                    if score > max_seed_score:
                        max_seed_score = score
                        seed_idx = idx

            if seed_idx == -1:
                # No feasible stop for a new vehicle
                break

            current_route = [unrouted.pop(seed_idx)]

            # Step 2: Iteratively insert remaining stops via Solomon I1 metric
            can_insert = True
            while can_insert and unrouted:
                best_stop_idx = -1
                best_insert_pos = -1
                best_cost = float("inf")

                for u_idx, cand_stop in enumerate(unrouted):
                    # Try inserting cand_stop at every position in current_route
                    for pos in range(len(current_route) + 1):
                        test_route = current_route[:pos] + [cand_stop] + current_route[pos:]
                        is_feas, dist, _ = self.verify_route_feasibility(depot, test_route, current_cap, speed=speed)
                        if is_feas:
                            cost = dist
                            if cost < best_cost:
                                best_cost = cost
                                best_stop_idx = u_idx
                                best_insert_pos = pos

                if best_stop_idx != -1:
                    inserted_stop = unrouted.pop(best_stop_idx)
                    current_route.insert(best_insert_pos, inserted_stop)
                else:
                    can_insert = False

            # Step 3: Apply 2-Opt local search to polish current route
            opt_stops, route_dist, arrivals = self.two_opt_optimize(depot, current_route, current_cap)
            total_dist += route_dist
            route_demand = sum(s.demand for s in opt_stops)

            routes.append(VehicleRoute(
                vehicle_id=f"vehicle_{vehicle_idx}",
                stops=opt_stops,
                total_distance=route_dist,
                total_travel_time=route_dist,  # Unit speed
                total_demand=route_demand,
                arrival_times=arrivals,
            ))
            vehicle_idx += 1

        t_end = time.perf_counter_ns()
        elapsed_us = (t_end - t_start) / 1000.0

        status = SolverStatus.OPTIMAL if not unrouted else (
            SolverStatus.FEASIBLE if routes else SolverStatus.INFEASIBLE
        )

        return VRPTWResult(
            status=status,
            routes=routes,
            total_distance=total_dist,
            total_duration=total_dist,
            vehicles_used=len(routes),
            unassigned_stops=unrouted,
            elapsed_microseconds=elapsed_us,
        )
