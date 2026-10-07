"""Branch-and-Bound and Adaptive Large Neighborhood Search (ALNS) Framework.

Solves Mixed-Integer Linear Programs (MILP) and general discrete combinatorial
optimization problems using exact LP-relaxation pruning and adaptive metaheuristics.
Zero external dependencies: 100% pure Python standard library.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple
from apex_industrial_solver.core.models import SolverStatus
from apex_industrial_solver.core.simplex import SimplexSolver, LPResult


@dataclass
class MILPResult:
    """Result of Mixed-Integer Linear Program optimization."""
    status: SolverStatus
    objective_value: float
    variables: List[float]
    nodes_explored: int
    elapsed_microseconds: float


class BranchAndBoundSolver:
    """Branch-and-Bound Integer Programming Solver.
    
    Solves:
        min c^T x
        s.t. A_ub x <= b_ub
             A_eq x == b_eq
             x >= 0
             x_j in Z for j in integer_indices
    """

    def __init__(self, time_limit_seconds: float = 3.0, max_nodes: int = 5000, eps: float = 1e-5):
        self.time_limit = time_limit_seconds
        self.max_nodes = max_nodes
        self.eps = eps
        self.lp_solver = SimplexSolver(eps=1e-8)

    def solve(
        self,
        c: List[float],
        A_ub: Optional[List[List[float]]] = None,
        b_ub: Optional[List[float]] = None,
        integer_indices: Optional[List[int]] = None,
        maximize: bool = False,
    ) -> MILPResult:
        """Solve MILP via Branch and Bound using Simplex relaxations."""
        start_time = time.perf_counter()
        deadline = start_time + self.time_limit

        n_vars = len(c)
        int_set = set(integer_indices or list(range(n_vars)))
        A_ub = [list(r) for r in (A_ub or [])]
        b_ub = list(b_ub or [])

        best_obj = float("-inf") if maximize else float("inf")
        best_solution: Optional[List[float]] = None
        nodes_explored = 0

        # Stack for Depth-First Search branch exploration: (A_ub_node, b_ub_node)
        stack: List[Tuple[List[List[float]], List[float]]] = [(A_ub, b_ub)]

        while stack and nodes_explored < self.max_nodes and time.perf_counter() < deadline:
            nodes_explored += 1
            cur_A_ub, cur_b_ub = stack.pop()

            # Solve LP relaxation at this node
            lp_res = self.lp_solver.solve(c=c, A_ub=cur_A_ub, b_ub=cur_b_ub, maximize=maximize)

            if lp_res.status != SolverStatus.OPTIMAL:
                # Prune by infeasibility
                continue

            # Bound pruning
            if maximize:
                if lp_res.objective_value <= best_obj + self.eps:
                    continue
            else:
                if lp_res.objective_value >= best_obj - self.eps:
                    continue

            # Check integrality of integer variables
            fractional_idx = -1
            max_fractionality = 0.0

            for j in sorted(list(int_set)):
                val = lp_res.variables[j]
                dist_to_int = abs(val - round(val))
                if dist_to_int > self.eps:
                    if dist_to_int > max_fractionality:
                        max_fractionality = dist_to_int
                        fractional_idx = j

            if fractional_idx == -1:
                # Node is integer feasible! Update incumbent best solution
                best_obj = lp_res.objective_value
                best_solution = list(lp_res.variables)
                continue

            # Most-fractional branching on fractional_idx
            val = lp_res.variables[fractional_idx]
            floor_val = math.floor(val)
            ceil_val = math.ceil(val)

            # Left branch: x[fractional_idx] <= floor_val
            left_row = [0.0] * n_vars
            left_row[fractional_idx] = 1.0
            left_A = [list(r) for r in cur_A_ub] + [left_row]
            left_b = list(cur_b_ub) + [float(floor_val)]

            # Right branch: -x[fractional_idx] <= -ceil_val  (i.e. x >= ceil_val)
            right_row = [0.0] * n_vars
            right_row[fractional_idx] = -1.0
            right_A = [list(r) for r in cur_A_ub] + [right_row]
            right_b = list(cur_b_ub) + [-float(ceil_val)]

            # Push child branches onto stack
            stack.append((left_A, left_b))
            stack.append((right_A, right_b))

        elapsed_us = (time.perf_counter() - start_time) * 1_000_000.0

        if best_solution is None:
            return MILPResult(
                status=SolverStatus.INFEASIBLE,
                objective_value=float("inf") if not maximize else float("-inf"),
                variables=[],
                nodes_explored=nodes_explored,
                elapsed_microseconds=round(elapsed_us, 2),
            )

        return MILPResult(
            status=SolverStatus.OPTIMAL if not stack else SolverStatus.FEASIBLE,
            objective_value=round(best_obj, 6),
            variables=[round(x, 6) for x in best_solution],
            nodes_explored=nodes_explored,
            elapsed_microseconds=round(elapsed_us, 2),
        )


class AdaptiveLargeNeighborhoodSearch:
    """Generic ALNS metaheuristic engine for complex combinatorial structures."""

    def __init__(
        self,
        destroy_operators: List[Callable[[Any, float], Any]],
        repair_operators: List[Callable[[Any], Any]],
        cost_evaluator: Callable[[Any], float],
        max_iterations: int = 1000,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.995,
    ):
        self.destroy_ops = destroy_operators
        self.repair_ops = repair_operators
        self.cost_fn = cost_evaluator
        self.max_iter = max_iterations
        self.temp = initial_temperature
        self.cooling = cooling_rate

    def optimize(self, initial_solution: Any) -> Tuple[Any, float]:
        """Execute ALNS simulated annealing loop."""
        current_sol = copy.deepcopy(initial_solution)
        current_cost = self.cost_fn(current_sol)
        best_sol = copy.deepcopy(current_sol)
        best_cost = current_cost

        temp = self.temp
        op_count = len(self.destroy_ops)

        for i in range(self.max_iter):
            # Select operators deterministically cycling or weighted
            d_idx = i % op_count
            r_idx = i % len(self.repair_ops)

            destroy_rate = 0.10 + 0.20 * ((i % 5) / 5.0)
            destroyed = self.destroy_ops[d_idx](current_sol, destroy_rate)
            candidate = self.repair_ops[r_idx](destroyed)
            candidate_cost = self.cost_fn(candidate)

            # Acceptance criterion: Metropolis
            cost_delta = candidate_cost - current_cost
            if cost_delta < 0 or math.exp(-cost_delta / max(1e-5, temp)) > ((i % 100) / 100.0):
                current_sol = candidate
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best_sol = copy.deepcopy(current_sol)
                    best_cost = current_cost

            temp *= self.cooling

        return best_sol, best_cost
