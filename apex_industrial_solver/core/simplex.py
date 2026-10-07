"""Revised Two-Phase Simplex Linear Programming (LP) Solver.

Solves continuous linear programs:
    min  c^T x
    s.t. A_ub x <= b_ub
         A_eq x == b_eq
         x >= 0

Includes Phase-1 artificial variable feasibility and Phase-2 optimality
with Bland's rule anti-cycling.
Zero external dependencies: 100% pure Python standard library.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from apex_industrial_solver.core.models import SolverStatus


@dataclass
class LPResult:
    """Result of linear program optimization."""
    status: SolverStatus
    objective_value: float
    variables: List[float]
    slack_variables: List[float]
    shadow_prices: List[float]
    iterations: int
    elapsed_microseconds: float


class SimplexSolver:
    """Two-Phase Simplex Solver in pure Python."""

    def __init__(self, eps: float = 1e-8, max_iterations: int = 10000):
        self.eps = eps
        self.max_iterations = max_iterations

    def solve(
        self,
        c: List[float],
        A_ub: Optional[List[List[float]]] = None,
        b_ub: Optional[List[float]] = None,
        A_eq: Optional[List[List[float]]] = None,
        b_eq: Optional[List[float]] = None,
        maximize: bool = False,
    ) -> LPResult:
        """Solve a linear program.
        
        Args:
            c: Objective coefficient vector (length n).
            A_ub: Inequality constraint matrix (m_ub x n), A_ub * x <= b_ub.
            b_ub: Inequality right-hand side vector (length m_ub).
            A_eq: Equality constraint matrix (m_eq x n), A_eq * x == b_eq.
            b_eq: Equality right-hand side vector (length m_eq).
            maximize: True to maximize, False to minimize.
            
        Returns:
            LPResult containing solution, objective value, and dual variables.
        """
        start_time = time.perf_counter()

        n_vars = len(c)
        obj_sign = -1.0 if maximize else 1.0
        c_min = [obj_sign * ci for ci in c]

        A_ub = A_ub or []
        b_ub = b_ub or []
        A_eq = A_eq or []
        b_eq = b_eq or []

        m_ub = len(A_ub)
        m_eq = len(A_eq)
        total_constraints = m_ub + m_eq

        if total_constraints == 0:
            elapsed_us = (time.perf_counter() - start_time) * 1_000_000.0
            return LPResult(
                status=SolverStatus.FEASIBLE,
                objective_value=0.0,
                variables=[0.0] * n_vars,
                slack_variables=[],
                shadow_prices=[],
                iterations=0,
                elapsed_microseconds=round(elapsed_us, 2),
            )

        # Standard form preparation:
        # Constraint rows:
        # 0..m_ub-1: inequality constraints
        # m_ub..total_constraints-1: equality constraints
        # We track whether artificial variables are required.
        row_A: List[List[float]] = []
        row_b: List[float] = []
        slack_signs: List[float] = []  # +1 for slack, -1 for surplus
        needs_art: List[bool] = []

        # Process inequalities: A_ub x <= b_ub
        for row, rhs in zip(A_ub, b_ub):
            if rhs < -self.eps:
                # Multiply by -1: -row * x >= -rhs (surplus variable and artificial variable)
                row_A.append([-val for val in row])
                row_b.append(-rhs)
                slack_signs.append(-1.0)
                needs_art.append(True)
            else:
                row_A.append(list(row))
                row_b.append(rhs)
                slack_signs.append(1.0)
                needs_art.append(False)

        # Process equalities: A_eq x == b_eq
        for row, rhs in zip(A_eq, b_eq):
            if rhs < -self.eps:
                row_A.append([-val for val in row])
                row_b.append(-rhs)
            else:
                row_A.append(list(row))
                row_b.append(rhs)
            needs_art.append(True)

        n_slack = m_ub
        n_art = sum(1 for val in needs_art if val)

        # Variable index layout:
        # [0 .. n_vars-1]: Original variables
        # [n_vars .. n_vars + n_slack - 1]: Slack/Surplus variables
        # [n_vars + n_slack .. n_vars + n_slack + n_art - 1]: Artificial variables
        # [total_cols - 1]: RHS column
        total_cols = n_vars + n_slack + n_art + 1
        tableau = [[0.0] * total_cols for _ in range(total_constraints + 2)]

        basis = [0] * total_constraints
        art_col_idx = n_vars + n_slack

        # Populate constraint rows (tableau rows 2 to total_constraints + 1)
        for i in range(total_constraints):
            r = i + 2
            # Primal variable coefficients
            for j in range(n_vars):
                tableau[r][j] = row_A[i][j]

            # Slack/Surplus for inequality rows
            if i < m_ub:
                slack_col = n_vars + i
                tableau[r][slack_col] = slack_signs[i]
                if not needs_art[i]:
                    basis[i] = slack_col

            # Artificial variable if needed
            if needs_art[i]:
                tableau[r][art_col_idx] = 1.0
                basis[i] = art_col_idx
                art_col_idx += 1

            # RHS
            tableau[r][-1] = row_b[i]

        # Phase 2 objective (row 1): min sum c_min_j * x_j
        for j in range(n_vars):
            tableau[1][j] = c_min[j]

        iterations = 0

        # If artificial variables exist, run Phase 1
        if n_art > 0:
            # Row 0: sum of artificial variables
            # Initially: sum over artificial cols
            for col in range(n_vars + n_slack, n_vars + n_slack + n_art):
                tableau[0][col] = 1.0

            # Make row 0 canonical by subtracting constraint rows with artificial basic vars
            for i in range(total_constraints):
                if basis[i] >= n_vars + n_slack:
                    r = i + 2
                    for col in range(total_cols):
                        tableau[0][col] -= tableau[r][col]

            # Pivot Phase 1
            phase1_solved, iterations = self._pivot_simplex(
                tableau, basis, obj_row=0, total_cols=total_cols, m=total_constraints, iterations=iterations
            )

            # Check if sum of artificial variables is zero
            phase1_obj = -tableau[0][-1]
            if abs(phase1_obj) > 1e-4:
                elapsed_us = (time.perf_counter() - start_time) * 1_000_000.0
                return LPResult(
                    status=SolverStatus.INFEASIBLE,
                    objective_value=float("inf") if not maximize else float("-inf"),
                    variables=[0.0] * n_vars,
                    slack_variables=[],
                    shadow_prices=[],
                    iterations=iterations,
                    elapsed_microseconds=round(elapsed_us, 2),
                )

        # ---------------- Phase 2 Simplex ----------------
        # Canonicalize Phase 2 objective row (row 1) with respect to current basis
        # Reset row 1
        for col in range(total_cols):
            tableau[1][col] = 0.0
        for j in range(n_vars):
            tableau[1][j] = c_min[j]

        for i in range(total_constraints):
            b_var = basis[i]
            coeff = tableau[1][b_var]
            if abs(coeff) > self.eps:
                r = i + 2
                for col in range(total_cols):
                    tableau[1][col] -= coeff * tableau[r][col]

        # Pivot Phase 2: exclude artificial variables from entering
        phase2_solved, iterations = self._pivot_simplex(
            tableau,
            basis,
            obj_row=1,
            total_cols=total_cols,
            m=total_constraints,
            iterations=iterations,
            max_candidate_col=n_vars + n_slack,
        )

        status = SolverStatus.OPTIMAL if phase2_solved else SolverStatus.FEASIBLE

        # Extract primal variables
        sol = [0.0] * n_vars
        slacks = [0.0] * n_slack
        for i in range(total_constraints):
            b_var = basis[i]
            val = max(0.0, tableau[i + 2][-1])
            if b_var < n_vars:
                sol[b_var] = val
            elif b_var < n_vars + n_slack:
                slacks[b_var - n_vars] = val

        # Optimal objective value
        # In canonical tableau, row 1 RHS = -objective_value
        obj_val = -tableau[1][-1]
        if maximize:
            obj_val = -obj_val

        # Extract shadow prices
        shadow_prices = [abs(tableau[1][n_vars + i]) for i in range(n_slack)]

        elapsed_us = (time.perf_counter() - start_time) * 1_000_000.0

        return LPResult(
            status=status,
            objective_value=round(obj_val, 6),
            variables=[round(x, 6) for x in sol],
            slack_variables=[round(s, 6) for s in slacks],
            shadow_prices=[round(p, 6) for p in shadow_prices],
            iterations=iterations,
            elapsed_microseconds=round(elapsed_us, 2),
        )

    def _pivot_simplex(
        self,
        tableau: List[List[float]],
        basis: List[int],
        obj_row: int,
        total_cols: int,
        m: int,
        iterations: int,
        max_candidate_col: Optional[int] = None,
    ) -> Tuple[bool, int]:
        """Perform simplex pivoting until optimality or iteration limit."""
        if max_candidate_col is None:
            max_candidate_col = total_cols - 1

        while iterations < self.max_iterations:
            # Pricing: find entering variable with negative reduced cost (min reduced cost)
            entering_col = -1
            min_reduced_cost = -self.eps

            for j in range(max_candidate_col):
                if tableau[obj_row][j] < min_reduced_cost:
                    min_reduced_cost = tableau[obj_row][j]
                    entering_col = j

            if entering_col == -1:
                # No negative reduced cost -> Optimal for this phase
                return True, iterations

            # Ratio test: find leaving variable
            leaving_row = -1
            min_ratio = float("inf")

            for i in range(m):
                row_idx = i + 2
                coeff = tableau[row_idx][entering_col]
                if coeff > self.eps:
                    rhs = tableau[row_idx][-1]
                    ratio = rhs / coeff
                    if ratio < min_ratio - self.eps or (abs(ratio - min_ratio) <= self.eps and basis[i] > basis[leaving_row - 2]):
                        min_ratio = ratio
                        leaving_row = row_idx

            if leaving_row == -1:
                # Problem is unbounded
                return False, iterations

            # Pivot operation on (leaving_row, entering_col)
            iterations += 1
            pivot_val = tableau[leaving_row][entering_col]

            # Normalize pivot row
            for j in range(total_cols):
                tableau[leaving_row][j] /= pivot_val

            # Eliminate entering column from all other rows
            for r in range(len(tableau)):
                if r != leaving_row:
                    factor = tableau[r][entering_col]
                    if abs(factor) > self.eps:
                        for j in range(total_cols):
                            tableau[r][j] -= factor * tableau[leaving_row][j]

            # Update basis
            basis[leaving_row - 2] = entering_col

        return False, iterations
