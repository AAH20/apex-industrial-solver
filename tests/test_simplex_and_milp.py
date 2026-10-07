"""Unit tests for Simplex LP and Branch-and-Bound MILP solvers."""

import unittest
from apex_industrial_solver.core.branch_and_bound import BranchAndBoundSolver
from apex_industrial_solver.core.models import SolverStatus
from apex_industrial_solver.core.simplex import SimplexSolver


class TestSimplexAndMILP(unittest.TestCase):
    def setUp(self):
        self.lp_solver = SimplexSolver()
        self.milp_solver = BranchAndBoundSolver()

    def test_lp_maximization(self):
        # Maximize 3 x1 + 2 x2
        # Subject to:
        #   x1 + x2 <= 4
        #   x1 - x2 <= 2
        #   x1, x2 >= 0
        # Optimal: x1 = 3, x2 = 1, z = 11
        c = [3.0, 2.0]
        A_ub = [
            [1.0, 1.0],
            [1.0, -1.0],
        ]
        b_ub = [4.0, 2.0]
        res = self.lp_solver.solve(c, A_ub=A_ub, b_ub=b_ub, maximize=True)
        self.assertEqual(res.status, SolverStatus.OPTIMAL)
        self.assertAlmostEqual(res.objective_value, 11.0, places=3)
        self.assertAlmostEqual(res.variables[0], 3.0, places=3)
        self.assertAlmostEqual(res.variables[1], 1.0, places=3)

    def test_lp_minimization(self):
        # Minimize 2 x1 + 3 x2
        # Subject to:
        #   x1 + x2 >= 2  <=>  -x1 - x2 <= -2
        #   x1, x2 >= 0
        # Optimal: x1 = 2, x2 = 0, z = 4
        c = [2.0, 3.0]
        A_ub = [
            [-1.0, -1.0],
        ]
        b_ub = [-2.0]
        res = self.lp_solver.solve(c, A_ub=A_ub, b_ub=b_ub, maximize=False)
        self.assertEqual(res.status, SolverStatus.OPTIMAL)
        self.assertAlmostEqual(res.objective_value, 4.0, places=3)

    def test_milp_integer_solution(self):
        # Maximize x1 + x2
        # Subject to:
        #   2 x1 + x2 <= 4.5
        #   x1, x2 in {0, 1, 2, ...}
        # LP relaxation would give x1=0, x2=4.5 (z=4.5) or x1=2.25, x2=0
        # Integer optimal: x1=0, x2=4 (z=4) or x1=1, x2=2 (z=3) etc. Max z = 4 (e.g. x1=0, x2=4)
        c = [1.0, 1.0]
        A_ub = [
            [2.0, 1.0],
        ]
        b_ub = [4.5]
        res = self.milp_solver.solve(c, A_ub=A_ub, b_ub=b_ub, integer_indices=[0, 1], maximize=True)
        self.assertEqual(res.status, SolverStatus.OPTIMAL)
        self.assertAlmostEqual(res.objective_value, 4.0, places=2)
        # All integer variables must be integral
        for val in res.variables:
            self.assertAlmostEqual(val, round(val), places=3)


if __name__ == "__main__":
    unittest.main()
