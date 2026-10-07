"""Unit tests for Capacitated Facility Location Problem (CFLP)."""

import unittest
from apex_industrial_solver.core.facility_location import FacilityLocationSolver
from apex_industrial_solver.core.models import SolverStatus


class TestFacilityLocation(unittest.TestCase):
    def setUp(self):
        self.solver = FacilityLocationSolver(max_iterations=100)

    def test_basic_facility_siting(self):
        facilities = {
            "F1": {"x": 0.0, "y": 0.0, "fixed_cost": 100.0, "capacity": 50.0},
            "F2": {"x": 10.0, "y": 10.0, "fixed_cost": 100.0, "capacity": 50.0},
        }
        customers = {
            "C1": {"x": 1.0, "y": 1.0, "demand": 10.0},
            "C2": {"x": 9.0, "y": 9.0, "demand": 10.0},
        }
        res = self.solver.solve(facilities, customers)
        self.assertIn(res.status, [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE])
        self.assertTrue(len(res.opened_facility_ids) >= 1)
        self.assertEqual(len(res.demand_assignments), 2)
        self.assertTrue(res.total_cost > 0)

    def test_infeasible_capacity(self):
        facilities = {
            "F1": {"x": 0.0, "y": 0.0, "fixed_cost": 100.0, "capacity": 10.0},
        }
        customers = {
            "C1": {"x": 1.0, "y": 1.0, "demand": 50.0},
        }
        res = self.solver.solve(facilities, customers)
        self.assertEqual(res.status, SolverStatus.INFEASIBLE)


if __name__ == "__main__":
    unittest.main()
