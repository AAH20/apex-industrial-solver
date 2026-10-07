"""Unit tests for Capacitated Vehicle Routing with Time Windows (VRPTW)."""

import unittest
from apex_industrial_solver.core.models import FleetStop, SolverStatus
from apex_industrial_solver.core.vrptw import VRPTWSolver


class TestVRPTW(unittest.TestCase):
    def setUp(self):
        self.solver = VRPTWSolver(max_iterations=100)
        self.depot = FleetStop("DEPOT", x=0.0, y=0.0, demand=0.0, ready_time=0.0, due_time=500.0, service_time=0.0)

    def test_single_vehicle_routing(self):
        stops = [
            FleetStop("S1", x=10.0, y=10.0, demand=15.0, ready_time=10.0, due_time=100.0, service_time=5.0),
            FleetStop("S2", x=20.0, y=10.0, demand=20.0, ready_time=25.0, due_time=150.0, service_time=5.0),
            FleetStop("S3", x=15.0, y=20.0, demand=10.0, ready_time=40.0, due_time=200.0, service_time=5.0),
        ]
        res = self.solver.solve(self.depot, stops, vehicle_capacities=[100.0])
        self.assertIn(res.status, [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE])
        self.assertEqual(len(res.unassigned_stops), 0)
        self.assertEqual(res.vehicles_used, 1)
        self.assertTrue(res.total_distance > 0)

    def test_multi_vehicle_capacity_split(self):
        stops = [
            FleetStop("S1", x=10.0, y=10.0, demand=40.0, ready_time=0.0, due_time=200.0, service_time=5.0),
            FleetStop("S2", x=-10.0, y=10.0, demand=40.0, ready_time=0.0, due_time=200.0, service_time=5.0),
            FleetStop("S3", x=10.0, y=-10.0, demand=40.0, ready_time=0.0, due_time=200.0, service_time=5.0),
        ]
        # Vehicles with capacity 50 each; must use at least 3 vehicles
        res = self.solver.solve(self.depot, stops, vehicle_capacities=[50.0, 50.0, 50.0])
        self.assertEqual(len(res.unassigned_stops), 0)
        self.assertEqual(res.vehicles_used, 3)

    def test_empty_stops(self):
        res = self.solver.solve(self.depot, [], vehicle_capacities=[100.0])
        self.assertEqual(res.vehicles_used, 0)
        self.assertEqual(res.total_distance, 0.0)


if __name__ == "__main__":
    unittest.main()
