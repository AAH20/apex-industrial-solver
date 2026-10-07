"""Unit tests for IndustrialSolverHypervisor unified API."""

import unittest
from apex_industrial_solver.core.models import (
    Box3D,
    Container3D,
    FleetStop,
    SolverStatus,
    TaskOperation,
)
from apex_industrial_solver.hypervisor import IndustrialSolverHypervisor


class TestHypervisor(unittest.TestCase):
    def setUp(self):
        self.hypervisor = IndustrialSolverHypervisor()

    def test_hypervisor_facade(self):
        # 3D Bin packing
        c = Container3D("C1", 100.0, 100.0, 100.0, 1000.0)
        boxes = [Box3D("B1", 20.0, 20.0, 20.0, 10.0)]
        res_pack = self.hypervisor.pack_3d(c, boxes)
        self.assertEqual(res_pack.status, SolverStatus.OPTIMAL)

        # VRPTW
        depot = FleetStop("D", 0.0, 0.0, 0.0, 0.0, 100.0, 0.0)
        stop = FleetStop("S", 10.0, 0.0, 5.0, 5.0, 50.0, 5.0)
        res_vrp = self.hypervisor.route_fleet(depot, [stop], [50.0])
        self.assertEqual(res_vrp.vehicles_used, 1)

        # JSSP
        op = TaskOperation("T1", "J1", 0, {"M1": 5.0})
        res_jssp = self.hypervisor.schedule_jobs([op])
        self.assertEqual(res_jssp.makespan, 5.0)

        # Facility location
        fac = {"F1": {"x": 0.0, "y": 0.0, "fixed_cost": 10.0, "capacity": 10.0}}
        cust = {"C1": {"x": 1.0, "y": 1.0, "demand": 2.0}}
        res_cflp = self.hypervisor.locate_facilities(fac, cust)
        self.assertEqual(len(res_cflp.opened_facility_ids), 1)


if __name__ == "__main__":
    unittest.main()
