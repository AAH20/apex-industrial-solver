"""Unit tests for 3D Container Bin Packing with CoG Stability."""

import unittest
from apex_industrial_solver.core.bin_packing_3d import BinPacking3DSolver
from apex_industrial_solver.core.models import Box3D, Container3D, SolverStatus


class TestBinPacking3D(unittest.TestCase):
    def setUp(self):
        self.solver = BinPacking3DSolver()
        self.container = Container3D(
            container_id="ISO-20FT",
            width=235.0,
            length=590.0,
            height=239.0,
            max_weight=28000.0,
        )

    def test_single_box_pack(self):
        box = Box3D("B1", width=50.0, length=50.0, height=50.0, weight=100.0)
        res = self.solver.pack(self.container, [box])
        self.assertEqual(res.status, SolverStatus.OPTIMAL)
        self.assertEqual(len(res.packed_items), 1)
        self.assertEqual(len(res.unpacked_items), 0)
        self.assertTrue(res.volume_utilization > 0)
        self.assertEqual(len(res.center_of_gravity), 3)

    def test_multiple_boxes_packing(self):
        boxes = [
            Box3D(f"B{i}", width=40.0, length=60.0, height=30.0, weight=45.0)
            for i in range(20)
        ]
        res = self.solver.pack(self.container, boxes)
        self.assertIn(res.status, [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE])
        self.assertTrue(len(res.packed_items) >= 15)
        self.assertTrue(res.weight_utilization > 0)

        # Check that no two packed boxes overlap
        packed = res.packed_items
        for i in range(len(packed)):
            for j in range(i + 1, len(packed)):
                b1 = packed[i]
                b2 = packed[j]
                overlap = (
                    b1.x < b2.x + b2.placed_width - 1e-4 and
                    b1.x + b1.placed_width > b2.x + 1e-4 and
                    b1.y < b2.y + b2.placed_length - 1e-4 and
                    b1.y + b1.placed_length > b2.y + 1e-4 and
                    b1.z < b2.z + b2.placed_height - 1e-4 and
                    b1.z + b1.placed_height > b2.z + 1e-4
                )
                self.assertFalse(overlap, f"Overlap detected between {b1.item_id} and {b2.item_id}")

    def test_weight_capacity_overflow(self):
        # A box heavier than container max_weight
        heavy_box = Box3D("HEAVY", width=50.0, length=50.0, height=50.0, weight=30000.0)
        res = self.solver.pack(self.container, [heavy_box])
        self.assertEqual(len(res.packed_items), 0)
        self.assertEqual(len(res.unpacked_items), 1)

    def test_empty_box_list(self):
        res = self.solver.pack(self.container, [])
        self.assertEqual(res.status, SolverStatus.OPTIMAL)
        self.assertEqual(len(res.packed_items), 0)


if __name__ == "__main__":
    unittest.main()
