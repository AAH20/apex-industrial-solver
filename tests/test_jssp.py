"""Unit tests for Flexible Job Shop Scheduling (JSSP)."""

import unittest
from apex_industrial_solver.core.jssp import FlexibleJobShopSolver
from apex_industrial_solver.core.models import SolverStatus, TaskOperation


class TestJSSP(unittest.TestCase):
    def setUp(self):
        self.solver = FlexibleJobShopSolver(max_iterations=100)

    def test_two_job_schedule(self):
        # Job 0: Op0 (M0: 10, M1: 15) -> Op1 (M0: 20, M1: 10)
        # Job 1: Op0 (M0: 15, M1: 10) -> Op1 (M0: 10, M1: 20)
        ops = [
            TaskOperation("J0_O0", "J0", 0, {"M0": 10.0, "M1": 15.0}),
            TaskOperation("J0_O1", "J0", 1, {"M0": 20.0, "M1": 10.0}),
            TaskOperation("J1_O0", "J1", 0, {"M0": 15.0, "M1": 10.0}),
            TaskOperation("J1_O1", "J1", 1, {"M0": 10.0, "M1": 20.0}),
        ]
        res = self.solver.solve(ops)
        self.assertIn(res.status, [SolverStatus.OPTIMAL, SolverStatus.FEASIBLE])
        self.assertEqual(len(res.scheduled_tasks), 4)
        self.assertTrue(res.makespan > 0)
        self.assertTrue(res.average_machine_utilization > 0)

        # Verify precedence constraints: Op1 start >= Op0 completion
        task_map = {t.task_id: t for t in res.scheduled_tasks}
        self.assertGreaterEqual(task_map["J0_O1"].start_time, task_map["J0_O0"].completion_time - 1e-5)
        self.assertGreaterEqual(task_map["J1_O1"].start_time, task_map["J1_O0"].completion_time - 1e-5)

        # Verify no machine conflicts
        for m, tasks in res.machine_schedules.items():
            sorted_tasks = sorted(tasks, key=lambda t: t.start_time)
            for i in range(len(sorted_tasks) - 1):
                t1, t2 = sorted_tasks[i], sorted_tasks[i + 1]
                self.assertLessEqual(t1.completion_time, t2.start_time + 1e-5, f"Machine {m} collision")

    def test_empty_operations(self):
        res = self.solver.solve([])
        self.assertEqual(res.makespan, 0.0)
        self.assertEqual(len(res.scheduled_tasks), 0)


if __name__ == "__main__":
    unittest.main()
