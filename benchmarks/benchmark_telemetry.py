"""Microsecond Benchmark Telemetry Suite for Apex Industrial Solver.

Pure Python 3.10+ standard library.
"""

from __future__ import annotations

import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex_industrial_solver.core.models import (
    Box3D,
    Container3D,
    FleetStop,
    TaskOperation,
)
from apex_industrial_solver.hypervisor import IndustrialSolverHypervisor


def benchmark_call(fn: Callable, runs: int = 20) -> Tuple[float, float, float]:
    """Measures execution latency over multiple runs.
    
    Returns (mean_us, p95_us, ops_per_sec).
    """
    times_us: List[float] = []
    # Warmup
    fn()

    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times_us.append((t1 - t0) / 1000.0)

    mean_us = statistics.mean(times_us)
    times_us.sort()
    p95_idx = int(0.95 * len(times_us))
    p95_us = times_us[min(p95_idx, len(times_us) - 1)]
    ops_per_sec = 1_000_000.0 / mean_us if mean_us > 0 else 0.0

    return mean_us, p95_us, ops_per_sec


def run_benchmarks():
    hypervisor = IndustrialSolverHypervisor()
    print("================================================================================")
    print("        APEX INDUSTRIAL SOLVER - MICROSECOND BENCHMARK TELEMETRY                ")
    print("================================================================================\n")

    results = []

    # 1. 3D Bin Packing
    container = Container3D("CONT-40FT", 235.0, 1200.0, 239.0, 30000.0)
    for n_boxes in [20, 50, 100]:
        boxes = [
            Box3D(f"B-{i}", 40.0 + (i % 4) * 15.0, 60.0 + (i % 3) * 20.0, 30.0 + (i % 5) * 10.0, 45.0 + (i % 7) * 20.0)
            for i in range(n_boxes)
        ]
        mean_us, p95_us, ops = benchmark_call(lambda: hypervisor.pack_3d(container, boxes), runs=15)
        results.append(("3D Bin Packing", f"{n_boxes} cargo boxes", mean_us, p95_us, ops))

    # 2. VRPTW
    depot = FleetStop("DEPOT", 50.0, 50.0, 0.0, 0.0, 600.0, 0.0)
    for n_stops in [10, 25, 50]:
        stops = [
            FleetStop(f"S-{i}", 10.0 + (i * 17) % 80, 15.0 + (i * 29) % 75, 10.0 + (i % 5) * 5.0, (i * 15) % 180, (i * 15) % 180 + 120.0, 10.0)
            for i in range(1, n_stops + 1)
        ]
        caps = [120.0] * 5
        mean_us, p95_us, ops = benchmark_call(lambda: hypervisor.route_fleet(depot, stops, caps), runs=15)
        results.append(("VRPTW Routing", f"{n_stops} stops, 5 veh", mean_us, p95_us, ops))

    # 3. JSSP Flexible Job Shop
    for (n_jobs, n_mach) in [(4, 3), (8, 4), (12, 5)]:
        ops = []
        for j in range(n_jobs):
            for s in range(3):
                m_map = {f"M{(s + k) % n_mach}": 10.0 + ((j + s + k) % 3) * 5.0 for k in range(min(2, n_mach))}
                ops.append(TaskOperation(f"J{j}_O{s}", f"J{j}", s, m_map))
        mean_us, p95_us, ops_s = benchmark_call(lambda: hypervisor.schedule_jobs(ops), runs=15)
        results.append(("JSSP Scheduling", f"{n_jobs} jobs, {n_mach} mach", mean_us, p95_us, ops_s))

    # 4. CFLP Facility Location
    for (n_fac, n_cust) in [(4, 15), (8, 30)]:
        facs = {f"F-{i}": {"x": 20.0 + i * 20.0, "y": 30.0 + (i % 2) * 30.0, "fixed_cost": 2000.0 + i * 500.0, "capacity": 300.0} for i in range(n_fac)}
        custs = {f"C-{i}": {"x": 10.0 + (i * 13) % 90, "y": 10.0 + (i * 19) % 90, "demand": 5.0 + (i % 4) * 4.0} for i in range(n_cust)}
        mean_us, p95_us, ops_s = benchmark_call(lambda: hypervisor.locate_facilities(facs, custs), runs=15)
        results.append(("CFLP Facility Siting", f"{n_fac} fac, {n_cust} cust", mean_us, p95_us, ops_s))

    # 5. Two-Phase Simplex LP
    c_lp = [2.0, 3.0, 1.5, 4.0, 2.5]
    A_ub = [
        [1.0, 2.0, 1.0, 1.0, 2.0],
        [2.0, 1.0, 2.0, 3.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ]
    b_ub = [20.0, 25.0, 15.0]
    mean_us, p95_us, ops_s = benchmark_call(lambda: hypervisor.solve_lp(c_lp, A_ub=A_ub, b_ub=b_ub, maximize=True), runs=30)
    results.append(("Simplex LP", "5 vars, 3 constr", mean_us, p95_us, ops_s))

    # 6. Branch-and-Bound MILP
    c_milp = [3.0, 5.0, 2.0]
    A_milp = [
        [2.0, 3.0, 1.0],
        [1.0, 2.0, 2.0],
    ]
    b_milp = [14.0, 10.0]
    mean_us, p95_us, ops_s = benchmark_call(lambda: hypervisor.solve_milp(c_milp, A_ub=A_milp, b_ub=b_milp, integer_indices=[0, 1, 2], maximize=True), runs=20)
    results.append(("Branch-and-Bound MILP", "3 int vars, 2 constr", mean_us, p95_us, ops_s))

    # Print Table
    header = f"| {'Solver Domain':<22} | {'Problem Scale':<20} | {'Mean Latency (µs)':<18} | {'p95 Latency (µs)':<18} | {'Throughput (ops/s)':<18} |"
    sep = f"|{'-'*24}|{'-'*22}|{'-'*20}|{'-'*20}|{'-'*20}|"
    print(header)
    print(sep)
    for domain, scale, mean_u, p95_u, ops_s in results:
        print(f"| {domain:<22} | {scale:<20} | {mean_u:>16.2f} µs | {p95_u:>16.2f} µs | {ops_s:>16.1f} /s |")
    print(sep)


if __name__ == "__main__":
    run_benchmarks()
