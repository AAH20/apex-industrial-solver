"""Unified Command Line Interface for Apex Industrial Solver.

Pure Python standard library: zero external dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List
from apex_industrial_solver.core.models import (
    Box3D,
    Container3D,
    FleetStop,
    TaskOperation,
)
from apex_industrial_solver.hypervisor import IndustrialSolverHypervisor


def main():
    parser = argparse.ArgumentParser(
        prog="apex-solver",
        description="Apex Industrial Solver: Zero-Dependency Industrial Operations Research Kernels",
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational sub-commands")

    # pack3d
    p_pack = subparsers.add_parser("pack3d", help="3D Container Bin Packing with CoG Stability")
    p_pack.add_argument("--container", type=str, default="240,600,240,25000", help="w,l,h,max_weight")
    p_pack.add_argument("--count", type=int, default=30, help="Number of cargo boxes to pack")

    # route
    p_route = subparsers.add_parser("route", help="Capacitated Vehicle Routing with Time Windows (VRPTW)")
    p_route.add_argument("--stops", type=int, default=15, help="Number of delivery stops")
    p_route.add_argument("--vehicles", type=int, default=3, help="Number of vehicles in fleet")

    # schedule
    p_sched = subparsers.add_parser("schedule", help="Flexible Job Shop Scheduling (JSSP)")
    p_sched.add_argument("--jobs", type=int, default=5, help="Number of multi-operation jobs")
    p_sched.add_argument("--machines", type=int, default=4, help="Number of production machines")

    # locate
    p_locate = subparsers.add_parser("locate", help="Capacitated Facility Location (CFLP)")
    p_locate.add_argument("--facilities", type=int, default=5, help="Candidate facility sites")
    p_locate.add_argument("--customers", type=int, default=20, help="Customer demand nodes")

    # benchmark
    subparsers.add_parser("benchmark", help="Execute full microsecond benchmark telemetry suite")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    hypervisor = IndustrialSolverHypervisor()

    if args.command == "pack3d":
        cw, cl, ch, cweight = [float(x) for x in args.container.split(",")]
        container = Container3D("CONT-01", cw, cl, ch, cweight)
        boxes = []
        for i in range(args.count):
            w = 40.0 + (i % 4) * 15.0
            l = 60.0 + (i % 3) * 20.0
            h = 30.0 + (i % 5) * 10.0
            wt = 50.0 + (i % 7) * 25.0
            boxes.append(Box3D(f"BOX-{i:03d}", w, l, h, wt))

        print(f"\n[Apex-Solver] Executing 3D Bin Packing: {len(boxes)} items into Container ({cw}x{cl}x{ch})...")
        res = hypervisor.pack_3d(container, boxes)
        print(f"Status:              {res.status.value.upper()}")
        print(f"Packed Items:        {len(res.packed_items)} / {len(boxes)}")
        print(f"Volume Utilization:  {res.volume_utilization * 100:.2f}%")
        print(f"Weight Utilization:  {res.weight_utilization * 100:.2f}%")
        print(f"Center of Gravity:   X={res.center_of_gravity[0]:.1f}, Y={res.center_of_gravity[1]:.1f}, Z={res.center_of_gravity[2]:.1f}")
        print(f"CoG Stable:          {res.cog_is_stable}")
        print(f"Latency:             {res.elapsed_microseconds:.2f} µs ({res.elapsed_microseconds / 1000.0:.3f} ms)\n")

    elif args.command == "route":
        depot = FleetStop("DEPOT", 50.0, 50.0, 0.0, 0.0, 480.0, 0.0)
        stops = []
        for i in range(1, args.stops + 1):
            x = 10.0 + (i * 17) % 80
            y = 15.0 + (i * 29) % 75
            demand = 10.0 + (i % 5) * 5.0
            ready = (i * 15) % 180
            due = ready + 90.0
            stops.append(FleetStop(f"STOP-{i:02d}", x, y, demand, ready, due, 10.0))

        capacities = [100.0] * args.vehicles
        print(f"\n[Apex-Solver] Routing {len(stops)} stops with {len(capacities)} vehicles (VRPTW)...")
        res = hypervisor.route_fleet(depot, stops, capacities)
        print(f"Status:              {res.status.value.upper()}")
        print(f"Vehicles Used:       {res.vehicles_used} / {len(capacities)}")
        print(f"Total Distance:      {res.total_distance:.2f} km")
        print(f"Unassigned Stops:    {len(res.unassigned_stops)}")
        print(f"Latency:             {res.elapsed_microseconds:.2f} µs ({res.elapsed_microseconds / 1000.0:.3f} ms)\n")

    elif args.command == "schedule":
        ops = []
        for j in range(args.jobs):
            for s in range(3):
                m_map = {
                    f"M{(s + k) % args.machines}": 10.0 + ((j + s + k) % 4) * 5.0
                    for k in range(min(2, args.machines))
                }
                ops.append(TaskOperation(f"J{j}_O{s}", f"JOB-{j}", s, m_map))

        print(f"\n[Apex-Solver] Scheduling {len(ops)} operations across {args.machines} machines (JSSP)...")
        res = hypervisor.schedule_jobs(ops)
        print(f"Status:              {res.status.value.upper()}")
        print(f"Makespan (C_max):    {res.makespan:.2f} hrs")
        print(f"Avg Utilization:     {res.average_machine_utilization * 100:.2f}%")
        print(f"Latency:             {res.elapsed_microseconds:.2f} µs ({res.elapsed_microseconds / 1000.0:.3f} ms)\n")

    elif args.command == "locate":
        facilities = {
            f"DC-{i}": {
                "x": 20.0 + i * 25.0,
                "y": 30.0 + (i % 2) * 40.0,
                "fixed_cost": 5000.0 + i * 1500.0,
                "capacity": 200.0,
            }
            for i in range(args.facilities)
        }
        customers = {
            f"CUST-{i}": {
                "x": 10.0 + (i * 13) % 90,
                "y": 10.0 + (i * 19) % 90,
                "demand": 5.0 + (i % 4) * 3.0,
            }
            for i in range(args.customers)
        }

        print(f"\n[Apex-Solver] Siting {args.facilities} candidate facilities for {args.customers} customers (CFLP)...")
        res = hypervisor.locate_facilities(facilities, customers)
        print(f"Status:              {res.status.value.upper()}")
        print(f"Opened Facilities:   {', '.join(res.opened_facility_ids)}")
        print(f"Total Cost:          ${res.total_cost:,.2f} (Fixed: ${res.fixed_capital_cost:,.2f}, Trans: ${res.transportation_cost:,.2f})")
        print(f"Latency:             {res.elapsed_microseconds:.2f} µs ({res.elapsed_microseconds / 1000.0:.3f} ms)\n")

    elif args.command == "benchmark":
        from benchmarks.benchmark_telemetry import run_benchmarks
        run_benchmarks()


if __name__ == "__main__":
    main()
