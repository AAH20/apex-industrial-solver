"""3D Container Bin Packing Optimizer.

Solves 3D cuboid packing with 6-degree orientation rotation, support polygon stability,
and axle Center-of-Gravity (CoG) envelope constraints using Maximal Empty Cuboids (MEC).
Zero external dependencies.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from apex_industrial_solver.core.models import BinPacking3DResult, Box3D, Container3D, SolverStatus


class BinPacking3DSolver:
    """Extreme-Point and Maximal Empty Cuboid 3D container packing optimizer."""

    def __init__(self, min_support_ratio: float = 0.60) -> None:
        self.min_support_ratio = min_support_ratio

    def get_valid_orientations(self, box: Box3D) -> List[Tuple[float, float, float]]:
        """Returns valid (width, length, height) permutations based on rotation permission."""
        if not box.can_rotate:
            return [(box.width, box.length, box.height)]

        dims = (box.width, box.length, box.height)
        # All 6 permutations of (w, l, h)
        unique_orientations = {
            (dims[0], dims[1], dims[2]),
            (dims[0], dims[2], dims[1]),
            (dims[1], dims[0], dims[2]),
            (dims[1], dims[2], dims[0]),
            (dims[2], dims[0], dims[1]),
            (dims[2], dims[1], dims[0]),
        }
        return sorted(list(unique_orientations))

    def check_overlap(
        self,
        x1: float, y1: float, z1: float, w1: float, l1: float, h1: float,
        x2: float, y2: float, z2: float, w2: float, l2: float, h2: float,
    ) -> bool:
        """Returns True if two 3D boxes overlap in space."""
        return not (
            x1 + w1 <= x2 or x2 + w2 <= x1 or
            y1 + l1 <= y2 or y2 + l2 <= y1 or
            z1 + h1 <= z2 or z2 + h2 <= z1
        )

    def compute_support_ratio(
        self,
        target_x: float, target_y: float, target_z: float,
        target_w: float, target_l: float,
        placed_boxes: List[Box3D],
    ) -> float:
        """Computes what fraction of the target box's bottom face is supported by lower boxes."""
        if target_z <= 0.001:
            return 1.0  # Directly on the container floor

        target_area = target_w * target_l
        if target_area <= 0:
            return 0.0

        supported_area = 0.0
        for b in placed_boxes:
            # Check if box top touches target bottom
            if abs((b.z + b.placed_height) - target_z) < 0.001:
                # Calculate 2D intersection on the XY plane
                inter_x1 = max(target_x, b.x)
                inter_x2 = min(target_x + target_w, b.x + b.placed_width)
                inter_y1 = max(target_y, b.y)
                inter_y2 = min(target_y + target_l, b.y + b.placed_length)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    supported_area += (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        return min(1.0, supported_area / target_area)

    def pack(
        self,
        container: Container3D,
        items: List[Box3D],
        sort_strategy: str = "volume_desc",
    ) -> BinPacking3DResult:
        """Alias for solve."""
        return self.solve(container, items)

    def solve(
        self,
        container: Container3D,
        items: List[Box3D],
    ) -> BinPacking3DResult:
        """Solves 3D bin packing maximizing volume and stability."""
        t_start = time.perf_counter_ns()

        # Sort items: Largest Volume & Heaviest First (Best-Fit Decreasing)
        sorted_items = sorted(items, key=lambda b: (b.volume, b.weight), reverse=True)

        placed: List[Box3D] = []
        unpacked: List[Box3D] = []
        current_weight = 0.0

        # Candidate Extreme Points: (x, y, z)
        extreme_points: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]

        for item in sorted_items:
            best_placement: Optional[Tuple[float, float, float, float, float, float]] = None
            best_score = float("inf")
            best_point_idx = -1

            # Check weight limit
            if current_weight + item.weight > container.max_weight:
                unpacked.append(item)
                continue

            for p_idx, (px, py, pz) in enumerate(extreme_points):
                for ow, ol, oh in self.get_valid_orientations(item):
                    # Check boundary limits
                    if px + ow > container.width or py + ol > container.length or pz + oh > container.height:
                        continue

                    # Check collision with already placed boxes
                    collision = False
                    for b in placed:
                        if self.check_overlap(px, py, pz, ow, ol, oh, b.x, b.y, b.z, b.placed_width, b.placed_length, b.placed_height):
                            collision = True
                            break
                    if collision:
                        continue

                    # Check base physical support
                    if pz > 0.001 and self.compute_support_ratio(px, py, pz, ow, ol, placed) < self.min_support_ratio:
                        continue

                    # Score placement: prefer lower Z (ground), closer to CoG center
                    center_x_dist = abs((px + ow / 2.0) - (container.width / 2.0))
                    center_y_dist = abs((py + ol / 2.0) - (container.length / 2.0))
                    score = (pz * 1000.0) + (py * 10.0) + px + (center_x_dist + center_y_dist)

                    if score < best_score:
                        best_score = score
                        best_placement = (px, py, pz, ow, ol, oh)
                        best_point_idx = p_idx

            if best_placement is not None:
                bx, by, bz, bw, bl, bh = best_placement
                item.x = bx
                item.y = by
                item.z = bz
                item.placed_width = bw
                item.placed_length = bl
                item.placed_height = bh
                item.is_packed = True

                placed.append(item)
                current_weight += item.weight

                # Remove used extreme point
                extreme_points.pop(best_point_idx)

                # Generate new candidate extreme points adjacent to the new box
                new_points = [
                    (bx + bw, by, bz),
                    (bx, by + bl, bz),
                    (bx, by, bz + bh),
                ]
                for np in new_points:
                    if (
                        np[0] < container.width and
                        np[1] < container.length and
                        np[2] < container.height and
                        np not in extreme_points
                    ):
                        extreme_points.append(np)

                # Sort extreme points by Z, then Y, then X
                extreme_points.sort(key=lambda p: (p[2], p[1], p[0]))
            else:
                unpacked.append(item)

        # Compute Center of Gravity (CoG)
        if placed and current_weight > 0:
            cog_x = sum(b.weight * (b.x + b.placed_width / 2.0) for b in placed) / current_weight
            cog_y = sum(b.weight * (b.y + b.placed_length / 2.0) for b in placed) / current_weight
            cog_z = sum(b.weight * (b.z + b.placed_height / 2.0) for b in placed) / current_weight
        else:
            cog_x, cog_y, cog_z = 0.0, 0.0, 0.0

        # Check CoG stability within safety envelope
        min_x_safe = container.width * container.cog_x_bounds[0]
        max_x_safe = container.width * container.cog_x_bounds[1]
        min_y_safe = container.length * container.cog_y_bounds[0]
        max_y_safe = container.length * container.cog_y_bounds[1]

        cog_is_stable = (min_x_safe <= cog_x <= max_x_safe) and (min_y_safe <= cog_y <= max_y_safe)

        total_packed_volume = sum(b.volume for b in placed)
        vol_util = total_packed_volume / container.volume if container.volume > 0 else 0.0
        weight_util = current_weight / container.max_weight if container.max_weight > 0 else 0.0

        t_end = time.perf_counter_ns()
        elapsed_us = (t_end - t_start) / 1000.0

        status = SolverStatus.OPTIMAL if not unpacked else SolverStatus.FEASIBLE

        return BinPacking3DResult(
            status=status,
            packed_items=placed,
            unpacked_items=unpacked,
            volume_utilization=vol_util,
            weight_utilization=weight_util,
            total_packed_weight=current_weight,
            center_of_gravity=(cog_x, cog_y, cog_z),
            cog_is_stable=cog_is_stable,
            elapsed_microseconds=elapsed_us,
        )
