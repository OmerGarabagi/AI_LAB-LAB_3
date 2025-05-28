#!/usr/bin/env python3

import sys
import os
import re


# -------------------------------------------------
# Data‑model classes for representing a CVRP state
# -------------------------------------------------

class Node:
    """Represents a vertex (customer or depot) in a CVRP instance."""
    __slots__ = ("id", "x", "y", "demand")

    def __init__(self, node_id: int, x: float, y: float, demand: int = 0):
        self.id = int(node_id)
        self.x = float(x)
        self.y = float(y)
        self.demand = int(demand)

    def distance_to(self, other: "Node") -> float:
        """Euclidean distance to another node."""
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5

    def __repr__(self):
        return f"Node({self.id}, ({self.x:.1f}, {self.y:.1f}), d={self.demand})"


class Vehicle:
    """A vehicle with fixed capacity and an ordered list of visited nodes."""
    __slots__ = ("capacity", "route")

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.route: list[Node] = []

    @property
    def load(self) -> int:
        return sum(n.demand for n in self.route)

    def can_add(self, node: Node) -> bool:
        return self.load + node.demand <= self.capacity

    def add_node(self, node: Node):
        if not self.can_add(node):
            raise ValueError("Cannot add node – capacity would be exceeded")
        self.route.append(node)

    def __repr__(self):
        ids = [n.id for n in self.route]
        return f"Vehicle(cap={self.capacity}, load={self.load}, route={ids})"


class Route:
    """A standalone route (helpful when vehicles are not explicitly modeled)."""
    __slots__ = ("nodes",)

    def __init__(self, nodes: list[Node] | None = None):
        self.nodes = nodes[:] if nodes else []

    def distance(self) -> float:
        if len(self.nodes) < 2:
            return 0.0
        return sum(
            self.nodes[i].distance_to(self.nodes[i + 1])
            for i in range(len(self.nodes) - 1)
        )

    def load(self) -> int:
        return sum(n.demand for n in self.nodes)

    def __repr__(self):
        ids = [n.id for n in self.nodes]
        return f"Route(len={len(self.nodes)}, dist={self.distance():.1f}, nodes={ids})"


class Solution:
    """Collection of routes representing a complete CVRP solution."""
    __slots__ = ("routes",)

    def __init__(self, routes: list[Route] | None = None):
        self.routes = routes[:] if routes else []

    def total_distance(self) -> float:
        return sum(r.distance() for r in self.routes)

    def total_load(self) -> int:
        return sum(r.load() for r in self.routes)

    def __repr__(self):
        return (
            f"Solution(routes={len(self.routes)}, "
            f"distance={self.total_distance():.1f}, "
            f"load={self.total_load()})"
        )


def parse_cvrp(filepath):
    """
    Parse a CVRP instance in TSPLIB format.

    Args:
        filepath (str | Path): path to the *.vrp file.

    Returns:
        tuple:
            num_vehicles (int or None) – number of trucks if specified in the COMMENT line,
            num_vertices (int) – total number of vertices (including the depot),
            capacity (int) – vehicle capacity,
            coords (list[tuple[float, float]]) – list of (x, y) coordinates ordered by vertex id,
            demands (list[int]) – list of demands ordered by vertex id.
    """
    num_vehicles = None
    num_vertices = None
    capacity = None

    coords_section = False
    demand_section = False

    coords = {}
    demands = {}

    with open(filepath, "r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue

            # ---- headers ----
            if line.startswith("COMMENT"):
                match = re.search(r"No of trucks:\s*(\d+)", line)
                if match:
                    num_vehicles = int(match.group(1))
            elif line.startswith("DIMENSION"):
                num_vertices = int(line.split(":")[1])
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1])

            # ---- section switches ----
            if line.startswith("NODE_COORD_SECTION"):
                coords_section = True
                demand_section = False
                continue
            elif line.startswith("DEMAND_SECTION"):
                coords_section = False
                demand_section = True
                continue
            elif line.startswith("DEPOT_SECTION"):
                # we can ignore the depot list for this basic parser
                break

            # ---- read section contents ----
            if coords_section:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x, y = map(float, parts[1:3])
                    coords[node_id] = (x, y)
            elif demand_section:
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    demands[node_id] = demand

    if num_vertices is None or capacity is None:
        raise ValueError("DIMENSION or CAPACITY not found in file header.")

    # Order coordinates and demands by node id
    ordered_ids = sorted(coords)
    coord_list = [coords[i] for i in ordered_ids]
    demand_list = [demands.get(i, 0) for i in ordered_ids]

    return num_vehicles, num_vertices, capacity, coord_list, demand_list


def main():
    print("=== CVRP File Parser ===")
    print("This program will parse a VRP (Vehicle Routing Problem) file.")
    print()
    
    # Get VRP file path from user input
    while True:
        vrp_path = input("Please enter the path to your VRP file: ").strip()
        
        if not vrp_path:
            print("Error: Please enter a valid file path.")
            continue
            
        # Check if file exists
        if not os.path.exists(vrp_path):
            print(f"Error: File '{vrp_path}' not found.")
            retry = input("Would you like to try again? (y/n): ").strip().lower()
            if retry not in ['y', 'yes']:
                print("Exiting program.")
                return
            continue
        
        # File exists, break out of loop
        break
    
    print(f"\nParsing VRP file: {vrp_path}")
    print("-" * 50)
    
    try:
        num_veh, n_vertices, cap, coords, demands = parse_cvrp(vrp_path)
    except Exception as exc:
        print(f"Failed to parse file: {exc}")
        return

    print("=== CVRP Instance Summary ===")
    print(f"File: {vrp_path}")
    print(f"Number of vehicles: {num_veh if num_veh else 'Not specified'}")
    print(f"Number of vertices: {n_vertices}")
    print(f"Vehicle capacity: {cap}")
    print(f"Total demand: {sum(demands)}")
    print()
    
    print("=== Coordinates (first 10) ===")
    for i, (x, y) in enumerate(coords[:10]):
        print(f"Vertex {i+1}: ({x:.2f}, {y:.2f}) - Demand: {demands[i]}")
    
    if len(coords) > 10:
        print(f"... and {len(coords) - 10} more vertices")
    
    print()
    print("=== Parsing Complete ===")
    print(f"Successfully parsed {n_vertices} vertices with total demand {sum(demands)}")
    
    # Optional: You can add your CVRP solving algorithm here
    # solve_cvrp(coords, demands, cap, num_veh)

__all__ = [
    "Node",
    "Vehicle",
    "Route",
    "Solution",
    "route_cost",
    "solution_cost",
    "parse_cvrp",
]
#
# -------------------------------------------------
# Cost utilities
# -------------------------------------------------
def route_cost(route: Route, capacity: int) -> float:
    """
    Return the distance of a route if it does not exceed capacity,
    otherwise raise ValueError.
    """
    load = route.load()
    if load > capacity:
        raise ValueError(
            f"Route load {load} exceeds vehicle capacity {capacity}"
        )
    return route.distance()


def solution_cost(solution: Solution, capacity: int) -> float:
    """
    Compute the total cost (sum of distances) of a solution,
    ensuring every route respects the vehicle capacity.
    """
    total = 0.0
    for r in solution.routes:
        total += route_cost(r, capacity)
    return total


if __name__ == "__main__":
    main()
    
