#!/usr/bin/env python3

import sys
import os
import re
import math
from typing import Union, Optional
import random
import itertools
from collections import deque

import unittest


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

    def __init__(self, nodes: Optional[list[Node]] = None):
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

    def __init__(self, routes: Optional[list[Route]] = None):
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
    
    # Demonstrate baseline heuristic
    baseline = greedy_nearest_neighbor(coords, demands, cap)
    print("=== Baseline Greedy Solution ===")
    print(f"Total routes : {len(baseline.routes)}")
    print(f"Total dist.  : {baseline.total_distance():.2f}")
    for idx, r in enumerate(baseline.routes, 1):
        ids = [n.id for n in r.nodes]
        print(f" Route {idx}: {ids}  (load {r.load()})")

    print("\n=== Multi‑Stage (Clarke‑Wright + Relocate) Solution ===")
    ms_solution = multistage_clarke_wright(coords, demands, cap)
    print(f"Total routes : {len(ms_solution.routes)}")
    print(f"Total dist.  : {ms_solution.total_distance():.2f}")
    for idx, r in enumerate(ms_solution.routes, 1):
        ids = [n.id for n in r.nodes]
        print(f" Route {idx}: {ids}  (load {r.load()})")

    print("\n=== ILS‑Tabu (30 iterations) ===")
    ils_sol = ils_tabu(coords, demands, cap, n_iters=30, seed=42)
    print(f"Total dist. : {ils_sol.total_distance():.2f}")

__all__ = [
    "Node",
    "Vehicle",
    "Route",
    "Solution",
    "route_cost",
    "solution_cost",
    "parse_cvrp",
    "greedy_nearest_neighbor",
    "ackley",
    "multistage_clarke_wright",
    "ils_tabu",
]
# -------------------------------------------------
# Iterated Local Search (ILS) with Tabu‑Search core
# -------------------------------------------------
def ils_tabu(
    coords: list[tuple[float, float]],
    demands: list[int],
    capacity: int,
    n_iters: int = 30,
    tabu_tenure: int = 15,
    ls_iters: int = 250,
    seed: int | None = None,
) -> Solution:
    """
    ILS framework:
        • Initial solution: greedy_nearest_neighbor.
        • repeat n_iters times:
              1. Perturb best solution using a random INTER‑ROUTE swap.
              2. Apply Tabu Search local search to the perturbed solution.
              3. Accept if better than current best.
    Perturbation promotes diversification; TS provides intensification.

    Complexity:
        – Each TS run explores O(ls_iters · n²) neighborhood checks in worst case
          (we sample candidate SWAP moves only once per iteration).
        – Overall ≈ O(n_iters · ls_iters · n²) but with modest constants.
    """
    rng = random.Random(seed)

    # ---------- helpers ----------
    def copy_solution(sol: Solution) -> Solution:
        return Solution([Route(nodes=r.nodes[:]) for r in sol.routes])

    def random_swap_between_routes(sol: Solution) -> bool:
        customers = []
        for r_idx, r in enumerate(sol.routes):
            for n_idx, node in enumerate(r.nodes[1:-1], 1):
                customers.append((r_idx, n_idx))
        if len(customers) < 2:
            return False
        (r1, i1), (r2, i2) = rng.sample(customers, 2)
        if r1 == r2:
            return False
        sol.routes[r1].nodes[i1], sol.routes[r2].nodes[i2] = (
            sol.routes[r2].nodes[i2],
            sol.routes[r1].nodes[i1],
        )
        return True

    # ---------- Tabu Search core ----------
    def tabu_search(start: Solution) -> Solution:
        current = copy_solution(start)
        best = copy_solution(current)
        best_cost = best.total_distance()
        tabu: deque[tuple[int, int]] = deque(maxlen=tabu_tenure)

        for _ in range(ls_iters):
            best_neighbor = None
            best_move = None
            best_delta = float("inf")

            # collect customers indexes once
            customers = []
            for r_idx, r in enumerate(current.routes):
                for n_idx, node in enumerate(r.nodes[1:-1], 1):
                    customers.append((r_idx, n_idx, node))

            for (r1, i1, n1), (r2, i2, n2) in itertools.combinations(
                customers, 2
            ):
                if r1 == r2 or (n1.id, n2.id) in tabu:
                    continue

                # evaluate swap
                delta = (
                    _swap_delta(current.routes[r1], i1, n2, capacity)
                    + _swap_delta(current.routes[r2], i2, n1, capacity)
                )
                if delta is None:
                    continue  # infeasible capacity
                if delta < best_delta:
                    best_delta = delta
                    best_neighbor = (r1, i1, r2, i2)
                    best_move = (n1.id, n2.id)

            if best_neighbor is None or best_delta >= -1e-6:
                break  # no improving move

            r1, i1, r2, i2 = best_neighbor
            current.routes[r1].nodes[i1], current.routes[r2].nodes[i2] = (
                current.routes[r2].nodes[i2],
                current.routes[r1].nodes[i1],
            )
            tabu.append(best_move)

            cur_cost = current.total_distance()
            if cur_cost < best_cost:
                best_cost = cur_cost
                best = copy_solution(current)

        return best

    # Small helper to compute cost delta of swapping one node into a position
    def _swap_delta(route: Route, idx: int, new_node: "Node", cap: int):
        """Return cost delta or None if capacity violated."""
        load_without = route.load() - route.nodes[idx].demand + new_node.demand
        if load_without > cap:
            return None
        prev = route.nodes[idx - 1]
        nxt = route.nodes[idx + 1]
        old_node = route.nodes[idx]
        old_cost = prev.distance_to(old_node) + old_node.distance_to(nxt)
        new_cost = prev.distance_to(new_node) + new_node.distance_to(nxt)
        return new_cost - old_cost

    # ---------- ILS main ----------
    best = greedy_nearest_neighbor(coords, demands, capacity)
    best_cost = best.total_distance()

    for _ in range(n_iters):
        cand = copy_solution(best)
        if not random_swap_between_routes(cand):
            continue
        improved = tabu_search(cand)
        cost = improved.total_distance()
        if cost < best_cost - 1e-6:
            best, best_cost = improved, cost

    return best


# -------------------------------------------------
# Ackley benchmark function (d‑dimensional)
# -------------------------------------------------
def ackley(
    x: list[float] | tuple[float, ...],
    a: float = 20.0,
    b: float = 0.2,
    c: float = 2 * math.pi,
) -> float:
    """
    Compute the Ackley function value for a point x.

    Args:
        x (list[float] | tuple[float]): point in R^d.
        a, b, c (float): Ackley parameters (default 20, 0.2, 2π).

    Returns:
        float: f(x); the global optimum is 0 at x = (0,…,0).
    """
    d = len(x)
    if d == 0:
        raise ValueError("Ackley input vector must have positive length")

    sum_sq = 0.0
    sum_cos = 0.0
    for xi in x:
        sum_sq += xi * xi
        sum_cos += math.cos(c * xi)

    term1 = -a * math.exp(-b * math.sqrt(sum_sq / d))
    term2 = -math.exp(sum_cos / d)
    return term1 + term2 + a + math.e

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


# -------------------------------------------------
# Baseline Greedy – Nearest‑Neighbor with capacity
# -------------------------------------------------
def greedy_nearest_neighbor(
    coords: list[tuple[float, float]],
    demands: list[int],
    capacity: int,
) -> Solution:
    """
    Produce a baseline CVRP Solution using a nearest‑neighbor heuristic.

    Strategy:
    1. Build Node objects (id 0 is the depot).
    2. While there are unserved customers:
       * Start a new route at the depot with an empty load.
       * Iteratively pick the closest still‑unserved customer that fits
         in the remaining capacity.
       * When no feasible customer can be added, return to the depot and
         close the route.
    3. Repeat until all customers are routed.

    Args:
        coords (list[(x,y)]): coordinates of all vertices, depot first.
        demands (list[int]): demand of each vertex, depot first (must be 0).
        capacity (int): vehicle capacity.

    Returns:
        Solution: collection of routes found by the heuristic.
    """
    # --- build Node list ---
    nodes = [Node(i, *coords[i], demands[i]) for i in range(len(coords))]
    depot = nodes[0]
    customers = nodes[1:]  # exclude depot

    unrouted = set(c.id for c in customers)
    routes: list[Route] = []

    while unrouted:
        current_load = 0
        route_nodes = [depot]  # always start at depot
        current = depot

        while True:
            # find nearest feasible customer
            next_id = None
            best_dist = float("inf")
            for cid in unrouted:
                cust = nodes[cid]
                if current_load + cust.demand > capacity:
                    continue
                dist = current.distance_to(cust)
                if dist < best_dist:
                    best_dist = dist
                    next_id = cid
            if next_id is None:
                break  # cannot add more customers to this route

            # add the chosen customer
            cust = nodes[next_id]
            route_nodes.append(cust)
            current_load += cust.demand
            current = cust
            unrouted.remove(next_id)

        # return to depot
        route_nodes.append(depot)
        routes.append(Route(route_nodes))

    return Solution(routes)


# -------------------------------------------------
# Multi‑Stage Heuristic – Clarke‑Wright + Relocate
# -------------------------------------------------
def multistage_clarke_wright(
    coords: list[tuple[float, float]],
    demands: list[int],
    capacity: int,
) -> Solution:
    n = len(coords)
    nodes = [Node(i, *coords[i], demands[i]) for i in range(n)]
    depot = nodes[0]

    # --- Stage 1: Clarke–Wright parallel savings ---
    # Start with one route per customer (depot‑i‑depot)
    routes: dict[int, Route] = {
        i: Route([depot, nodes[i], depot]) for i in range(1, n)
    }

    # Compute savings S_ij
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = (
                depot.distance_to(nodes[i])
                + depot.distance_to(nodes[j])
                - nodes[i].distance_to(nodes[j])
            )
            savings.append((s, i, j))
    savings.sort(reverse=True)  # descending

    # Helper maps: which route a customer currently belongs to,
    # whether customer is at start/end (needed for feasible merge).
    route_of = {i: i for i in range(1, n)}

    for s, i, j in savings:
        ri_id = route_of[i]
        rj_id = route_of[j]
        if ri_id == rj_id:
            continue  # already in same route

        ri = routes[ri_id]
        rj = routes[rj_id]

        # We can only merge if i is at route end of ri and j at route start of rj
        if (
            ri.nodes[-2].id != i
            or rj.nodes[1].id != j
        ):
            # try the opposite direction
            if ri.nodes[1].id != i or rj.nodes[-2].id != j:
                continue

            # flip rj to have j at start
            rj.nodes = [depot] + rj.nodes[1:-1][::-1] + [depot]

        # Capacity check
        if ri.load() + rj.load() > capacity:
            continue

        # Merge rj after ri (excluding duplicate depots)
        merged_nodes = ri.nodes[:-1] + rj.nodes[1:]
        new_route = Route(merged_nodes)

        # Replace routes
        new_id = ri_id
        routes[new_id] = new_route
        del routes[rj_id]

        # Update mapping
        for node in new_route.nodes[1:-1]:
            route_of[node.id] = new_id

    initial_solution = Solution(list(routes.values()))

    # --- Stage 2: single‑pass relocate improvement ---
    improved = True
    while improved:
        improved = False
        for r_from in initial_solution.routes:
            # skip depot only routes
            if len(r_from.nodes) <= 3:
                continue
            for customer_idx in range(1, len(r_from.nodes) - 1):
                cust = r_from.nodes[customer_idx]
                for r_to in initial_solution.routes:
                    if r_to is r_from:
                        continue
                    if r_to.load() + cust.demand > capacity:
                        continue

                    # try inserting cust at best position in r_to
                    best_pos = None
                    best_delta = 0.0
                    for insert_idx in range(1, len(r_to.nodes)):
                        prev = r_to.nodes[insert_idx - 1]
                        nxt = r_to.nodes[insert_idx]
                        delta = (
                            prev.distance_to(cust)
                            + cust.distance_to(nxt)
                            - prev.distance_to(nxt)
                        )
                        if best_pos is None or delta < best_delta:
                            best_pos = insert_idx
                            best_delta = delta

                    # cost change in r_from (remove cust)
                    prev_f = r_from.nodes[customer_idx - 1]
                    nxt_f = r_from.nodes[customer_idx + 1]
                    delta_from = (
                        prev_f.distance_to(nxt_f)
                        - prev_f.distance_to(cust)
                        - cust.distance_to(nxt_f)
                    )

                    if best_pos is not None and best_delta + delta_from < -1e-6:
                        # perform relocate
                        r_from.nodes.pop(customer_idx)
                        r_to.nodes.insert(best_pos, cust)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return initial_solution


# -------------------------------------------------
# Unit tests – run with:  python -m unittest Solution
# -------------------------------------------------
class TestCostAndCapacity(unittest.TestCase):
    """Verify that cost utilities respect Euclidean distance and capacity."""

    def setUp(self):
        # simple coordinate system:
        # depot (0,0), customer1 at (3,4) --> distance 5
        # customer2 at (6,8)             --> distance 10 from depot
        self.depot = Node(0, 0, 0, 0)
        self.c1 = Node(1, 3, 4, 4)   # demand 4
        self.c2 = Node(2, 6, 8, 5)   # demand 5
        self.capacity_ok = 10
        self.capacity_small = 8

    # ---------- route_cost ----------

    def test_route_cost_valid_distance(self):
        """route_cost returns correct distance when within capacity."""
        r = Route([self.depot, self.c1, self.c2, self.depot])
        expected = (
            self.depot.distance_to(self.c1)
            + self.c1.distance_to(self.c2)
            + self.c2.distance_to(self.depot)
        )
        self.assertAlmostEqual(route_cost(r, self.capacity_ok), expected, places=6)

    def test_route_cost_capacity_violation(self):
        """route_cost raises ValueError if load exceeds capacity."""
        r = Route([self.depot, self.c1, self.c2, self.depot])  # load = 9
        with self.assertRaises(ValueError):
            route_cost(r, self.capacity_small)

    # ---------- solution_cost ----------

    def test_solution_cost_sum(self):
        """solution_cost is the sum of route_cost over all routes."""
        r1 = Route([self.depot, self.c1, self.depot])
        r2 = Route([self.depot, self.c2, self.depot])
        sol = Solution([r1, r2])

        expected = route_cost(r1, self.capacity_ok) + route_cost(
            r2, self.capacity_ok
        )
        self.assertAlmostEqual(solution_cost(sol, self.capacity_ok), expected, places=6)

    def test_solution_cost_propagates_capacity_error(self):
        """solution_cost propagates capacity violations from any route."""
        heavy = Node(3, 1, 1, 20)
        bad_route = Route([self.depot, heavy, self.depot])
        sol = Solution([bad_route])
        with self.assertRaises(ValueError):
            solution_cost(sol, self.capacity_ok)

    # (end TestCostAndCapacity)

    # Optionally, more tests could be added here.


if __name__ == "__main__":
    main()
    # Quick Ackley test (d=10)
    zero_vec = [0.0] * 10
    print("\nAckley(0-vector) =", ackley(zero_vec))
