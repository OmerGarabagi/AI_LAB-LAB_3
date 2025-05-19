"""
---------------------------------------------------------------
Clarke & Wright Savings-based solver for the Capacitated VRP
Author : Omer (AI-Lab Assignment 3 – Task 1 solution)
Python 3.9+, standard library only
---------------------------------------------------------------
The program exposes one public function:
    solve_cvrp(depot, customers, capacity)
where
    depot     = (x, y)                      – coordinates of warehouse (#0)
    customers = [(x, y, demand), …]        – list in the order 1..n
    capacity  = vehicle load limit (float or int)

Returned value
--------------
List of routes; every route is a list of vertex indices that
starts and ends with 0 (the depot).  Example:
    [[0, 4, 2, 0], [0, 3, 1, 5, 0]]
Total length can be recomputed with `route_length(route, dist)`.

Complexity
----------
Building the savings list :  Θ(n²)  
Sorting savings          :  Θ(n² log n)  
Merging routes           :  ≤ Θ(n²)     (worst-case)
Memory                   :  Θ(n²)

---------------------------------------------------------------"""
import math
from collections import defaultdict

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def euclidean(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def build_distance_matrix(nodes):
    """nodes = list of (x, y); returns symmetric |V|×|V| distance table"""
    n = len(nodes)
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = euclidean(nodes[i], nodes[j])
            dist[i][j] = dist[j][i] = d
    return dist

def route_load(route, demand):
    """Sum of customer demands on a route (skip the two depot zeros)."""
    return sum(demand[v] for v in route if v != 0)

def route_length(route, dist):
    """Euclidean length of a closed tour."""
    return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

# ------------------------------------------------------------------
# Correct, self-contained Clarke & Wright implementation
# ------------------------------------------------------------------
def clarke_wright(depot_xy, cust_xyz, capacity):
    """
    depot_xy : (x,y) of node 0
    cust_xyz : [(x,y,demand), …]  for nodes 1‥n
    capacity : vehicle capacity
    returns   : (list_of_routes, distance_matrix)
    """
    import math

    # ---------- helpers ----------
    def euclid(p, q):     # length of segment pq
        return math.hypot(p[0]-q[0], p[1]-q[1])

    def dist_matrix(pts):
        n = len(pts)
        d = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                dij = euclid(pts[i], pts[j])
                d[i][j] = d[j][i] = dij
        return d
    # -----------------------------

    coords  = [depot_xy] + [(x, y) for x, y, _ in cust_xyz]
    demand  = [0] + [d for _, _, d in cust_xyz]
    n       = len(coords)
    dist    = dist_matrix(coords)

    # ―― stage 1: one tiny route per customer ――
    routes      = {i: [0, i, 0] for i in range(1, n)}   # key = right end
    load        = {i: demand[i] for i in range(1, n)}
    route_of    = {i: i          for i in range(1, n)}   # cust → route-key

    # ―― stage 2: compute all savings ――
    savings = [(dist[0][i] + dist[0][j] - dist[i][j], i, j)
               for i in range(1, n) for j in range(i+1, n)]
    savings.sort(reverse=True)          # largest saving first

    # ―― greedy merging ――
    for s, i, j in savings:
        ri = route_of[i]          # route that currently ends with …i-0
        rj = route_of[j]          # route that starts   0-j…
        if ri == rj:
            continue              # already in same tour

        R_i = routes[ri]
        R_j = routes[rj]
        if R_i[-2] != i or R_j[1] != j:  # i not rightmost OR j not leftmost
            continue
        if load[ri] + load[rj] > capacity:
            continue              # would violate capacity

        # merge feasible → build the new tour
        merged     = R_i[:-1] + R_j[1:]
        new_key    = R_j[-2]                 # rightmost customer in new tour
        new_load   = load[ri] + load[rj]

        # delete the two old tours
        for k in (ri, rj):
            routes.pop(k)
            load.pop(k)

        routes[new_key] = merged
        load[new_key]   = new_load

        # update mapping for every customer now in the merged tour
        for c in merged[1:-1]:               # skip depot at both ends
            route_of[c] = new_key

    return list(routes.values()), dist

# ------------------------------------------------------------------
# Public convenience wrapper
# ------------------------------------------------------------------
def solve_cvrp(depot_xy, customers, capacity, pretty=False):
    """
    Wrapper that prints a summary when pretty=True.
    customers : list of (x, y, demand)
    """
    routes, dist = clarke_wright(depot_xy, customers, capacity)

    if not pretty:            # let callers skip console output
        return routes

    # pre-extract the demand list so we can compute route loads
    cust_demand = [d for _, _, d in customers]

    total_len = sum(route_length(r, dist) for r in routes)
    print(f"--- Clarke-Wright solution  |  vehicles used: {len(routes)}  ---")
    for k, r in enumerate(routes, 1):
        load = sum(cust_demand[i-1] for i in r if i != 0)   # i==0 → depot
        print(f"Route {k:<2}: {r}  | load = {load}")
    print(f"Total distance: {total_len:.2f}")
    return routes

# ------------------------------------------------------------------
# Basic demo with the textbook 4-city example (numbers as in the PDF)
# ------------------------------------------------------------------
if __name__ == "__main__":
    depot = (0, 0)
    # city: x, y, demand
    customers = [
        (10,  0, 4),   # 1
        (10, 10, 4),   # 2
        (10, 20, 4),   # 3
        ( 0, 20, 4)    # 4
    ]
    capacity = 10      # each truck
    solve_cvrp(depot, customers, capacity, pretty=True)
