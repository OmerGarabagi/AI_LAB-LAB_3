#!/usr/bin/env python3

import sys
import os
import re


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

__all__ = ["parse_cvrp"]


if __name__ == "__main__":
    main()
    
