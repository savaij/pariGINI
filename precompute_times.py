"""
Precompute pairwise metro travel times between all graph nodes.

Outputs 4-matrix NPZ + comprehensive JSON metadata:
- times_min: full travel time (metro + penalties)
- metro_time: metro-only time (no change penalties)
- line_changes: number of line changes per route
- distinct_lines: number of distinct metro lines per route
- nodes: node name list

JSON metadata includes node coordinates, parameters, and timestamps.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import time

import geopandas as gpd
import numpy as np

from gini_paris_distances_calculations import (
    build_graph_from_edgelist,
    shortest_path_max_1_change,
    MAX_LINE_CHANGES,
    CHANGE_PENALTY_MIN,
    WALK_SPEED_M_PER_MIN,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute all-pairs metro travel times.")
    parser.add_argument(
        "--edgelist",
        default="./timed_edgelist.geojson",
        help="Path to timed edgelist GeoJSON.",
    )
    parser.add_argument(
        "--max-line-changes",
        type=int,
        default=int(MAX_LINE_CHANGES),
        help="Maximum number of line changes.",
    )
    parser.add_argument(
        "--change-penalty-min",
        type=float,
        default=float(CHANGE_PENALTY_MIN),
        help="Penalty (minutes) per line change.",
    )
    parser.add_argument(
        "--walk-speed-m-per-min",
        type=float,
        default=float(WALK_SPEED_M_PER_MIN),
        help="Walking speed in meters per minute.",
    )
    parser.add_argument(
        "--max-walk-min-start",
        type=float,
        default=15.0,
        help="Max walking time from origin (minutes).",
    )
    parser.add_argument(
        "--max-walk-min-end",
        type=float,
        default=15.0,
        help="Max walking time to destination (minutes).",
    )
    parser.add_argument(
        "--max-candidate-stations",
        type=int,
        default=25,
        help="Max candidate stations per origin/destination.",
    )
    parser.add_argument(
        "--out",
        default="./precomputed_node_times.npz",
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--meta",
        default="./precomputed_node_times_meta.json",
        help="Output JSON metadata path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data sources.
    graph = build_graph_from_edgelist(args.edgelist)

    nodes = list(graph.nodes())
    n_nodes = len(nodes)

    # Initialize 4 matrices for comprehensive precomputation
    times_min = np.full((n_nodes, n_nodes), np.nan, dtype=float)
    metro_time = np.full((n_nodes, n_nodes), np.nan, dtype=float)
    line_changes = np.full((n_nodes, n_nodes), np.nan, dtype=float)
    distinct_lines = np.full((n_nodes, n_nodes), np.nan, dtype=float)
    
    np.fill_diagonal(times_min, 0.0)
    np.fill_diagonal(metro_time, 0.0)
    np.fill_diagonal(line_changes, 0.0)
    np.fill_diagonal(distinct_lines, 0.0)

    # Extract node coordinates for metadata
    node_coords = {}
    for node in nodes:
        coords = graph.nodes[node].get("coordinates")
        if coords:
            node_coords[node] = {"lon": float(coords[0]), "lat": float(coords[1])}

    start_ts = time()
    failures = 0
    failed_pairs = []

    print(f"Computing {n_nodes}x{n_nodes}={n_nodes*n_nodes} pairs (symmetric)...")
    
    for i in range(n_nodes):
        if i % max(1, n_nodes // 10) == 0:
            print(f"  Progress: {i}/{n_nodes}")
            
        src = nodes[i]
        for j in range(i + 1, n_nodes):
            dst = nodes[j]
            try:
                path_nodes, edge_records, metro_time_val, lines_seq, line_changes_val, total_cost = shortest_path_max_1_change(
                    graph,
                    src,
                    dst,
                    max_changes=args.max_line_changes,
                    change_penalty_min=args.change_penalty_min,
                )
                t_min = float(total_cost)
                m_time = float(metro_time_val)
                n_changes = int(line_changes_val)
                n_distinct = len(set(lines_seq)) if lines_seq else 0
                
            except Exception as e:
                t_min = np.nan
                m_time = np.nan
                n_changes = np.nan
                n_distinct = np.nan
                failures += 1
                failed_pairs.append({"src": src, "dst": dst, "error": str(e)})

            times_min[i, j] = t_min
            times_min[j, i] = t_min
            
            metro_time[i, j] = m_time
            metro_time[j, i] = m_time
            
            line_changes[i, j] = n_changes
            line_changes[j, i] = n_changes
            
            distinct_lines[i, j] = n_distinct
            distinct_lines[j, i] = n_distinct

    elapsed = time() - start_ts
    
    out_path = Path(args.out)
    meta_path = Path(args.meta)

    # Save 4-matrix NPZ
    np.savez_compressed(
        out_path,
        nodes=np.array(nodes, dtype=object),
        times_min=times_min,
        metro_time=metro_time,
        line_changes=line_changes,
        distinct_lines=distinct_lines,
    )

    # Build comprehensive node_list for metadata
    node_list = []
    for idx, node in enumerate(nodes):
        coords = node_coords.get(node, {})
        node_list.append({
            "id": idx,
            "name": str(node),
            "lon": coords.get("lon"),
            "lat": coords.get("lat"),
        })

    # Comprehensive metadata
    meta = {
        "computed_at": datetime.now().isoformat(),
        "edgelist": str(Path(args.edgelist)),
        "n_nodes": int(n_nodes),
        "node_list": node_list,
        "parameters": {
            "max_line_changes": int(args.max_line_changes),
            "change_penalty_min": float(args.change_penalty_min),
            "walk_speed_m_per_min": float(args.walk_speed_m_per_min),
            "max_walk_min_start": float(args.max_walk_min_start),
            "max_walk_min_end": float(args.max_walk_min_end),
            "max_candidate_stations": int(args.max_candidate_stations),
        },
        "computation_stats": {
            "failures": int(failures),
            "failed_pairs_count": len(failed_pairs),
            "elapsed_sec": float(elapsed),
            "pairs_computed": int(n_nodes * (n_nodes - 1) / 2),
        },
        "npz_matrices": {
            "times_min": "Total travel time with penalties (minutes)",
            "metro_time": "Metro-only time, no walking (minutes)",
            "line_changes": "Number of line changes per route",
            "distinct_lines": "Number of distinct metro lines per route",
            "nodes": "Array of node names/IDs",
        },
        "output_files": {
            "npz": str(out_path),
            "json": str(meta_path),
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    print(f"\n✓ Precomputation complete!")
    print(f"  NPZ: {out_path}")
    print(f"  JSON: {meta_path}")
    print(f"  Elapsed: {elapsed:.1f}s, Failures: {failures}/{n_nodes*(n_nodes-1)//2}")


if __name__ == "__main__":
    main()
