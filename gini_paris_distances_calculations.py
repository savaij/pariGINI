# =========================
# METRO PARIS ROUTING & INEQUALITY METRICS
# =========================
# Aggiornamenti principali:
# - NON carica il grafo all'import (niente side effects)
# - Correzione velocità cammino: 4 km/h = 4000/60 m/min
# - Routing: invece di snappare SOLO alla stazione più vicina, prova TUTTE le stazioni
#   entro un range di camminata (es. 15 min) sia per start che per target, poi sceglie il migliore
# - Opzionale: considera anche "walk-only" (camminata diretta) e prende il minimo
# - Ritorna dettagli aggiuntivi in info["candidates"] per debug/performance

# --- Imports ---
import heapq
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point

# --- Constants ---
WGS84_EPSG = 4326
METRIC_EPSG = 2154  # Lambert-93 (meters)

# 4 km/h = 4000 m / 60 min
WALK_SPEED_M_PER_MIN = 4000 / 60

CHANGE_PENALTY_MIN = 3.0          # penalità per cambio linea
MAX_LINE_CHANGES = 1              # <= 1 cambio => max 2 linee
START_LINE_PENALTY_MIN = 0.0      # penalità per inizio linea


# =========================
# GRAPH INITIALIZATION
# =========================

def build_graph_from_edgelist(geojson_path: str) -> nx.MultiGraph:
    """Carica edgelist geojson e costruisce un MultiGraph con stazioni come nodi."""
    timed_edges_gdf = gpd.read_file(geojson_path)

    # Ensure WGS84 CRS
    g = timed_edges_gdf.copy()
    if g.crs is not None and g.crs.to_epsg() != WGS84_EPSG:
        g = g.to_crs(epsg=WGS84_EPSG)

    # Accumula coordinate di inizio/fine per ogni stazione
    coords_acc = defaultdict(list)
    for _, r in g.iterrows():
        coords = list(r.geometry.coords)
        coords_acc[r["from_station"]].append(coords[0])
        coords_acc[r["to_station"]].append(coords[-1])

    # Calcola coordinate finali per ogni nodo (mediana)
    station_coords = {}
    for st, pts in coords_acc.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        station_coords[st] = (float(np.median(xs)), float(np.median(ys)))

    # Costruisci grafo
    G = nx.MultiGraph()

    # Aggiungi nodi con attributi
    for name, coord in station_coords.items():
        G.add_node(name, name=name, coordinates=coord)

    # Aggiungi archi con attributi
    for _, r in g.iterrows():
        u = r["from_station"]
        v = r["to_station"]
        G.add_edge(
            u,
            v,
            line=r.get("line", r.get("line_norm")),
            time_min=(float(r["time_min"]) if pd.notnull(r.get("time_min")) else None),
            length_m=(float(r["length_m"]) if pd.notnull(r.get("length_m")) else None),
        )

    return G


# =========================
# ROUTING HELPERS
# =========================

def _node_lonlat(G, n):
    """Estrai coordinate (lon, lat) di un nodo."""
    if "coordinates" not in G.nodes[n]:
        raise KeyError(f"Nodo {n} senza attributo 'coordinates'")
    lon, lat = G.nodes[n]["coordinates"]
    return float(lon), float(lat)


def build_node_index(G, metric_epsg=METRIC_EPSG):
    """Costruisci indice spaziale dei nodi in coordinate metriche."""
    nodes = list(G.nodes())
    pts = [Point(_node_lonlat(G, n)) for n in nodes]
    nodes_gdf = gpd.GeoDataFrame({"node": nodes}, geometry=pts, crs=f"EPSG:{WGS84_EPSG}")
    nodes_m = nodes_gdf.to_crs(epsg=metric_epsg)
    # sindex verrà costruito on-demand (geopandas/shapely)
    return {"nodes_m": nodes_m, "metric_epsg": metric_epsg}


def snap_point_to_nearest_node(lonlat, node_index):
    """Snap un punto (lon, lat) al nodo della rete più vicino."""
    metric_epsg = node_index["metric_epsg"]
    nodes_m = node_index["nodes_m"]

    p = gpd.GeoDataFrame({"_": [0]}, geometry=[Point(lonlat)], crs=f"EPSG:{WGS84_EPSG}").to_crs(epsg=metric_epsg)
    j = gpd.sjoin_nearest(p, nodes_m[["node", "geometry"]], how="left", distance_col="dist_m")
    nearest_node = j.loc[0, "node"]
    dist_m = float(j.loc[0, "dist_m"])
    return nearest_node, dist_m


def candidate_nodes_within_walk_minutes(
    lonlat,
    node_index,
    walk_speed_m_per_min=WALK_SPEED_M_PER_MIN,
    max_walk_min=15.0,
    max_candidates=25,
):
    """
    Ritorna lista di candidati [(node, dist_m), ...] entro max_walk_min.
    Usa l'indice spaziale per filtrare rapidamente.
    """
    metric_epsg = node_index["metric_epsg"]
    nodes_m = node_index["nodes_m"]

    max_dist_m = float(walk_speed_m_per_min) * float(max_walk_min)
    if max_dist_m <= 0:
        return []

    p = gpd.GeoSeries([Point(lonlat)], crs=f"EPSG:{WGS84_EPSG}").to_crs(epsg=metric_epsg).iloc[0]
    buf = p.buffer(max_dist_m)

    # candidati via bounding box
    idxs = list(nodes_m.sindex.intersection(buf.bounds))
    if not idxs:
        return []

    sub = nodes_m.iloc[idxs].copy()
    sub["dist_m"] = sub.geometry.distance(p)
    sub = sub[sub["dist_m"] <= max_dist_m].sort_values("dist_m")

    if max_candidates is not None:
        sub = sub.head(int(max_candidates))

    return [(row["node"], float(row["dist_m"])) for _, row in sub.iterrows()]


def direct_walk_time_min(
    a_lonlat,
    b_lonlat,
    walk_speed_m_per_min=WALK_SPEED_M_PER_MIN,
    metric_epsg=METRIC_EPSG
):
    """Tempo camminata DIRETTA tra due lon/lat (minuti, dist_m)."""
    g = gpd.GeoSeries([Point(a_lonlat), Point(b_lonlat)], crs=f"EPSG:{WGS84_EPSG}").to_crs(epsg=metric_epsg)
    dist_m = float(g.iloc[0].distance(g.iloc[1]))
    return float(dist_m / float(walk_speed_m_per_min)), dist_m


# =========================
# SHORTEST PATH WITH LINE CHANGES
# =========================

def shortest_path_max_1_change(
    G,
    source,
    target,
    max_changes=MAX_LINE_CHANGES,
    change_penalty_min=CHANGE_PENALTY_MIN,
    start_line_penalty_min=START_LINE_PENALTY_MIN
):
    """
    Dijkstra su stato espanso: (node, current_line, changes_used).

    Permette al massimo max_changes cambi di linea.
    Ritorna: path_nodes, edge_records, metro_time_min, lines_sequence, line_changes, total_cost
    """
    dist = {}
    prev = {}

    start_state = (source, None, 0)
    dist[start_state] = 0.0
    pq = [(0.0, start_state)]

    best_end_state = None
    best_end_cost = np.inf

    while pq:
        cost, state = heapq.heappop(pq)
        if cost != dist.get(state, np.inf):
            continue

        node, cur_line, k = state

        if node == target:
            best_end_state = state
            best_end_cost = cost
            break

        for nbr in G.neighbors(node):
            edge_dict = G.get_edge_data(node, nbr)
            if edge_dict is None:
                continue

            for key, attrs in edge_dict.items():
                t = attrs.get("time_min", None)
                line = attrs.get("line", None)
                if t is None or line is None:
                    continue

                t = float(t)
                line = str(line)

                new_k = k
                extra_penalty = 0.0

                if cur_line is None:
                    extra_penalty += float(start_line_penalty_min)
                elif cur_line != line:
                    new_k = k + 1
                    if new_k > max_changes:
                        continue
                    extra_penalty += float(change_penalty_min)

                new_state = (nbr, line, new_k)
                new_cost = cost + t + extra_penalty

                if new_cost < dist.get(new_state, np.inf):
                    dist[new_state] = new_cost
                    prev[new_state] = (
                        state,
                        {
                            "u": node,
                            "v": nbr,
                            "key": key,
                            "time_min": t,
                            "line": line,
                            "length_m": attrs.get("length_m", None),
                            "change_penalty_applied": float(extra_penalty),
                        },
                    )
                    heapq.heappush(pq, (new_cost, new_state))

    if best_end_state is None:
        raise nx.NetworkXNoPath(f"Nessun percorso con ≤ {max_changes} cambi linea tra {source} e {target}")

    # Reconstruct
    edge_records_rev = []
    s = best_end_state
    while s != start_state:
        s_prev, edge_info = prev[s]
        edge_records_rev.append(edge_info)
        s = s_prev
    edge_records = list(reversed(edge_records_rev))

    path_nodes = [source]
    for e in edge_records:
        path_nodes.append(e["v"])

    lines_seq = [e["line"] for e in edge_records]
    line_changes = sum(1 for a, b in zip(lines_seq[:-1], lines_seq[1:]) if a != b)
    metro_time_min = float(sum(e["time_min"] for e in edge_records))

    return path_nodes, edge_records, metro_time_min, lines_seq, int(line_changes), float(best_end_cost)


# =========================
# ROUTE TIME (METRO + WALK) WITH WALK RANGE CANDIDATES
# =========================

def route_time_minutes(
    G,
    start_lonlat,
    end_lonlat,
    node_index=None,
    walk_speed_m_per_min=WALK_SPEED_M_PER_MIN,
    max_line_changes=MAX_LINE_CHANGES,
    change_penalty_min=CHANGE_PENALTY_MIN,
    max_walk_min_start=15.0,
    max_walk_min_end=15.0,
    max_candidate_stations=25,
    allow_walk_only=True,
):
    """
    Calcola tempo da start a end includendo camminata + metro con vincolo cambi.

    Miglioria:
    - considera tutte le stazioni raggiungibili entro max_walk_min_start e max_walk_min_end
      (cap max_candidate_stations), poi sceglie la combinazione migliore.
    - salva dettagli separati di cammino start/end: walk_time_start_min, walk_time_end_min
      e relative distanze: walk_dist_start_m, walk_dist_end_m
    - opzionale: allow_walk_only prova anche camminata diretta e prende il minimo.

    Ritorna: (best_total_time_min, details_dict)
    """
    if node_index is None:
        node_index = build_node_index(G)

    # Candidate start nodes
    start_candidates = candidate_nodes_within_walk_minutes(
        start_lonlat,
        node_index,
        walk_speed_m_per_min=walk_speed_m_per_min,
        max_walk_min=max_walk_min_start,
        max_candidates=max_candidate_stations,
    )
    if not start_candidates:
        u0, d0_m = snap_point_to_nearest_node(start_lonlat, node_index)
        start_candidates = [(u0, float(d0_m))]

    # Candidate end nodes
    end_candidates = candidate_nodes_within_walk_minutes(
        end_lonlat,
        node_index,
        walk_speed_m_per_min=walk_speed_m_per_min,
        max_walk_min=max_walk_min_end,
        max_candidates=max_candidate_stations,
    )
    if not end_candidates:
        u1, d1_m = snap_point_to_nearest_node(end_lonlat, node_index)
        end_candidates = [(u1, float(d1_m))]

    best_total = np.inf
    best_pack = None
    tried = 0
    pairs_with_path = 0

    for u0, d0_m in start_candidates:
        for u1, d1_m in end_candidates:
            tried += 1
            try:
                path_nodes, edges, metro_time_min, lines_seq, line_changes, cost_with_penalties = shortest_path_max_1_change(
                    G,
                    u0,
                    u1,
                    max_changes=max_line_changes,
                    change_penalty_min=change_penalty_min,
                )
                pairs_with_path += 1

                # --- walk start/end separati ---
                walk_dist_start_m = float(d0_m)
                walk_dist_end_m = float(d1_m)
                walk_dist_m = float(walk_dist_start_m + walk_dist_end_m)

                walk_time_start_min = float(walk_dist_start_m / float(walk_speed_m_per_min))
                walk_time_end_min = float(walk_dist_end_m / float(walk_speed_m_per_min))
                walk_time_min = float(walk_time_start_min + walk_time_end_min)

                total_time_min = float(cost_with_penalties + walk_time_min)

                if total_time_min < best_total:
                    best_total = total_time_min
                    best_pack = {
                        "mode": "metro_walk",
                        "snapped_start_node": u0,
                        "snapped_end_node": u1,
                        "snap_start_dist_m": float(d0_m),
                        "snap_end_dist_m": float(d1_m),

                        # ✅ nuovi dettagli
                        "walk_dist_start_m": float(walk_dist_start_m),
                        "walk_dist_end_m": float(walk_dist_end_m),
                        "walk_time_start_min": float(walk_time_start_min),
                        "walk_time_end_min": float(walk_time_end_min),

                        # totali
                        "walk_dist_m_total": float(walk_dist_m),
                        "walk_time_min": float(walk_time_min),

                        "path_nodes": path_nodes,
                        "edges": edges,
                        "metro_time_min": float(metro_time_min),
                        "lines_sequence": list(lines_seq),
                        "distinct_lines": sorted(set(lines_seq)),
                        "line_changes": int(line_changes),
                        "change_penalty_min": float(change_penalty_min),
                        "max_line_changes": int(max_line_changes),
                        "penalty_paid_in_path_min": float(cost_with_penalties - metro_time_min),
                        "total_time_min": float(total_time_min),
                    }
            except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, ValueError):
                continue

    walk_only_time = None
    walk_only_dist_m = None
    if allow_walk_only:
        walk_only_time, walk_only_dist_m = direct_walk_time_min(
            start_lonlat, end_lonlat, walk_speed_m_per_min=walk_speed_m_per_min, metric_epsg=METRIC_EPSG
        )
        if walk_only_time < best_total:
            return float(walk_only_time), {
                "mode": "walk_only",
                "start_input_lonlat": tuple(map(float, start_lonlat)),
                "end_input_lonlat": tuple(map(float, end_lonlat)),

                # ✅ nuovi dettagli (tutto su "start")
                "walk_dist_start_m": float(walk_only_dist_m),
                "walk_dist_end_m": 0.0,
                "walk_time_start_min": float(walk_only_time),
                "walk_time_end_min": 0.0,

                # totali
                "walk_dist_m_total": float(walk_only_dist_m),
                "walk_time_min": float(walk_only_time),

                "metro_time_min": 0.0,
                "penalty_paid_in_path_min": 0.0,
                "total_time_min": float(walk_only_time),
                "line_changes": 0,
                "distinct_lines": [],
                "lines_sequence": [],
                "path_nodes": [],
                "edges": [],
                "candidates": {
                    "start_candidates": len(start_candidates),
                    "end_candidates": len(end_candidates),
                    "pairs_tried": tried,
                    "pairs_with_path": pairs_with_path,
                    "max_walk_min_start": float(max_walk_min_start),
                    "max_walk_min_end": float(max_walk_min_end),
                    "max_candidate_stations": int(max_candidate_stations),
                },
                "walk_only_time_min": float(walk_only_time),
            }

    if best_pack is None:
        raise nx.NetworkXNoPath(
            f"Nessun percorso trovato (≤{max_line_changes} cambi). "
            f"start_candidates={len(start_candidates)}, end_candidates={len(end_candidates)}, "
            f"pairs_tried={tried}, pairs_with_path={pairs_with_path}."
        )

    best_pack["start_input_lonlat"] = tuple(map(float, start_lonlat))
    best_pack["end_input_lonlat"] = tuple(map(float, end_lonlat))

    best_pack["candidates"] = {
        "start_candidates": len(start_candidates),
        "end_candidates": len(end_candidates),
        "pairs_tried": tried,
        "pairs_with_path": pairs_with_path,
        "max_walk_min_start": float(max_walk_min_start),
        "max_walk_min_end": float(max_walk_min_end),
        "max_candidate_stations": int(max_candidate_stations),
    }
    if walk_only_time is not None:
        best_pack["walk_only_time_min"] = float(walk_only_time)
        best_pack["walk_only_dist_m"] = float(walk_only_dist_m)

    return float(best_pack["total_time_min"]), best_pack


# =========================
# INEQUALITY METRICS
# =========================

import numpy as np

def gini_coefficient(values, normalize=True, square_x=False):
    """Gini per array di valori non-negativi.
    Se normalize=True, normalizza dividendo per il massimo possibile (n-1)/n.
    Se square_x=True, usa x^2 (dopo clip a 0 e prima del calcolo).
    Ritorna NaN se input vuoto / somma <= 0 / n < 2 (in normalizzato).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan

    x = np.clip(x, 0, None)

    if square_x:
        x = x**2

    s = x.sum()
    if s <= 0:
        return np.nan

    x_sorted = np.sort(x)
    n = x_sorted.size
    i = np.arange(1, n + 1, dtype=float)

    g = float((2.0 * np.sum(i * x_sorted) / (n * s)) - (n + 1) / n)

    if not normalize:
        return g

    if n < 2:
        return np.nan

    g_max = (n - 1) / n
    g_norm = g / g_max

    return float(np.clip(g_norm, 0.0, 1.0))


def theil_index(values):
    """Theil T (indice entropico). Ritorna NaN se mean<=0 o input vuoto."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    x = np.clip(x, 0, None)
    mu = x.mean()
    if mu <= 0:
        return np.nan
    x_pos = x[x > 0]
    return float(np.mean((x_pos / mu) * np.log(x_pos / mu)))


# =========================
# BATCH: MANY STARTS -> ONE TARGET
# =========================

def accessibility_inequality_to_target(
    G,
    starts_lonlat,
    target_lonlat,
    node_index=None,
    walk_speed_m_per_min=None,
    max_line_changes=None,
    change_penalty_min=None,
    max_walk_min_start=15.0,
    max_walk_min_end=15.0,
    max_candidate_stations=25,
    allow_walk_only=True,
    keep_details=False,
):
    """
    Calcola tempi di percorrenza da molti start verso un target.
    Ritorna: (results_df, metrics_dict)

    Aggiornamento:
    - salva anche walk_time_start_min / walk_time_end_min e walk_dist_start_m / walk_dist_end_m
      quando disponibili nei dettagli.
    """
    rows = []

    if node_index is None:
        node_index = build_node_index(G)

    for idx, start in enumerate(starts_lonlat):
        rec = {
            "i": idx,
            "start_lon": float(start[0]),
            "start_lat": float(start[1]),
            "target_lon": float(target_lonlat[0]),
            "target_lat": float(target_lonlat[1]),
            "ok": True,
            "total_time_min": np.nan,
            "metro_time_min": np.nan,
            "walk_time_min": np.nan,

            # ✅ nuovi campi
            "walk_time_start_min": np.nan,
            "walk_time_end_min": np.nan,
            "walk_dist_start_m": np.nan,
            "walk_dist_end_m": np.nan,

            "penalty_paid_in_path_min": np.nan,
            "snapped_start_node": None,
            "snapped_end_node": None,
            "line_changes": np.nan,
            "distinct_lines": None,
            "mode": None,
            "error": None,
        }

        try:
            kwargs = dict(node_index=node_index)

            if walk_speed_m_per_min is not None:
                kwargs["walk_speed_m_per_min"] = walk_speed_m_per_min
            if max_line_changes is not None:
                kwargs["max_line_changes"] = max_line_changes
            if change_penalty_min is not None:
                kwargs["change_penalty_min"] = change_penalty_min

            kwargs["max_walk_min_start"] = float(max_walk_min_start)
            kwargs["max_walk_min_end"] = float(max_walk_min_end)
            kwargs["max_candidate_stations"] = int(max_candidate_stations)
            kwargs["allow_walk_only"] = bool(allow_walk_only)

            total, info = route_time_minutes(G, start, target_lonlat, **kwargs)

            rec["total_time_min"] = float(total)
            rec["metro_time_min"] = float(info.get("metro_time_min", np.nan))
            rec["walk_time_min"] = float(info.get("walk_time_min", np.nan))

            # ✅ nuovi dettagli (se presenti)
            rec["walk_time_start_min"] = float(info.get("walk_time_start_min", np.nan))
            rec["walk_time_end_min"] = float(info.get("walk_time_end_min", np.nan))
            rec["walk_dist_start_m"] = float(info.get("walk_dist_start_m", np.nan))
            rec["walk_dist_end_m"] = float(info.get("walk_dist_end_m", np.nan))

            rec["penalty_paid_in_path_min"] = float(info.get("penalty_paid_in_path_min", np.nan))
            rec["snapped_start_node"] = info.get("snapped_start_node", None)
            rec["snapped_end_node"] = info.get("snapped_end_node", None)
            rec["line_changes"] = int(info.get("line_changes", 0))
            rec["distinct_lines"] = info.get("distinct_lines", [])
            rec["mode"] = info.get("mode", None)

            if keep_details:
                rec["details"] = info

        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, ValueError) as e:
            rec["ok"] = False
            rec["error"] = type(e).__name__ + ": " + str(e)

        rows.append(rec)

    results_df = pd.DataFrame(rows)

    valid_times = results_df.loc[results_df["ok"], "total_time_min"].to_numpy(dtype=float)

    metrics = {
        "n_total": int(len(results_df)),
        "n_ok": int(np.sum(results_df["ok"])),
        "share_ok": float(np.mean(results_df["ok"])) if len(results_df) else np.nan,
        "mean_time_min": float(np.mean(valid_times)) if valid_times.size else np.nan,
        "median_time_min": float(np.median(valid_times)) if valid_times.size else np.nan,
        "p90_time_min": float(np.percentile(valid_times, 90)) if valid_times.size else np.nan,
        "min_time_min": float(np.min(valid_times)) if valid_times.size else np.nan,
        "max_time_min": float(np.max(valid_times)) if valid_times.size else np.nan,
        "gini_time": gini_coefficient(valid_times, normalize=True, square_x=True),
        "theil_time": theil_index(valid_times),
    }

    return results_df, metrics


# =========================
# EXAMPLE USAGE
# =========================
# G = build_graph_from_edgelist("./timed_edgelist.geojson")
# node_index = build_node_index(G)
# starts = [(2.30,48.88), (2.33,48.86), (2.25,48.84)]
# target = (2.377442453169209, 48.84950447433732)
# df, m = accessibility_inequality_to_target(
#     G,
#     starts,
#     target,
#     node_index=node_index,
#     max_line_changes=1,
#     change_penalty_min=2.0,
#     max_walk_min_start=15.0,
#     max_walk_min_end=15.0,
#     max_candidate_stations=25,
#     allow_walk_only=True,
#     keep_details=True,
# )
# print(df.head())
# print(m)
