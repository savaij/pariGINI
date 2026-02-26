"""
Streamlit App: pariGINI
Calcola e visualizza il Gini Index per la disuguaglianza di accessibilità in metro a Parigi

Versione riscritta:
- Colorazione esagoni: SOLO verde->rosso (niente blu)
- Esagoni ricentrati sui centri + scala x2
- Riempimento più opaco
- Rendering mappa molto più veloce:
    * usa Plotly Choroplethmapbox (1 trace) invece di migliaia di Scattermapbox
    * geometrie semplificate e pre-cached (topology-preserving)
- Ripulisce plot obsoleti (mean_time_softmax), usa gini_time(score) corretti
"""

import json
import math
import random
import requests

import numpy as np
import pandas as pd
import geopandas as gpd

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from streamlit_searchbox import st_searchbox  # pip install streamlit-searchbox

from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, Point

from gini_paris_distances_calculations import (
    load_precomputed_data,
    build_graph_from_edgelist,
    build_node_index,
    accessibility_inequality_to_targets,
)


# st.markdown("""
#     <style>
#     @import url("https://fonts.googleapis.com/css2?family=Archivo:ital,wght@0,100..900;1,100..900&family=Epilogue:ital,wght@0,100..900;1,100..900&family=Sora:wght@100..800&display=swap");

#     html, body, [class*="st-"], button, input, textarea, p, h1, h2, h3, h4, h5, h6, label {
#         font-family: "Archivo", sans-serif !important;
#         font-optical-sizing: auto;
#     }
#     </style>
# """, unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Archivo:ital,wdth,wght@0,90,100..900;1,90,100..900&display=swap");

    html, body, [class*="st-"], button, input, textarea, p, label {
        font-family: "Archivo", sans-serif !important;
        font-optical-sizing: auto;
    }

    h1 {
        font-family: "Archivo", sans-serif !important;
        font-weight: 900 !important;
        font-stretch: 20% !important;
    }
            
    h2 {
    font-size: 1.4rem !important;
    font-family: "Archivo", sans-serif !important;
    font-weight: 900 !important;
    font-stretch: 20% !important;
    }

    h3, h4, h5, h6 {
        font-family: "Archivo", sans-serif !important;
    }
            
    div.stButton > button[data-testid="baseButton-primary"] {
    font-size: 1.4rem !important;
    font-weight: 900 !important;
    padding: 0.6rem 1.2rem !important;
    }       
    </style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS pythonh1, h2 {

# ============================================================
WGS84_EPSG = 4326
METRIC_EPSG = 2154


def round_minutes(x) -> int:
    try:
        v = float(x)
    except Exception:
        return 0
    if not np.isfinite(v) or v <= 0:
        return 0
    return int(math.floor(v + 0.5))


def fmt_min(x) -> str:
    return str(round_minutes(x))


def _hex_to_rgba(h, a):
    h = str(h).lstrip("#")
    if len(h) != 6:
        return f"rgba(200,200,200,{a})"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

@st.cache_data(show_spinner=False)
def load_metro_lines():
    gdf = gpd.read_file("./metro_lignes_merged.geojson")
    if gdf.crs is None or gdf.crs.to_epsg() != WGS84_EPSG:
        gdf = gdf.to_crs(epsg=WGS84_EPSG)
    gdf["indice_lig"] = gdf["indice_lig"].astype(str).str.strip()
    return gdf
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="pariGINI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# COLORS
# ============================================================
LINE_COLORS = {
    "1": "#FFCD00",
    "2": "#003CA6",
    "3": "#7A8B2E",
    "3bis": "#8E9AE6",
    "4": "#7C2E83",
    "5": "#FF7E2E",
    "6": "#6EC4B1",
    "7": "#FA9ABA",
    "7bis": "#6EC4B1",
    "8": "#CEADD2",
    "9": "#B7D84B",
    "10": "#C9910D",
    "11": "#704B1C",
    "12": "#007852",
    "13": "#8E9AE6",
    "14": "#62259D",
}
WALK_COLOR = "#9CA3AF"

# Solo verde -> rosso (per score 1..10 o norm 0..1)
GREEN_HEX = "#22c55e"
RED_HEX = "#ef4444"

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown(
    """
<style>
:root, html { color-scheme: light !important; }
html, body { background: #ffffff !important; }
[data-testid="stAppViewContainer"] { background: #ffffff !important; }
[data-testid="stHeader"] { background: #ffffff !important; }
[data-testid="stSidebar"] { background: #ffffff !important; }
[data-testid="stSidebarContent"] { background: #ffffff !important; }

body, p, li, label, span, div { color: #111827 !important; }
.block-container { padding-top: 2.2rem; }

.pg-title {
  font-size: 3.8rem;
  font-weight: 900;
  line-height: 1.02;
  margin: 0.6rem 0 0.5rem 0;
}

div.stButton > button {
  border-radius: 12px;
  border: 1px solid rgba(17,24,39,0.18) !important;
  background: #ffffff !important;
  color: #111827 !important;
}
div.stButton > button:hover {
  border-color: rgba(17,24,39,0.35) !important;
}

div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
  background: #F54927 !important;
  color: #ffffff !important;
  border: 1px solid #F54927 !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
  background: #d93d1f !important;
  border-color: #d93d1f !important;
}

[data-testid="stMetricValue"] { color: #111827 !important; }
[data-testid="stMetricLabel"] { color: rgba(17,24,39,0.75) !important; }

/* Searchbox */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="input-container"],
div[data-baseweb="select"] {
  background-color: #f9fafb !important;
  background: #f9fafb !important;
}
div[data-baseweb="input"] *,
div[data-baseweb="base-input"] *,
div[data-baseweb="input-container"] *,
div[data-baseweb="select"] * {
  background-color: transparent !important;
  color: #111827 !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
div[data-baseweb="select"] input {
  background-color: transparent !important;
  background: transparent !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="base-input"] input::placeholder,
div[data-baseweb="select"] input::placeholder {
  color: rgba(17,24,39,0.55) !important;
  -webkit-text-fill-color: rgba(17,24,39,0.55) !important;
}
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div {
  background: #f3f4f6 !important;
  background-color: #f3f4f6 !important;
  color: #111827 !important;
  border: 1px solid rgba(17,24,39,0.18) !important;
  box-shadow: 0 10px 24px rgba(17,24,39,0.12) !important;
}
div[data-baseweb="menu"],
div[data-baseweb="menu"] ul {
  background: #f3f4f6 !important;
  background-color: #f3f4f6 !important;
  color: #111827 !important;
}
div[data-baseweb="menu"] [role="option"]:hover,
div[data-baseweb="menu"] [role="option"][aria-selected="true"],
div[data-baseweb="menu"] li:hover {
  background: #e5e7eb !important;
  background-color: #e5e7eb !important;
}
div[data-baseweb="tag"],
span[data-baseweb="tag"] {
  background-color: #e5e7eb !important;
  color: #111827 !important;
}



</style>
""",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)



# ============================================================
# API: Géoplateforme - Autocompletion (IGN)
# ============================================================
COMPLETION_URL = "https://data.geopf.fr/geocodage/completion/"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def geopf_completion(
    text: str,
    terr: str = "75",
    maximumResponses: int = 8,
    types: str = "StreetAddress,PositionOfInterest",
):
    text = (text or "").strip()
    if not text:
        return []

    r = requests.get(
        COMPLETION_URL,
        params={
            "text": text,
            "terr": terr,
            "type": types,
            "maximumResponses": maximumResponses,
        },
        timeout=10,
        headers={"User-Agent": "streamlit-pariGINI/1.0"},
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        return []
    return data.get("results", []) or []


def make_search_fn(map_key: str):
    def _search(searchterm: str):
        if not searchterm or len(searchterm.strip()) < 3:
            st.session_state[map_key] = {}
            return []

        results = geopf_completion(searchterm, terr="75", maximumResponses=8)
        mp, opts = {}, []
        for r in results:
            ft = r.get("fulltext")
            x = r.get("x")
            y = r.get("y")
            if ft and x is not None and y is not None:
                opts.append(ft)
                mp[ft] = (float(x), float(y))
        st.session_state[map_key] = mp
        return opts

    return _search


def address_autocomplete(label: str, key: str, placeholder: str):
    map_key = f"{key}__map"
    search_fn = make_search_fn(map_key)

    selected = st_searchbox(
        search_fn,
        placeholder=placeholder,
        label=label,
        key=key,
        clear_on_submit=False,
    )

    coords = None
    if selected:
        coords = (st.session_state.get(map_key) or {}).get(selected)

    return selected, coords


# ============================================================
# ROUTE BARS (HTML/CSS)  [invariato, ma tenuto]
# ============================================================
def _norm_line_for_color(line):
    if line is None:
        return None
    s = str(line).strip().lower().replace(" ", "")
    if s in {"3b", "3bis"}:
        return "3bis"
    if s in {"7b", "7bis"}:
        return "7bis"
    return str(line).strip()


def compress_edges_to_line_segments(edges):
    segs = []
    cur_line = None
    cur_t = 0.0

    for e in edges:
        line = e.get("line")
        t = float(e.get("time_min", 0.0) or 0.0)

        if cur_line is None:
            cur_line = line
            cur_t = t
            continue

        if line == cur_line:
            cur_t += t
        else:
            segs.append({"kind": "metro", "line": cur_line, "time_min": float(cur_t)})
            cur_line = line
            cur_t = t

    if cur_line is not None:
        segs.append({"kind": "metro", "line": cur_line, "time_min": float(cur_t)})

    return segs


def _get_walk_split(details):
    if not isinstance(details, dict):
        return 0.0, 0.0

    w_start = details.get("walk_time_start_min", None)
    w_end = details.get("walk_time_end_min", None)

    total_walk = float(details.get("walk_time_min", 0.0) or 0.0)
    edges = details.get("edges", []) or []
    has_metro = len(edges) > 0

    if w_start is not None or w_end is not None:
        return float(w_start or 0.0), float(w_end or 0.0)

    if total_walk <= 0:
        return 0.0, 0.0

    if has_metro:
        return total_walk / 2.0, total_walk / 2.0
    return total_walk, 0.0


def build_segments_for_friend(details):
    mode = details.get("mode", "metro_walk")

    if mode == "walk_only":
        w = float(details.get("walk_time_min", 0.0) or 0.0)
        return [{"kind": "walk", "time_min": w}]

    segs = []
    w_start, w_end = _get_walk_split(details)

    if w_start > 0:
        segs.append({"kind": "walk", "time_min": float(w_start)})

    edges = details.get("edges", []) or []
    segs.extend(compress_edges_to_line_segments(edges))

    if w_end > 0:
        segs.append({"kind": "walk", "time_min": float(w_end)})

    return segs


def render_routes_html(results_df):
    ok_df = results_df[results_df["ok"] == True].copy()
    if ok_df.empty:
        st.info("No routes available to display.")
        return

    ok_df = ok_df.sort_values("i")
    max_total = 0.0
    precomp = []
    used_lines = set()

    for _, r in ok_df.iterrows():
        details = r["details"]
        total = float(details.get("total_time_min", r.get("total_time_min", 0.0)) or 0.0)
        max_total = max(max_total, total)

        segs = build_segments_for_friend(details)
        for s in segs:
            if s["kind"] == "metro":
                lk = _norm_line_for_color(s.get("line")) or "?"
                used_lines.add(lk)

        precomp.append((int(r["i"]), total, segs))

    max_total = max(max_total, 1e-9)

    def _legend_pill(label, color):
        return f"""<span class="pill" style="background:{color}"></span><span class="pilltxt">{label}</span>"""

    def _line_sort_key(x):
        try:
            return (0, float(str(x).replace("bis", ".5")))
        except Exception:
            return (1, str(x))

    legend_html = f"""
    <div class="legend">
      <div class="legend-item">{_legend_pill("Walking", WALK_COLOR)}</div>
      {"".join([f'<div class="legend-item">{_legend_pill("Metro " + str(l), LINE_COLORS.get(l, "#666"))}</div>' for l in sorted(list(used_lines), key=_line_sort_key)])}
    </div>
    """

    rows = []
    for i, total, segs in precomp:
        name = f"Friend {i+1}"
        seg_html = ""

        for s in segs:
            dt = float(s.get("time_min", 0.0) or 0.0)
            if dt <= 0:
                continue

            w_pct = (dt / max_total) * 100.0
            if s["kind"] == "walk":
                color = WALK_COLOR
                title = f"Walking: {fmt_min(dt)} min"
            else:
                lk = _norm_line_for_color(s.get("line")) or "?"
                color = LINE_COLORS.get(lk, "#666666")
                title = f"Metro {lk}: {fmt_min(dt)} min"

            seg_html += f"""
              <div class="seg" title="{title}" style="width:{w_pct:.4f}%; background:{color};"></div>
            """

        rows.append(
            f"""
            <div class="r">
              <div class="who">{name}</div>
              <div class="bar">{seg_html}</div>
              <div class="tot">{fmt_min(total)} min</div>
            </div>
            """
        )

    iframe_height = int(min(380, 95 + 32 * len(rows)))

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  :root {{
    --text: rgba(17,24,39,0.92);
    --muted: rgba(17,24,39,0.70);
    --border: rgba(17,24,39,0.18);
  }}

  .wrap {{
    width: 48vw;
    max-width: 680px;
    min-width: 340px;
  }}

  .legend {{
    display:flex;
    flex-wrap: wrap;
    gap: 10px 14px;
    align-items: center;
    margin: 0 0 10px 0;
  }}
  .legend-item {{
    display:flex; gap:8px; align-items:center;
    font: 13px/1.2 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    color: var(--muted);
  }}
  .pill {{
    width: 12px;
    height: 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    display:inline-block;
  }}
  .pilltxt {{ white-space: nowrap; }}

  .r {{
    display: grid;
    grid-template-columns: 80px 1fr 74px;
    gap: 10px;
    align-items: center;
    margin: 6px 0;
  }}

  .who {{
    font: 14px/1.2 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    color: var(--text);
    font-weight: 700;
  }}

  .tot {{
    font: 13px/1.2 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    color: var(--muted);
    text-align: right;
    white-space: nowrap;
  }}

  .bar {{
    display:flex;
    align-items:center;
    gap: 4px;
    height: 14px;
  }}

  .seg {{
    height: 14px;
    border-radius: 999px;
    min-width: 6px;
    border: 1px solid var(--border);
    box-sizing: border-box;
  }}
</style>
</head>
<body style="margin:0; background:transparent;">
  <div class="wrap">
    {legend_html}
    {''.join(rows)}
  </div>
</body>
</html>
"""
    components.html(html, height=iframe_height, scrolling=False)


# ============================================================
# COLOR MAP: verde -> rosso (no blu)
# ============================================================
def gini_to_green_red_hex(v01: float) -> str:
    """
    v01 in [0,1] -> colore lineare tra verde e rosso.
    """
    v = float(np.clip(v01, 0.0, 1.0))
    g = np.array([34, 197, 94], dtype=float)   # #22c55e
    r = np.array([239, 68, 68], dtype=float)   # #ef4444
    rgb = g + (r - g) * v
    rgb = np.clip(np.round(rgb).astype(int), 0, 255)
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


# ============================================================
# FAST HEX MAP PREP (cache)
# ============================================================


@st.cache_data(show_spinner=False)
def prepare_hex_geojson_fast(_hexes_gdf, simplify_m=2.0):
    hex_w = _hexes_gdf.copy().reset_index(drop=True)

    if hex_w.crs is None or hex_w.crs.to_epsg() != WGS84_EPSG:
        hex_w = hex_w.to_crs(epsg=WGS84_EPSG)

    if simplify_m and simplify_m > 0:
        hex_m = hex_w.to_crs(epsg=METRIC_EPSG)
        hex_m["geometry"] = hex_m["geometry"].simplify(float(simplify_m), preserve_topology=True)
        hex_w = hex_m.to_crs(epsg=WGS84_EPSG)

    # Riassegna id progressivi 0,1,2,... identici a target_id
    hex_w["id"] = np.arange(len(hex_w), dtype=int)

    centroid = hex_w.geometry.union_all().centroid
    center = dict(lat=float(centroid.y), lon=float(centroid.x))

    hex_w["id_str"] = hex_w["id"].astype(str)
    geojson = json.loads(hex_w.set_index("id_str").to_json())

    return geojson, center


def render_hexagon_map_fast(geojson, metrics_df, hexes_gdf):
    from scipy.interpolate import RBFInterpolator
    import io
    import base64
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    df = metrics_df.copy()
    df["id_str"] = pd.to_numeric(df["target_id"], errors="coerce").astype("Int64").astype(str)

    # --- Gini ---
    if "gini_time" in df.columns:
        gini = pd.to_numeric(df["gini_time"], errors="coerce")
    else:
        gini = pd.Series(np.nan, index=df.index)
    gini = gini.clip(lower=0.0, upper=1.0)

    # --- Tempo medio ---
    mean_time = pd.to_numeric(df.get("mean_time_min", np.nan), errors="coerce")

    # --- Fair index ---
    if "fair_index" in df.columns:
        fair = pd.to_numeric(df["fair_index"], errors="coerce")
    else:
        fair = mean_time * (gini + 1.0)

    # --- Normalizzazione ---
    fair_valid = fair[np.isfinite(fair)]
    if fair_valid.empty:
        z = pd.Series(0.5, index=df.index)
    else:
        lo = float(np.nanpercentile(fair_valid.to_numpy(dtype=float), 5))
        hi = float(np.nanpercentile(fair_valid.to_numpy(dtype=float), 95))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(fair_valid.to_numpy(dtype=float)))
            hi = float(np.nanmax(fair_valid.to_numpy(dtype=float)))
            if hi <= lo:
                lo, hi = 0.0, lo + 1.0
        z = (fair - lo) / (hi - lo)
        z = z.clip(lower=0.0, upper=1.0)

        GAMMA = 3.5  
        z = 1.0 - (1.0 - z) ** GAMMA

    # --- Centroidi esagoni in WGS84 ---
    hexes_w = hexes_gdf.copy().reset_index(drop=True)
    if hexes_w.crs is None or hexes_w.crs.to_epsg() != WGS84_EPSG:
        hexes_w = hexes_w.to_crs(epsg=WGS84_EPSG)
    centroids = hexes_w.geometry.centroid
    lons = centroids.x.values
    lats = centroids.y.values

    # Allinea z ai centroidi tramite target_id
    z_vals = np.full(len(lons), np.nan)
    for idx, row in df.iterrows():
        tid = pd.to_numeric(row["target_id"], errors="coerce")
        if np.isfinite(tid) and int(tid) < len(z_vals):
            v = z.iloc[idx] if hasattr(z, 'iloc') else z[idx]
            if np.isfinite(v):
                z_vals[int(tid)] = float(v)

   # Rimuovi NaN
    valid_mask = np.isfinite(z_vals)
    lons_v = lons[valid_mask]
    lats_v = lats[valid_mask]
    z_v = z_vals[valid_mask]

    # --- Hover allineato ai punti validi ---
    arr_col = df.get("c_ar", pd.Series([None] * len(df)))
    nn_col = df.get("nearest_node", pd.Series([None] * len(df)))
    hover_all = []
    for ar, gg, mt, nn in zip(arr_col, gini, mean_time, nn_col):
        atxt = f"Arrondissement: {str(ar)}" if not pd.isna(ar) else "Arrondissement: N/A"
        gtxt = f"Gini: {float(gg):.3f}" if not pd.isna(gg) else "Gini: N/A"
        mtxt = f"Avg time: {fmt_min(mt)} min" if not pd.isna(mt) else "Avg time: N/A"
        nntxt = f"Station: {str(nn)}" if not pd.isna(nn) else "Station: N/A"
        hover_all.append(f"{atxt}<br>{gtxt}, {mtxt}<br>{nntxt}")
    hover_valid = [hover_all[i] for i in range(len(z_vals)) if valid_mask[i]]

    # --- Maschera di Parigi: calcolata PRIMA dell'interpolazione ---
    from shapely.ops import unary_union
    from shapely.prepared import prep
    from PIL import ImageDraw

    arr_gdf = gpd.read_file("./paris_senza_fiume.geojson")
    if arr_gdf.crs is None or arr_gdf.crs.to_epsg() != WGS84_EPSG:
        arr_gdf = arr_gdf.to_crs(epsg=WGS84_EPSG)
    paris_shape = unary_union(arr_gdf.geometry)

    # --- Interpolazione RBF solo sui pixel interni a Parigi ---
    # Bounds geografici
    lon_min, lon_max = lons_v.min() - 0.01, lons_v.max() + 0.01
    lat_min, lat_max = lats_v.min() - 0.01, lats_v.max() + 0.01

    grid_res = 500  # risoluzione immagine in pixel
    grid_lon = np.linspace(lon_min, lon_max, grid_res)
    grid_lat = np.linspace(lat_min, lat_max, grid_res)
    glon, glat = np.meshgrid(grid_lon, grid_lat)

    # Maschera raster: rasterizza il poligono di Parigi sulla griglia pixel
    def geo_to_pixel(lon, lat):
        px = int((lon - lon_min) / (lon_max - lon_min) * grid_res)
        py = int((lat_max - lat) / (lat_max - lat_min) * grid_res)
        return (px, py)

    mask_img = Image.new("L", (grid_res, grid_res), 0)
    draw = ImageDraw.Draw(mask_img)
    geoms = paris_shape.geoms if hasattr(paris_shape, 'geoms') else [paris_shape]
    for geom in geoms:
        exterior_pixels = [geo_to_pixel(lon, lat) for lon, lat in geom.exterior.coords]
        draw.polygon(exterior_pixels, fill=255)
    mask_array = np.array(mask_img) / 255.0  # shape (grid_res, grid_res), 1.0 dentro Parigi

    # Individua indici pixel dentro Parigi (sulla griglia non-flippata)
    # mask_array ha riga 0 = lat_max (come immagine), glon/glat: riga 0 = lat_min
    # Quindi flippiamo mask_array per allinearla alla griglia meshgrid
    mask_grid = np.flipud(mask_array)  # ora riga 0 = lat_min, come meshgrid
    inside_flat = mask_grid.ravel() > 0.5
    n_inside = int(inside_flat.sum())

    all_grid_points = np.column_stack([glon.ravel(), glat.ravel()])
    inside_points = all_grid_points[inside_flat]  # solo pixel dentro Parigi

    # RBF interpolation solo sui punti interni
    points = np.column_stack([lons_v, lats_v])
    rbf = RBFInterpolator(points, z_v, kernel='linear', smoothing=0.01)
    inside_z = rbf(inside_points)
    inside_z = np.clip(inside_z, 0.0, 1.0)

    # Ricostruisci griglia completa (NaN fuori)
    grid_z_flat = np.full(grid_res * grid_res, np.nan)
    grid_z_flat[inside_flat] = inside_z
    grid_z = grid_z_flat.reshape(grid_res, grid_res)
    # Flip verticale perché le immagini hanno y invertito
    grid_z = np.flipud(grid_z)

    # --- Colormap verde -> rosso ---
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gini_cmap",
        ["#059669", "#93a750", "#e2792a", "#dc2626"]
    )

    # --- Alpha: fadeout ai bordi usando distanza dai punti dati (solo dentro Parigi) ---
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([lons_v, lats_v]))
    inside_dist, _ = tree.query(inside_points, k=1)

    dist_flat = np.full(grid_res * grid_res, np.inf)
    dist_flat[inside_flat] = inside_dist
    dist_grid = dist_flat.reshape(grid_res, grid_res)
    dist_grid = np.flipud(dist_grid)

    fade_radius = 0.04
    alpha = np.clip(1.0 - dist_grid / fade_radius, 0.0, 1.0)
    alpha = (alpha ** 0.5) * 0.75

    # --- Costruisci immagine RGBA ---
    # Per i pixel fuori Parigi, grid_z è NaN: cmap li rende trasparenti,
    # ma forziamo alpha=0 esplicitamente
    grid_z_safe = np.where(np.isfinite(grid_z), grid_z, 0.0)
    rgba = cmap(grid_z_safe)  # shape (H, W, 4)
    rgba[:, :, 3] = alpha * mask_array  # mask_array già orientata come immagine
    img_array = (rgba * 255).astype(np.uint8)

    img = Image.fromarray(img_array, mode="RGBA")

    # Encode in base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    img_src = f"data:image/png;base64,{img_b64}"

    # --- Figura Plotly: mappa base + immagine sovrapposta ---

    centroid_union = hexes_w.geometry.union_all().centroid
    center = dict(lat=float(centroid_union.y), lon=float(centroid_union.x))

    fig = go.Figure(go.Scattermap(
        lat=lats_v.tolist(),
        lon=lons_v.tolist(),
        mode="markers",
        marker=dict(size=8, opacity=0),
        text=hover_valid,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        map=dict(
            style="carto-positron",
            zoom=11,
            center=center,
            layers=[{
                "sourcetype": "image",
                "source": img_src,
                "coordinates": [
                    [lon_min, lat_max],
                    [lon_max, lat_max],
                    [lon_max, lat_min],
                    [lon_min, lat_min],
                ],
                "opacity": 0.60,
                "below": "",
            }]
        ),
        height=620,
    )

    # --- Carica arrondissement con confini precisi ---
    arr_noparks = gpd.read_file("./arrondissement_noparks.geojson")
    if arr_noparks.crs is None or arr_noparks.crs.to_epsg() != WGS84_EPSG:
        arr_noparks = arr_noparks.to_crs(epsg=WGS84_EPSG)

    # Costruisci un trace Scattermap per ogni arrondissement (poligono + hover)
    for _, arr_row in arr_noparks.iterrows():
        geom = arr_row.geometry
        nom = arr_row.get("l_ar", "")
        c_ar = arr_row.get("c_ar", "")

        # Estrai coordinate del poligono (gestisci MultiPolygon)
        polys = geom.geoms if hasattr(geom, 'geoms') else [geom]
        for poly in polys:
            lons_poly, lats_poly = zip(*[(c[0], c[1]) for c in poly.exterior.coords])
            fig.add_trace(go.Scattermap(
                lat=list(lats_poly),
                lon=list(lons_poly),
                mode="lines",
                line=dict(width=1.5, color="rgba(17,24,39,0.35)"),
                fill="toself",
                fillcolor="rgba(0,0,0,0)",
                # hovertext=f"{nom} ({c_ar}ème)",
                hoverinfo="skip",
                showlegend=False,
            ))

    
    # --- Linee metro ---
    metro_lines_gdf = load_metro_lines()

    def _geom_to_lonlat(geom):
        lons, lats = [], []
        if geom is None or geom.is_empty:
            return lons, lats
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            xs, ys = part.xy
            lons.extend(xs); lons.append(None)
            lats.extend(ys); lats.append(None)
        return lons, lats

    for _, line_row in metro_lines_gdf.iterrows():
        ln = str(line_row["indice_lig"]).strip()
        lk = _norm_line_for_color(ln)
        color = LINE_COLORS.get(lk, "#888888")
        lons_l, lats_l = _geom_to_lonlat(line_row.geometry)
        if not lons_l:
            continue
        fig.add_trace(go.Scattermap(
            lat=lats_l,
            lon=lons_l,
            mode="lines",
            line=dict(width=2.5, color=color),
            name=f"M{ln}",
            hoverinfo="name",
            showlegend=False,
        ))

    return fig, center




# ============================================================
# HEADER
# ============================================================

st.markdown("""
    <span class="parigini-title" style="font-size: 2.5rem;">
        pari<span class="gini-red">GINI</span>
    </span>
    <style>
    .parigini-title {
        font-family: 'Archivo', sans-serif;
        font-weight: 900;
        font-stretch: 60%;
    }
    .gini-red {
        color: #F54927 !important;
    }
    </style>
""", unsafe_allow_html=True)

# st.markdown("<div class='pg-title'>pariGINI</div>", unsafe_allow_html=True)
st.markdown("""
Are your friends suggesting a bar that is too far? **Check how fair that choice is.**<br>
Measure inequality in metro travel times using the Gini-based _Fair index_. <br>
Methodology is described below.
""", unsafe_allow_html=True)

# ============================================================
# LOAD GRAPH & HEXAGONS (cache)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_graph():
    G_ = build_graph_from_edgelist("./timed_edgelist.geojson")
    idx_ = build_node_index(G_)
    return G_, idx_


@st.cache_data(show_spinner=False)
def load_hexagon_data():
    hexes_gdf = gpd.read_file("./esagoni_parigi_simplified_name.geojson")
    return hexes_gdf


@st.cache_data(show_spinner=False)
def load_precomputed_safe():
    """
    Tenta di caricari dati precomputati.
    Se errore, ritorna None (fallback a Dijkstra).
    """
    try:
        precomp = load_precomputed_data(
            npz_path="./precomputed_node_times.npz",
            json_path="./precomputed_node_times_meta.json"
        )
        return precomp
    except Exception as e:
        st.write(f"⚠️ Precomputed data not available ({type(e).__name__}): falling back to Dijkstra.")
        return None


with st.spinner("Loading metro network, hex grid, and precomputed data..."):
    try:
        G, node_index = load_graph()
        hexes_gdf = load_hexagon_data()
        precomputed = load_precomputed_safe()
    except Exception as e:
        st.error(f"Loading error: {e}")
        st.stop()

# Prepara geometrie mappa (cache) una volta: ricentra + scala + semplifica + geojson
geojson_hex, map_center = prepare_hex_geojson_fast(
    _hexes_gdf=hexes_gdf,
)



# ============================================================
# PARAMETRI ROUTING FISSI
# ============================================================
max_line_changes = 1
change_penalty_min = 2.0
DELETE_CHATELET = False

# ============================================================
# STATE: numero amici (min 2)
# ============================================================
if "n_friends" not in st.session_state:
    st.session_state.n_friends = 2


def add_friend():
    st.session_state.n_friends += 1


def remove_friend():
    if st.session_state.n_friends > 2:
        st.session_state.n_friends -= 1


# ============================================================
# INPUT: ORIGINS (AMICI)
# ============================================================
st.header("Where are you starting from?")

top_row = st.columns([1, 1, 8])
with top_row[0]:
    st.button("Add", use_container_width=True, on_click=add_friend)
with top_row[1]:
    st.button("Remove", use_container_width=True, disabled=(st.session_state.n_friends <= 2), on_click=remove_friend)
with top_row[2]:
    st.caption("Minimum 2 friends. Type the addresses and select a suggestion.")

starts = []
for i in range(st.session_state.n_friends):
    row = st.columns([1, 7])
    with row[0]:
        st.markdown(f"**Friend {i+1}**")
    with row[1]:
        _, coords = address_autocomplete(
            label="",
            key=f"friend_{i}",
            placeholder="Ex: Gare de Lyon, 75012 Paris • 10 Rue de Rivoli 75004 Paris",
        )
        if coords:
            starts.append(coords)

ready = len(starts) >= 2
# if not ready:
#     st.warning(f"Manca: almeno 2 amici (con indirizzo selezionato). Attualmente: {len(starts)}.")

# ============================================================
# ANALYSIS & RESULTS
# ============================================================
if st.button("Compute Gini", type="primary", use_container_width=True, disabled=not ready):
    st.divider()

    with st.spinner("Computing travel times for all points in the city..."):
        try:
            # Usa i centroidi degli esagoni come target di routing.
            hexes_base = hexes_gdf
            if hexes_base.crs is None:
                hexes_base = hexes_base.set_crs(epsg=WGS84_EPSG, allow_override=True)
            hexes_metric = hexes_base.to_crs(epsg=METRIC_EPSG)
            centroids_metric = hexes_metric.geometry.centroid
            centroids_wgs84 = gpd.GeoSeries(
                centroids_metric, crs=METRIC_EPSG
            ).to_crs(epsg=WGS84_EPSG)
            targets_lonlat = [
                (geom.x, geom.y)
                for geom in centroids_wgs84
            ]

            metrics_df = accessibility_inequality_to_targets(
                G,
                starts_lonlat=starts,
                targets_lonlat=targets_lonlat,
                node_index=node_index,
                max_line_changes=max_line_changes,
                change_penalty_min=change_penalty_min,
                max_walk_min_start=15.0,
                max_walk_min_end=15.0,
                max_candidate_stations=25,
                allow_walk_only=True,
                keep_details=False,          # keep_details=True è molto pesante: disattivalo per velocità
                return_per_target_df=False,  # evita dict gigante in memoria
                precomputed=precomputed,     # O dati precomputati (veloce) O None (Dijkstra)
            )

            # Aggiungi c_ar e nearest_node dagli esagoni (mappa id target -> colonna)
            if "target_id" in metrics_df.columns:
                metrics_df = metrics_df.copy()
                if "c_ar" in hexes_gdf.columns:
                    hexes_ar = hexes_gdf.reset_index(drop=True)["c_ar"]
                    ar_map = pd.Series(hexes_ar.values, index=np.arange(len(hexes_ar))).to_dict()
                    metrics_df["c_ar"] = metrics_df["target_id"].map(ar_map)
                if "nearest_node" in hexes_gdf.columns:
                    hexes_nn = hexes_gdf.reset_index(drop=True)["nearest_node"]
                    nn_map = pd.Series(hexes_nn.values, index=np.arange(len(hexes_nn))).to_dict()
                    metrics_df["nearest_node"] = metrics_df["target_id"].map(nn_map)
        except Exception as e:
            st.error(f"Computation error: {e}")
            st.stop()

    # ===========================
    # TOP 3 FERMATE (fair_index più basso + snodi)
    # ===========================
    def _get_station_lines(graph, station_name):
        """Restituisce le linee metro che passano per una fermata."""
        if station_name not in graph:
            return []
        lines = set()
        for _, _, data in graph.edges(station_name, data=True):
            l = data.get("line")
            if l is not None:
                lines.add(str(l).strip())
        return sorted(lines, key=lambda x: (0, float(x.replace("bis", ".5"))) if x.replace("bis", ".5").replace(".", "").isdigit() else (1, x))

    # Calcola fair_index per ogni target
    _gini_col = pd.to_numeric(metrics_df.get("gini_time"), errors="coerce")
    _mean_col = pd.to_numeric(metrics_df.get("mean_time_min"), errors="coerce")
    if "fair_index" in metrics_df.columns:
        _fair_col = pd.to_numeric(metrics_df["fair_index"], errors="coerce")
    else:
        _fair_col = _mean_col * (_gini_col + 1.0)

    _top_df = metrics_df.copy()
    _top_df["_fair"] = _fair_col
    _top_df = _top_df.dropna(subset=["_fair", "nearest_node"])

    # --- NUOVA LOGICA: top 5 esagoni, TUTTE le fermate vicine, top 3 per linee ---
    _top3 = _top_df.sort_values("_fair").head(3).copy()

    hexes_metric = hexes_gdf.copy().to_crs(epsg=METRIC_EPSG)
    RADIUS_M = 400

    # Costruisci GeoDataFrame dei nodi metro dal grafo
    # Costruisci GeoDataFrame dei nodi metro dal grafo
    _metro_nodes = {}
    for node, data in G.nodes(data=True):
        coords = data.get("coordinates")
        if coords is not None and len(coords) == 2:
            _metro_nodes[node] = (float(coords[0]), float(coords[1]))  # (lon, lat)
    
 
    _nodes_gdf = gpd.GeoDataFrame(
        {"station": list(_metro_nodes.keys())},
        geometry=[Point(lon, lat) for lon, lat in _metro_nodes.values()],
        crs=f"EPSG:{WGS84_EPSG}",
    ).to_crs(epsg=METRIC_EPSG)

    _station_rows = {}
    for _, row in _top3.iterrows():
        tid = int(row["target_id"])
        if tid >= len(hexes_metric):
            continue
        centroid = hexes_metric.geometry.iloc[tid].centroid
        fair_val = row["_fair"]

        for idx_n, node_row in _nodes_gdf.iterrows():
            dist = centroid.distance(node_row.geometry)
            if dist <= RADIUS_M:
                s = node_row["station"]
                if s not in _station_rows or fair_val < _station_rows[s]["_fair"]:
                    _station_rows[s] = row

    _station_list = []
    for station, row in _station_rows.items():
        if DELETE_CHATELET and "châtelet" in str(station).lower():
            continue
        lines = _get_station_lines(G, station)
        if len(lines) == 0:
            continue
        _station_list.append({
            "station": station,
            "n_lines": len(lines),
            "lines": lines,
            "row": row,
        })

    _station_list.sort(key=lambda x: (-x["n_lines"], x["row"]["_fair"]))
    _top3_stations = _station_list[:3]

    if _top3_stations:
        st.header("Here are the top three stations where you can get off and meet:")

        cols = st.columns(len(_top3_stations))
        for idx, entry in enumerate(_top3_stations):
            station = entry["station"]
            row = entry["row"]
            lines = entry["lines"]
            gini_val = row.get("gini_time")
            mean_val = row.get("mean_time_min")

            with cols[idx]:
                line_pills = ""
                if lines:
                    pills = []
                    for l in lines:
                        lk = _norm_line_for_color(l)
                        color = LINE_COLORS.get(lk, "#666")
                        pills.append(
                            f'<span style="display:inline-block;background:{color};color:#fff;'
                            f'font-weight:700;font-size:0.8rem;padding:2px 8px;border-radius:999px;'
                            f'margin:0 3px 3px 0;line-height:1.4;">{l}</span>'
                        )
                    line_pills = "".join(pills)

                gini_str = f"{float(gini_val):.3f}" if pd.notna(gini_val) else "N/A"
                mean_str = fmt_min(mean_val) if pd.notna(mean_val) else "N/A"

                st.markdown(
f"""
<div style="background:#f9fafb;border:1px solid rgba(17,24,39,0.12);border-radius:14px;padding:18px 16px 14px 16px;">
    <div style="font-size:1.25rem;font-weight:800;margin-bottom:6px;">{idx+1}. {station}</div>
  <div style="margin-bottom:8px;">{line_pills}</div>
  <div style="font-size:0.88rem;color:rgba(17,24,39,0.7);">
    Average time: <b>{mean_str} min</b> · Gini: <b>{gini_str}</b> · Lines: <b>{entry['n_lines']}</b>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

        if DELETE_CHATELET:
            st.markdown(
                """
                <div style="margin-top:16px;padding:12px;background:#fef3c7;border:1px solid rgba(245,158,11,0.3);border-radius:8px;font-size:0.85rem;color:rgba(17,24,39,0.8);">
                     Among these stations, Châtelet will never be shown. This is an arbitrary choice, simply because it is a dull and sad area to hang out.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    # ===========================
    # MAPPA (FAST)
    # ===========================
    st.header("Fair index by area")

    # fig_map = render_hexagon_map_fast(
    #     geojson=geojson_hex,
    #     metrics_df=metrics_df,
    # )

    fig_map, map_center = render_hexagon_map_fast(
        geojson=geojson_hex,
        metrics_df=metrics_df,
        hexes_gdf=hexes_gdf,
    )
    fig_map.update_layout(map=dict(center=map_center, zoom=11))
    st.plotly_chart(fig_map, use_container_width=True)


    st.info(
    """
**How to read it**
- Green: better accessibility fairness (low average time, low inequality)
- Red: worse accessibility fairness (high average time, high inequality)

Fair index combines average travel time with inequality. It is defined as:

$\\small\\text{mean travel time} \\times (\\text{Gini index} + 1)$

Higher times or higher inequality increase the score.
"""
)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
"""
<style>
.footer-pariGINI {
    color: rgba(145, 153, 152, 0.84) !important;
    font-size: 0.9rem;
    line-height: 1.6;
}
.footer-pariGINI * {
    color: rgba(145, 153, 152, 0.84) !important;
}
.footer-pariGINI strong {
    font-weight: 700;
}
</style>
<div class="footer-pariGINI">

**pariGINI**

Data: RATP Metro Network  
Autocomplete: Geoplateforme (IGN) - completion  

Francesco Farina and Francesco Paolo Savatteri. For omett and for all.

</div>
""",
    unsafe_allow_html=True
)
