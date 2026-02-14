"""
Streamlit App: pariGINI
Calcola e visualizza il Gini Index per la disuguaglianza di accessibilità in metro a Parigi
"""

import json
import math
import random
import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

from gini_paris_distances_calculations import (
    build_graph_from_edgelist,
    build_node_index,
    accessibility_inequality_to_target,
)

# ============================================================
# HELPERS
# ============================================================
def round_minutes(x) -> int:
    """Arrotondamento classico (13.7 -> 14). Assume x >= 0."""
    try:
        v = float(x)
    except Exception:
        return 0
    if not np.isfinite(v) or v <= 0:
        return 0
    return int(math.floor(v + 0.5))


def fmt_min(x) -> str:
    return str(round_minutes(x))


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

# ============================================================
# GLOBAL CSS (TEMA CHIARO FORZATO + DROPDOWN CHIARO + PRIMARY LEGGIBILE)
# ============================================================
st.markdown(
    """
<style>
/* Forza schema colori chiaro */
:root, html, body { color-scheme: light !important; }

/* Sfondo sempre bianco */
html, body { background: #ffffff !important; }
[data-testid="stAppViewContainer"] { background: #ffffff !important; }
[data-testid="stHeader"] { background: #ffffff !important; }
[data-testid="stSidebar"] { background: #ffffff !important; }
[data-testid="stSidebarContent"] { background: #ffffff !important; }

/* Testi scuri */
body, p, li, label, span, div { color: #111827 !important; }

/* Padding top + un po' di spazio a sinistra per non coprire le decorazioni */
.block-container {
  padding-top: 2.2rem;
  padding-left: 3.4rem;
}

/* Titolo più grande */
.pg-title {
  font-size: 3.8rem;
  font-weight: 900;
  line-height: 1.02;
  margin: 0.6rem 0 0.5rem 0;
}

/* Bottoni base */
div.stButton > button {
  border-radius: 12px;
  border: 1px solid rgba(17,24,39,0.18) !important;
  background: #ffffff !important;
  color: #111827 !important;
}
div.stButton > button:hover {
  border-color: rgba(17,24,39,0.35) !important;
}

/* Primary button: chiaro con testo scuro */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
  background: #e5e7eb !important;
  color: #111827 !important;
  border: 1px solid rgba(17,24,39,0.25) !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
  background: #d1d5db !important;
  border-color: rgba(17,24,39,0.35) !important;
}

/* Metric */
[data-testid="stMetricValue"] { color: #111827 !important; }
[data-testid="stMetricLabel"] { color: rgba(17,24,39,0.75) !important; }

/* ------------------------------------------------------------
   INPUT + SELECT (STREAMLIT BASEWEB) - LOOK CHIARO GRIGIO
   ------------------------------------------------------------ */

/* Text input container */
div[data-baseweb="input"] > div {
  background: #f9fafb !important;
  border: 1px solid rgba(17,24,39,0.18) !important;
  border-radius: 12px !important;
}
div[data-baseweb="input"] input {
  background: #f9fafb !important;
  color: #111827 !important;
}
div[data-baseweb="input"] input::placeholder {
  color: rgba(17,24,39,0.55) !important;
}

/* Selectbox container */
div[data-baseweb="select"] > div {
  background: #f9fafb !important;
  border: 1px solid rgba(17,24,39,0.18) !important;
  border-radius: 12px !important;
  color: #111827 !important;
}

/* Portale menu (dropdown) */
div[data-baseweb="popover"] {
  background: #f3f4f6 !important;
  color: #111827 !important;
  border: 1px solid rgba(17,24,39,0.18) !important;
  box-shadow: 0 10px 24px rgba(17,24,39,0.12) !important;
}

/* Liste opzioni (variano tra versioni) */
div[role="listbox"],
div[role="listbox"] ul,
ul[role="listbox"] {
  background: #f3f4f6 !important;
  color: #111827 !important;
}
div[role="listbox"] * ,
ul[role="listbox"] * {
  color: #111827 !important;
}

/* Riga opzione */
div[role="option"],
li[role="option"] {
  background: #f3f4f6 !important;
  color: #111827 !important;
}

/* Hover/selezionata */
div[role="option"]:hover,
li[role="option"]:hover,
div[role="option"][aria-selected="true"],
li[role="option"][aria-selected="true"] {
  background: #e5e7eb !important;
}

/* ------------------------------------------------------------
   DECORAZIONI LATERALI: "TRATTINI" COME LE BARRE TEMPO
   (pill arrotondate, micro-gap, riempiono la viewport;
    alcune zone doppie con pattern camminata-metro-metro-camminata)
   ------------------------------------------------------------ */
.metro-decor-v {
  position: fixed;
  left: 10px;
  top: 0px;
  height: 100vh;
  width: 34px;              /* spazio per colonna doppia */
  z-index: 2;
  opacity: 0.90;
  pointer-events: none;
  display: flex;
  flex-direction: column;
  gap: 4px;                 /* come le barre tempo (micro separazione) */
  padding: 10px 0;
  box-sizing: border-box;
}

.metro-decor-v .vrow {
  display: flex;
  flex-direction: row;
  gap: 6px;                 /* distanza tra colonna 1 e colonna 2 */
  align-items: stretch;
}

.metro-decor-v .vseg {
  width: 10px;
  border-radius: 999px;     /* molto stondato */
  border: 1px solid rgba(17,24,39,0.18);
  box-sizing: border-box;
}

.metro-decor-v .vseg.ghost {
  border: none;
  background: transparent;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ============================================================
# DECORAZIONI (VERTICALI, QUASI CONTINUE)
# - pill come le barre tempi
# - nessun colore consecutivo nella stessa colonna
# - colori "sparati" random su TUTTE le linee
# - quando c'è la colonna a destra: pattern fisso
#   camminata -> metro -> metro -> camminata (ripetuto a blocchi)
# ============================================================
def _get_decor_rng() -> random.Random:
    if "decor_seed" not in st.session_state:
        st.session_state.decor_seed = random.randint(1, 10_000_000)
    return random.Random(st.session_state.decor_seed)


def _pick_metro_color(rng: random.Random, avoid_color: str | None = None) -> str:
    # pesca random da tutte le linee, evitando ripetizione consecutiva
    keys = list(LINE_COLORS.keys())
    for _ in range(50):
        c = LINE_COLORS[rng.choice(keys)]
        if c != avoid_color:
            return c
    # fallback
    return LINE_COLORS[keys[0]]


def render_left_decor_vertical():
    rng = _get_decor_rng()

    # 1) genera altezze (vh) per riempire ~100vh con segmenti piccoli
    heights = []
    total = 0.0
    while total < 100.0:
        h = rng.uniform(1.6, 4.4)  # "trattini" simili alle barre: piccoli e tanti
        if total + h > 100.0:
            h = 100.0 - total
        heights.append(h)
        total += h

    n = len(heights)

    # 2) scegli blocchi "doppi" (sparse), ognuno lungo 4 segmenti esatti
    #    (così la colonna destra può rispettare camminata-metro-metro-camminata)
    double = [False] * n
    i = 0
    while i <= n - 4:
        # probabilità di inizio blocco doppio
        if rng.random() < 0.18:
            # evita sovrapposizioni e troppi blocchi attaccati
            if (i > 0 and double[i - 1]) or (i > 1 and double[i - 2]):
                i += 1
                continue
            for k in range(4):
                double[i + k] = True
            i += 4
        else:
            i += 1

    # 3) colori colonna principale: no due uguali consecutivi
    primary_colors = []
    prev_p = None
    for idx in range(n):
        # un po' di camminata, ma non troppo
        want_walk = (rng.random() < 0.22)

        if want_walk and prev_p != WALK_COLOR:
            c = WALK_COLOR
        else:
            # metro
            c = _pick_metro_color(rng, avoid_color=prev_p)

        # garantisci no consecutivi uguali
        if c == prev_p:
            if c == WALK_COLOR:
                c = _pick_metro_color(rng, avoid_color=prev_p)
            else:
                c = WALK_COLOR if prev_p != WALK_COLOR else _pick_metro_color(rng, avoid_color=prev_p)

        primary_colors.append(c)
        prev_p = c

    # 4) colori colonna destra SOLO nei blocchi doppi
    #    pattern: walk, metro, metro, walk (ripetuto per ogni blocco)
    secondary_colors = [None] * n
    idx = 0
    prev_s = None
    while idx < n:
        if not double[idx]:
            prev_s = None  # quando torna singolo, "reset" per evitare vincoli strani
            idx += 1
            continue

        # siamo dentro un blocco doppio: individua l'inizio del blocco (4 segmenti)
        start = idx
        # assicurati di partire davvero dall'inizio del blocco
        while start > 0 and double[start - 1]:
            start -= 1

        # applica pattern sui 4
        pattern = ["walk", "metro", "metro", "walk"]
        for k in range(4):
            j = start + k
            if j >= n or not double[j]:
                continue

            if pattern[k] == "walk":
                c = WALK_COLOR
            else:
                # metro: evita ripetizione consecutiva nella colonna destra
                # e (se possibile) evita uguale al primario in quel punto
                avoid = prev_s
                c = _pick_metro_color(rng, avoid_color=avoid)
                if c == primary_colors[j]:
                    c = _pick_metro_color(rng, avoid_color=primary_colors[j])

            # ulteriore check no consecutivi
            if c == prev_s:
                if c == WALK_COLOR:
                    c = _pick_metro_color(rng, avoid_color=prev_s)
                else:
                    c = WALK_COLOR if prev_s != WALK_COLOR else _pick_metro_color(rng, avoid_color=prev_s)

            secondary_colors[j] = c
            prev_s = c

        idx = start + 4

    # 5) HTML
    rows = []
    for h, c1, c2 in zip(heights, primary_colors, secondary_colors):
        # secondaria: solo quando presente; altrimenti ghost per non “saltare” troppo visivamente
        s1 = f"<div class='vseg' style='height:{h:.3f}vh; background:{c1};'></div>"
        if c2 is None:
            s2 = "<div class='vseg ghost' style='height:{:.3f}vh;'></div>".format(h)
        else:
            s2 = f"<div class='vseg' style='height:{h:.3f}vh; background:{c2};'></div>"

        rows.append(f"<div class='vrow'>{s1}{s2}</div>")

    st.markdown(f"<div class='metro-decor-v'>{''.join(rows)}</div>", unsafe_allow_html=True)


render_left_decor_vertical()

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


def address_autocomplete(label: str, key: str, placeholder: str):
    """
    Autocomplete in tema chiaro: text_input + selectbox (niente componenti custom),
    così il dropdown resta sempre chiaro.
    """
    map_key = f"{key}__map"
    query_key = f"{key}__query"
    choice_key = f"{key}__choice"
    last_q_key = f"{key}__lastq"

    q = st.text_input(
        label if label else "",
        key=query_key,
        placeholder=placeholder,
        label_visibility="collapsed" if not label else "visible",
    )

    prev_q = st.session_state.get(last_q_key, "")
    if q != prev_q:
        st.session_state[choice_key] = ""
    st.session_state[last_q_key] = q

    mp = {}
    opts = []
    if q and len(q.strip()) >= 3:
        results = geopf_completion(q.strip(), terr="75", maximumResponses=8)
        for r in results:
            ft = r.get("fulltext")
            x = r.get("x")
            y = r.get("y")
            if ft and x is not None and y is not None:
                mp[ft] = (float(x), float(y))
                opts.append(ft)

    st.session_state[map_key] = mp

    selected = None
    coords = None

    if opts:
        selected_opt = st.selectbox(
            "",
            options=[""] + opts,
            key=choice_key,
            label_visibility="collapsed",
        )
        if selected_opt:
            selected = selected_opt
            coords = mp.get(selected_opt)

    return selected, coords


# ============================================================
# ROUTE BARS (HTML/CSS)
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

    keys_start = [
        "walk_time_start_min",
        "walk_start_time_min",
        "walk_start_min",
        "walk_min_start",
        "walk_time_min_start",
    ]
    keys_end = [
        "walk_time_end_min",
        "walk_end_time_min",
        "walk_end_min",
        "walk_min_end",
        "walk_time_min_end",
    ]

    w_start = None
    w_end = None

    for k in keys_start:
        if k in details and details.get(k) is not None:
            w_start = float(details.get(k) or 0.0)
            break
    for k in keys_end:
        if k in details and details.get(k) is not None:
            w_end = float(details.get(k) or 0.0)
            break

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
        st.info("Nessun percorso disponibile da visualizzare.")
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
      <div class="legend-item">{_legend_pill("Camminata", WALK_COLOR)}</div>
      {"".join([f'<div class="legend-item">{_legend_pill("Metro " + str(l), LINE_COLORS.get(l, "#666"))}</div>' for l in sorted(list(used_lines), key=_line_sort_key)])}
    </div>
    """

    rows = []
    for i, total, segs in precomp:
        name = f"Amico {i+1}"
        seg_html = ""

        for s in segs:
            dt = float(s.get("time_min", 0.0) or 0.0)
            if dt <= 0:
                continue

            w_pct = (dt / max_total) * 100.0
            if s["kind"] == "walk":
                color = WALK_COLOR
                title = f"Camminata: {fmt_min(dt)} min"
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
# GINI BAR (components.html)
# ============================================================
def gini_to_color_hex(v):
    v = float(np.clip(v, 0.0, 1.0))
    green = np.array([34, 197, 94])   # #22c55e
    amber = np.array([245, 158, 11])  # #f59e0b
    red = np.array([239, 68, 68])     # #ef4444

    if v <= 0.55:
        t = v / 0.55 if 0.55 else 0
        rgb = green + (amber - green) * t
    else:
        t = (v - 0.55) / (1 - 0.55)
        rgb = amber + (red - amber) * t

    rgb = np.clip(rgb.round().astype(int), 0, 255)
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def render_gini_bar(gini_value: float):
    st.header("Indice di Gini (Disuguaglianza)")

    if not np.isfinite(gini_value):
        st.warning("Valore Gini non disponibile.")
        return

    v = float(np.clip(gini_value, 0.0, 1.0))
    pct = v * 100.0
    color = gini_to_color_hex(v)

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  .wrap {{
    max-width: 980px;
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    color: rgba(17,24,39,0.92);
  }}
  .bar {{
    position: relative;
    height: 18px;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e 0%, #f59e0b 55%, #ef4444 100%);
    border: 1px solid rgba(17,24,39,0.18);
  }}
  .marker {{
    position:absolute;
    left:{pct:.2f}%;
    top:-20px;
    transform: translateX(-50%);
    font-weight: 900;
    font-size: 16px;
    color: rgba(17,24,39,0.95);
  }}
  .tick {{
    position:absolute;
    left:{pct:.2f}%;
    top:0px;
    transform: translateX(-50%);
    width: 2px;
    height: 18px;
    background: rgba(17,24,39,0.95);
  }}
  .labels {{
    display:flex;
    justify-content:space-between;
    margin-top:6px;
    font-size: 13px;
    color: rgba(17,24,39,0.78);
  }}
  .value {{
    margin-top: 10px;
    font-size: 22px;
    font-weight: 900;
    color: {color};
  }}
</style>
</head>
<body style="margin:0;background:transparent;">
  <div class="wrap">
    <div class="bar">
      <div class="marker">▼</div>
      <div class="tick"></div>
    </div>
    <div class="labels">
      <div>Massima uguaglianza</div>
      <div>Massima disuguaglianza</div>
    </div>
    <div class="value">{v:.4f}</div>
  </div>
</body>
</html>
"""
    components.html(html, height=115, scrolling=False)


# ============================================================
# HEADER
# ============================================================
st.markdown("<div class='pg-title'>pariGINI</div>", unsafe_allow_html=True)
st.markdown(
    """
I tuoi amici ti propongono un bar lontano? Calcola quanto è equa la scelta.  
Misura la disuguaglianza dei tempi di spostamento in metro usando il Gini Index.  
Inserisci da dove partite (minimo 2 amici) e dove andate: il Gini viene calcolato automaticamente.
"""
)

# ============================================================
# LOAD GRAPH (cache)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_graph():
    G_ = build_graph_from_edgelist("./timed_edgelist.geojson")
    idx_ = build_node_index(G_)
    return G_, idx_


with st.spinner("Caricamento rete metro..."):
    try:
        G, node_index = load_graph()
    except Exception as e:
        st.error(f"Errore nel caricamento rete: {e}")
        st.stop()

# ============================================================
# PARAMETRI ROUTING FISSI (non mostrati)
# ============================================================
max_line_changes = 1
change_penalty_min = 2.0

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
st.header("Da dove partite?")

top_row = st.columns([1, 1, 8])
with top_row[0]:
    st.button("Aggiungi", use_container_width=True, on_click=add_friend)
with top_row[1]:
    st.button(
        "Rimuovi",
        use_container_width=True,
        disabled=(st.session_state.n_friends <= 2),
        on_click=remove_friend,
    )
with top_row[2]:
    st.caption("Minimo 2 amici. Scrivi e seleziona un suggerimento dall’autocompletamento.")

starts = []
for i in range(st.session_state.n_friends):
    row = st.columns([1, 7])
    with row[0]:
        st.markdown(f"**Amico {i+1}**")
    with row[1]:
        _, coords = address_autocomplete(
            label="",
            key=f"friend_{i}",
            placeholder="Es: Gare de Lyon, 75012 Paris • 10 Rue de Rivoli 75004 Paris",
        )
        if coords:
            starts.append(coords)

# ============================================================
# INPUT: TARGET
# ============================================================
st.header("Dove andate?")

_, target_coords = address_autocomplete(
    label="",
    key="target",
    placeholder="Es: Tour Eiffel • Louvre • Châtelet • 20 Avenue de ... 750xx Paris",
)
target = target_coords

# ============================================================
# VALIDAZIONE (min 2 amici + target)
# ============================================================
ready = (target is not None) and (len(starts) >= 2)

if not ready:
    problems = []
    if len(starts) < 2:
        problems.append("almeno 2 amici (con indirizzo selezionato)")
    if target is None:
        problems.append("destinazione (seleziona un suggerimento)")
    st.warning("Manca: " + ", ".join(problems) + ".")

# ============================================================
# ANALYSIS & RESULTS
# ============================================================
if st.button(
    "Calcola Gini",
    type="primary",
    use_container_width=True,
    disabled=not ready,
):
    st.divider()

    with st.spinner("Calcolo tempi di percorrenza..."):
        try:
            results_df, metrics = accessibility_inequality_to_target(
                G,
                starts,
                target,
                node_index=node_index,
                max_line_changes=max_line_changes,
                change_penalty_min=change_penalty_min,
                max_walk_min_start=15.0,
                max_walk_min_end=15.0,
                max_candidate_stations=25,
                allow_walk_only=True,
                keep_details=True,
            )
        except Exception as e:
            st.error(f"Errore nel calcolo: {e}")
            st.stop()

    gini_value = float(metrics.get("gini_time", np.nan))
    render_gini_bar(gini_value)
    render_routes_html(results_df)

    if np.isfinite(gini_value):
        verdict = "abbastanza equo" if gini_value <= 0.2 else "poco equo"
        st.info(
            f"""
**Cosa significa il Gini Index?**

- Gini = 0: tutti i tempi di percorrenza sono uguali  
- Gini = 1: massima disuguaglianza possibile  
- Gini alto: tempi molto diversi, accessibilità disuguale  
- Gini basso: tempi simili, accessibilità più uguale  

In questo caso: Gini = {gini_value:.4f} → {verdict}.
"""
        )
    else:
        st.info(
            """
**Cosa significa il Gini Index?**

- Gini = 0: tutti i tempi di percorrenza sono uguali  
- Gini = 1: massima disuguaglianza possibile  
- Gini alto: tempi molto diversi, accessibilità disuguale  
- Gini basso: tempi simili, accessibilità più uguale
"""
        )

    st.subheader("Statistiche tempi di percorrenza")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Minimo", f"{fmt_min(metrics.get('min_time_min', np.nan))} min")
    with c2:
        st.metric("Medio", f"{fmt_min(metrics.get('mean_time_min', np.nan))} min")
    with c3:
        st.metric("Massimo", f"{fmt_min(metrics.get('max_time_min', np.nan))} min")

    with st.expander("Esporta risultati", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Scarica CSV (dettagli)",
                data=csv,
                file_name="metro_accessibility_results.csv",
                mime="text/csv",
            )

        with col2:
            summary = {
                "gini_index": float(gini_value) if pd.notna(gini_value) else None,
                "mean_time_min": float(metrics.get("mean_time_min", np.nan)),
                "min_time_min": float(metrics.get("min_time_min", np.nan)),
                "max_time_min": float(metrics.get("max_time_min", np.nan)),
                "n_successful": int(metrics.get("n_ok", 0)),
                "n_total": int(metrics.get("n_total", 0)),
                "target_lon": float(target[0]),
                "target_lat": float(target[1]),
                "n_starts": int(len(starts)),
                "routing": {
                    "max_line_changes": max_line_changes,
                    "change_penalty_min": change_penalty_min,
                    "max_walk_min_start": 15.0,
                    "max_walk_min_end": 15.0,
                    "max_candidate_stations": 25,
                    "allow_walk_only": True,
                },
            }
            json_data = json.dumps(summary, indent=2)
            st.download_button(
                label="Scarica JSON (sintesi)",
                data=json_data,
                file_name="metro_accessibility_summary.json",
                mime="application/json",
            )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    """
---
**pariGINI** | Routing Dijkstra con cambi linea ottimizzati  
Dati: RATP Metro Network (timed_edgelist.geojson)  
Autocomplete: Géoplateforme (IGN) - completion
"""
)