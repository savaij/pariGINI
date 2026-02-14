"""
Streamlit App: pariGINI
Calcola e visualizza il Gini Index per la disuguaglianza di accessibilità in metro a Parigi

UX:
- "Da dove partite?" con Amico 1..N (minimo 2), aggiungi/rimuovi
- Autocomplete "in-box" su Géoplateforme (IGN)
- "Dove andate?" con target (autocomplete)
- Parametri routing fissi: max_line_changes=1, change_penalty_min=2.0 (non mostrati)
- NO MAP: grafica “a barre” per ogni amico:
  - camminata (grigio)
  - tratte metro colorate per linea (e segmentate se cambia linea)
  - lunghezze proporzionali ai minuti reali
"""

import json
import requests
import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from streamlit_searchbox import st_searchbox  # pip install streamlit-searchbox

from gini_paris_distances_calculations import (
    build_graph_from_edgelist,
    build_node_index,
    accessibility_inequality_to_target,
)

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
    """
    Crea una search_function per streamlit-searchbox.
    Salva in session_state[map_key] una mappa fulltext -> (lon,lat)
    """
    def _search(searchterm: str):
        raw = (searchterm or "").strip()
        if not raw or len(raw) < 3:
            st.session_state[map_key] = {}
            return []

        # Aggiunge "paris" automaticamente se non è già presente (case-insensitive)
        if "paris" not in raw.lower():
            query = f"{raw} paris"
        else:
            query = raw

        results = geopf_completion(query, terr="75", maximumResponses=8)
        mp = {}
        opts = []
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
# ROUTE BAR (walk + metro segments)
# ============================================================
LINE_COLORS = {
    "1":  "#FFCD00",
    "2":  "#003CA6",
    "3":  "#7A8B2E",
    "3bis": "#8E9AE6",
    "4":  "#7C2E83",
    "5":  "#FF7E2E",
    "6":  "#6EC4B1",
    "7":  "#FA9ABA",
    "7bis": "#6EC4B1",
    "8":  "#CEADD2",
    "9":  "#B7D84B",
    "10": "#C9910D",
    "11": "#704B1C",
    "12": "#007852",
    "13": "#8E9AE6",
    "14": "#62259D",
}
WALK_COLOR = "#9CA3AF"  # grigio


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
    """
    Converte lista edges (con 'line' e 'time_min') in segmenti consecutivi per linea:
    ritorna list di dict: {kind: "metro", line: "8", time_min: 12.3}
    """
    segs = []
    cur_line = None
    cur_t = 0.0

    for e in edges:
        line = e.get("line")
        t = float(e.get("time_min", 0.0))
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


def build_segments_for_friend(details):
    """
    Segmenti ordinati che compongono la barra:
    - walk (grigio): tempo walk totale (start + end) o walk-only
    - metro: segmenti per linea (colori RATP)
    """
    mode = details.get("mode", "metro_walk")
    if mode == "walk_only":
        return [{"kind": "walk", "time_min": float(details.get("walk_time_min", 0.0))}]

    segs = []
    walk_t = float(details.get("walk_time_min", 0.0))
    if walk_t > 0:
        segs.append({"kind": "walk", "time_min": walk_t})

    edges = details.get("edges", []) or []
    segs.extend(compress_edges_to_line_segments(edges))
    return segs


def plot_routes_bars(results_df):
    """
    Grafico tipo "Gantt" orizzontale:
    ogni riga = un amico
    segmenti = walk (grigio) + metro (colorato per linea)
    lunghezza segmenti proporzionale ai minuti.
    """
    ok_df = results_df[results_df["ok"] == True].copy()
    if ok_df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Nessun percorso disponibile da visualizzare",
        )
        return fig

    # ordine righe: Amico 1,2,3...
    ok_df = ok_df.sort_values("i")

    # asse y: nomi amici
    friend_names = [f"Amico {int(i)+1}" for i in ok_df["i"].tolist()]

    # per altezza figura: ~50px per amico
    height = max(320, 60 + 55 * len(friend_names))

    fig = go.Figure()

    # costruisco rettangoli con bar "stacked" usando scatter (linee spesse)
    # y_pos: numerico per controllo più preciso
    y_pos = list(range(len(friend_names)))
    y_map = {friend_names[k]: y_pos[k] for k in range(len(friend_names))}

    # per legenda pulita: 1 entry walk + 1 entry per linea usata
    shown_legend = set()

    for _, row in ok_df.iterrows():
        i = int(row["i"])
        name = f"Amico {i+1}"
        y = y_map[name]
        details = row["details"]

        segments = build_segments_for_friend(details)
        x0 = 0.0

        # hover summary
        total = float(details.get("total_time_min", row.get("total_time_min", 0.0)))
        mode = details.get("mode", "metro_walk")

        for seg in segments:
            dt = float(seg["time_min"])
            if dt <= 0:
                continue

            x1 = x0 + dt

            if seg["kind"] == "walk":
                color = WALK_COLOR
                label = "Camminata"
                hover = (
                    f"<b>{name}</b><br>"
                    f"Camminata: {dt:.1f} min<br>"
                    f"Totale: {total:.1f} min<br>"
                    f"Mode: {mode}<extra></extra>"
                )
            else:
                line = seg.get("line", "?")
                line_key = _norm_line_for_color(line) or "?"
                color = LINE_COLORS.get(line_key, "#666666")
                label = f"Metro {line_key}"
                hover = (
                    f"<b>{name}</b><br>"
                    f"Metro linea {line}<br>"
                    f"Segmento: {dt:.1f} min<br>"
                    f"Totale: {total:.1f} min<extra></extra>"
                )

            showlegend = label not in shown_legend
            if showlegend:
                shown_legend.add(label)

            # Segmento come linea spessa (più semplice e “pulito” in Streamlit)
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y, y],
                    mode="lines",
                    line=dict(color=color, width=18),
                    name=label,
                    showlegend=showlegend,
                    hovertemplate=hover,
                )
            )

            x0 = x1

        # etichetta tempo totale a destra (solo testo)
        fig.add_trace(
            go.Scatter(
                x=[x0 + 0.5],
                y=[y],
                mode="text",
                text=[f"{total:.1f} min"],
                textposition="middle right",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        title="Tragitti (camminata + metro per linea) — lunghezze proporzionali ai minuti \n",
        xaxis=dict(
            title="Minuti",
            zeroline=False,
            showgrid=True,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_pos,
            ticktext=friend_names,
            autorange="reversed",  # Amico 1 in alto
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="pariGINI",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🚇 pariGINI ")
st.markdown(
    """
I tuoi amic ti propongono un bar lontano? calcola quanto è equa la scelta.  
Calcola la disuguaglianza di spostamento in metro usando il Gini Index.  
Inserisci da dove partite (min 2 amic) e dove andate, il Gini si calcola da se.
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
        st.error(f"❌ Errore nel caricamento rete: {e}")
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
st.header("🧑‍🤝‍🧑 Da dove partite?")

top_row = st.columns([1, 1, 8])
with top_row[0]:
    st.button("➕ Aggiungi", use_container_width=True, on_click=add_friend)
with top_row[1]:
    st.button(
        "➖ Rimuovi",
        use_container_width=True,
        disabled=(st.session_state.n_friends <= 2),
        on_click=remove_friend,
    )
with top_row[2]:
    st.caption("Minimo 2 amic. Scrivi e seleziona un suggerimento dall’autocompletamento.")

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
st.header("🎯 Dove andate?")

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
        problems.append("almeno 2 amic (con indirizzo selezionato)")
    if target is None:
        problems.append("destinazione (seleziona un suggerimento)")
    st.warning("Manca: " + ", ".join(problems) + ".")

# ============================================================
# ANALYSIS & RESULTS
# ============================================================
if st.button(
    "🚀 Calcola Gini",
    type="primary",
    use_container_width=True,
    disabled=not ready,
):
    st.divider()

    with st.spinner("Calcolo tempi di percorrenza..."):
        try:
            # qui assumo che nel file gini_paris_distances_calculations.py
            # tu abbia già implementato l'opzione “walk range” (15 min) e walk-only.
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
                keep_details=True,  # necessario per segmenti linee
            )
        except Exception as e:
            st.error(f"❌ Errore nel calcolo: {e}")
            st.stop()

    # =====================================
    # ROUTE BARS (NO MAP)
    # =====================================
    st.subheader("🧾 Tragitti (grafica proporzionale ai minuti)")
    bars_fig = plot_routes_bars(results_df)
    st.plotly_chart(bars_fig, use_container_width=True, config={"displayModeBar": False})

    # =====================================
    # SECTION 1: Gini Index & Explanation
    # =====================================
    st.header("📊 Indice di Gini (Disuguaglianza)")

    col1, col2, col3 = st.columns(3)
    gini_value = metrics.get("gini_time", np.nan)

    # Se gini_value è NaN, evitiamo confronti e mostriamo un messaggio neutro
    if pd.isna(gini_value):
        interpretation = "N/D"
        status = "info"
    else:
        if gini_value < 0.1:
            interpretation = "Molto Eguale"
            status = "success"   # verde
        elif gini_value < 0.2:
            interpretation = "Abbastanza Eguale"
            status = "warning"   # giallo
        else:
            interpretation = "Disuguale"
            status = "error"     # rosso

    with col1:
        # Mostro il numero come metrica (senza delta_color, perché non colora per testo)
        st.metric("Gini Index", f"{gini_value:.4f}" if pd.notna(gini_value) else "N/D")

        # E sotto mostro l’interpretazione con colore certo
        if status == "success":
            st.success(interpretation)
        elif status == "warning":
            st.warning(interpretation)
        elif status == "error":
            st.error(interpretation)
        else:
            st.info(interpretation)

    with col2:
        st.metric("Punti analizzati", metrics["n_ok"])

    with col3:
        st.metric(
            "Tempo medio",
            f"{metrics['mean_time_min']:.1f} min",
            delta=f"Max: {metrics['max_time_min']:.1f} min",
        )

    st.info(
        """
📖 **Cosa significa il Gini Index?**

- **Gini = 0**: Tutti i tempi di percorrenza sono **uguali** ✅
- **Gini = 1**: Massima disuguaglianza possibile ❌
- **Gini alto**: tempi molto diversi → **accessibilità disuguale**
- **Gini basso**: tempi simili → **accessibilità più eguale**

**In questo caso: Gini = {:.4f}** → {}
""".format(gini_value, "DISUGUALE ⚠️" if gini_value > 0.2 else "ABBASTANZA EGUALE ✅")
    )

    # =====================================
    # SECTION 2: Detailed Results Table
    # =====================================
    st.subheader("📋 Dettagli Percorsi")

    display_df = results_df[
        ["i", "start_lon", "start_lat", "total_time_min", "metro_time_min", "walk_time_min",
         "line_changes", "mode", "ok", "error"]
    ].copy()

    display_df.columns = [
        "ID", "Lon", "Lat", "Tempo Tot (min)", "Metro (min)", "Camminata (min)",
        "Cambiamenti", "Mode", "Successo", "Errore"
    ]
    display_df["Successo"] = display_df["Successo"].map({True: "✅", False: "❌"})

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # =====================================
    # SECTION 3: Statistics
    # =====================================
    st.subheader("📈 Statistiche Tempi di Percorrenza")

    col1, col2 = st.columns(2)

    with col1:
        stat_data = {
            "Metrica": ["Tempo Minimo", "Tempo Mediano", "Tempo Medio", "Tempo Massimo", "Percentile 90%"],
            "Valore (min)": [
                f"{metrics['min_time_min']:.2f}",
                f"{metrics['median_time_min']:.2f}",
                f"{metrics['mean_time_min']:.2f}",
                f"{metrics['max_time_min']:.2f}",
                f"{metrics['p90_time_min']:.2f}",
            ],
        }
        st.table(pd.DataFrame(stat_data))

    with col2:
        theil = metrics.get("theil_time", np.nan)
        st.metric("Indice Theil", f"{theil:.4f}", help="Misura alternativa di disuguaglianza (0=perfetta uguaglianza)")
        st.metric(
            "Successo routing",
            f"{metrics['n_ok']}/{metrics['n_total']}",
            help=f"{metrics['share_ok']*100:.1f}% dei percorsi calcolati con successo",
        )

    # =====================================
    # SECTION 4: Export Data
    # =====================================
    st.divider()
    st.subheader("💾 Esporta Risultati")

    col1, col2 = st.columns(2)

    with col1:
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Scarica CSV (dettagli)",
            data=csv,
            file_name="metro_accessibility_results.csv",
            mime="text/csv",
        )

    with col2:
        summary = {
            "gini_index": float(gini_value) if pd.notna(gini_value) else None,
            "theil_index": float(theil) if pd.notna(theil) else None,
            "mean_time_min": float(metrics["mean_time_min"]),
            "median_time_min": float(metrics["median_time_min"]),
            "min_time_min": float(metrics["min_time_min"]),
            "max_time_min": float(metrics["max_time_min"]),
            "n_successful": int(metrics["n_ok"]),
            "n_total": int(metrics["n_total"]),
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
            label="📥 Scarica JSON (sintesi)",
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
**pariGINI** | Basato su routing Dijkstra con cambi linea ottimizzati  
📊 Misura di disuguaglianza: Gini Index + Theil Index  
📍 Dati: RATP Metro Network (timed_edgelist.geojson)  
🧭 Autocomplete: Géoplateforme (IGN) - completion
"""
)