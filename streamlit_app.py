"""
Streamlit App: Metro Paris Accessibility Inequality Analysis
Calcola e visualizza il Gini Index per l'accessibilità del metro parigino

UX nuova:
- Sezione "Da dove partite?" con Amico 1..N (minimo 2), aggiungi/rimuovi
- Ogni campo è un autocomplete "in-box" (streamlit-searchbox) su Géoplateforme (IGN)
- Sezione "Dove andate?" con target (autocomplete)
"""

import json
import requests
import streamlit as st
import pandas as pd
import numpy as np

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
def geopf_completion(text: str, terr: str = "75", maximumResponses: int = 8,
                     types: str = "StreetAddress,PositionOfInterest"):
    """
    Chiama l'endpoint completion e ritorna la lista 'results'.
    Ogni result tipicamente ha: fulltext, x (lon), y (lat), kind, zipcode, ...
    """
    text = (text or "").strip()
    if not text:
        return []

    r = requests.get(
        COMPLETION_URL,
        params={
            "text": text,
            "terr": terr,  # 75 = Paris (département)
            "type": types,  # StreetAddress + POI
            "maximumResponses": maximumResponses,
        },
        timeout=10,
        headers={"User-Agent": "streamlit-metro-paris-gini/1.0"},
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
        if not searchterm or len(searchterm.strip()) < 3:
            st.session_state[map_key] = {}
            return []

        results = geopf_completion(searchterm, terr="75", maximumResponses=8)
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
    """
    Un campo autocomplete "vero" (una sola casella).
    Ritorna (selected_text, (lon,lat) or None)
    """
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
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Metro Paris - Gini Accessibility",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚇 Metro Paris - Accessibility Inequality Analysis")
st.markdown(
    """
Analizza la disuguaglianza di accessibilità al metro parigino usando il **Gini Index**.  
Inserisci **da dove partite** (min 2 amici) e **dove andate** con autocompletamento.
"""
)

# ============================================================
# SIDEBAR: Caricamento grafo
# ============================================================
with st.sidebar:
    st.header("⚙️ Configurazione")

    with st.spinner("Caricamento grafo metro parigino..."):
        try:
            G = build_graph_from_edgelist("./timed_edgelist.geojson")
            node_index = build_node_index(G)
            st.success("✅ Grafo caricato!")
        except Exception as e:
            st.error(f"❌ Errore nel caricamento: {e}")
            st.stop()

    st.divider()

    st.subheader("Parametri Routing")
    max_line_changes = st.slider("Cambii linea massimi", 0, 3, 1)
    change_penalty_min = st.number_input(
        "Penalità per cambio (minuti)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )

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

top_row = st.columns([1, 1, 6])
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
    st.caption("Minimo 2 persone. Scrivi e scegli un suggerimento dall’autocompletamento.")

starts = []
missing_friends = []

for i in range(st.session_state.n_friends):
    row = st.columns([1, 7])
    with row[0]:
        st.markdown(f"**Amico {i+1}**")
    with row[1]:
        sel, coords = address_autocomplete(
            label="",
            key=f"friend_{i}",
            placeholder="Es: Gare de Lyon, 75012 Paris  •  oppure: 10 Rue de Rivoli 75004 Paris",
        )
        if coords:
            starts.append(coords)
        else:
            # se non ha selezionato nulla (o testo troppo corto)
            missing_friends.append(f"Amico {i+1}")

if len(starts) >= 1:
    st.caption(f"✅ Origini valide: {len(starts)} / {st.session_state.n_friends}")

# ============================================================
# INPUT: TARGET
# ============================================================
st.header("🎯 Dove andate?")

target_sel, target_coords = address_autocomplete(
    label="",
    key="target",
    placeholder="Es: Tour Eiffel  •  Louvre  •  Châtelet  •  20 Avenue de ... 750xx Paris",
)

target = target_coords
if target:
    st.info(f"📌 Target: ({target[0]:.6f}, {target[1]:.6f})")
else:
    st.caption("Seleziona un suggerimento per fissare la destinazione.")

# ============================================================
# VALIDAZIONE (min 2 amici + target)
# ============================================================
ready = (target is not None) and (len(starts) >= 2)

if not ready:
    problems = []
    if target is None:
        problems.append("destinazione")
    if len(starts) < 2:
        problems.append("almeno 2 amici (origini valide)")

    st.warning("Manca: " + ", ".join(problems) + ".")

# ============================================================
# ANALYSIS & RESULTS
# ============================================================
if st.button(
    "🚀 Calcola Accessibilità",
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
                keep_details=False,
            )
        except Exception as e:
            st.error(f"❌ Errore nel calcolo: {e}")
            st.stop()

    # =====================================
    # SECTION 1: Gini Index & Explanation
    # =====================================
    st.header("📊 Indice di Gini (Disuguaglianza)")

    col1, col2, col3 = st.columns(3)
    gini_value = metrics.get("gini_time", np.nan)

    with col1:
        if gini_value < 0.1:
            interpretation = "Molto Eguale"
        elif gini_value < 0.2:
            interpretation = "Abbastanza Eguale"
        else:
            interpretation = "Disuguale"

        st.metric("Gini Index", f"{gini_value:.4f}", delta=interpretation, delta_color="inverse")

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
        ["i", "start_lon", "start_lat", "total_time_min", "metro_time_min", "walk_time_min", "line_changes", "ok"]
    ].copy()

    display_df.columns = [
        "ID", "Lon", "Lat", "Tempo Tot (min)",
        "Metro (min)", "Camminata (min)", "Cambiamenti", "Successo"
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
**🚇 Metro Paris Gini Analysis** | Basato su routing Dijkstra con cambii linea ottimizzati  
📊 Misura di disuguaglianza: Gini Index + Theil Index  
📍 Dati: RATP Metro Network (timed_edgelist.geojson)  
🧭 Autocomplete: Géoplateforme (IGN) - completion
"""
)
