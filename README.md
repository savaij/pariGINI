# 🚇 Metro Paris Gini Accessibility Analysis

## Panoramica

Questa cartella contiene un'analisi della **disuguaglianza di accessibilità** della metropolitana parigina usando il **Gini Index**.

### File principali:

1. **`gini_paris_distances_calculations.py`** - Modulo core con tutte le funzioni di calcolo:
   - Caricamento grafo metro
   - Routing Dijkstra con cambii linea ottimizzati
   - Metriche di disuguaglianza (Gini, Theil)

2. **`streamlit_app.py`** - Applicazione web interattiva (Streamlit):
   - Interfaccia per inserire coordinate di partenza e destinazione
   - Calcolo automatico del Gini Index
   - Visualizzazione risultati con spiegazioni

## Installazione

```bash
pip install streamlit geopandas networkx pandas numpy shapely
```

## Utilizzo

### Avviare l'app Streamlit:

```bash
streamlit run streamlit_app.py
```

L'app si aprirà nel browser a `http://localhost:8501`

### Utilizzare il modulo come libreria:

```python
from gini_paris_distances_calculations import (
    build_graph_from_edgelist,
    build_node_index,
    accessibility_inequality_to_target,
)

# Carica grafo
G = build_graph_from_edgelist("./timed_edgelist.geojson")
node_index = build_node_index(G)

# Definisci punti di partenza e destinazione
starts = [(2.30, 48.88), (2.33, 48.86), (2.25, 48.84)]
target = (2.377442453169209, 48.84950447433732)

# Calcola accessibilità
results_df, metrics = accessibility_inequality_to_target(
    G, 
    starts, 
    target, 
    node_index=node_index,
    max_line_changes=1
)

print(f"Gini Index: {metrics['gini_time']:.4f}")
print(f"Tempo medio: {metrics['mean_time_min']:.1f} minuti")
```

## 📊 Interpretazione del Gini Index

| Valore Gini | Interpretazione |
|------------|-----------------|
| **0.0** | Perfetta uguaglianza - tutti hanno stesso tempo di percorrenza |
| **0.05-0.10** | Molto eguale - tempi molto simili |
| **0.10-0.20** | Abbastanza eguale - discrepanze minori |
| **0.20-0.30** | Moderatamente disuguale - alcune differenze significative |
| **> 0.30** | Molto disuguale - forti disparità di accessibilità ⚠️ |

**Regola semplice:** Più **alto** il Gini → Più **disuguale** l'accessibilità

## 📈 Metriche disponibili

- **Gini Index**: Misura classica di disuguaglianza (0-1)
- **Theil Index**: Indice entropico alternativo
- **Tempi di percorrenza**: Min, mediana, media, max, p90
- **Cambii linea**: Numero di cambiamenti per ogni percorso
- **Distanze snapping**: Distanze a piedi dai punti di input ai nodi metro

## 🗺️ Dati richiesti

- `timed_edgelist.geojson` - Grafo della metropolitana con:
  - Stazioni come nodi
  - Archi tra stazioni consecutive
  - Tempi di percorrenza calcolati
  - Informazioni di linea

## 🛠️ Configurazione avanzata

Nell'app Streamlit puoi regolare:
- **Cambii linea massimi**: Numero di cambiamenti permessi (0-3)
- **Penalità per cambio**: Minuti aggiunti per ogni cambio (default: 2.0)

## 📝 Note

- Il routing usa Dijkstra su uno spazio di stato espanso: `(nodo, linea_attuale, numero_cambi)`
- I tempi di percorrenza includono il tempo di camminata per raggiungere la stazione più vicina (snapping)
- Velocità di camminata: 4 km/h di default

## 🎯 Esempi di utilizzo

### Esempio 1: Una sola origine

```python
starts = [(2.30, 48.88)]
```

### Esempio 2: Griglia di punti

```python
import numpy as np
lons = np.linspace(2.2, 2.5, 10)
lats = np.linspace(48.8, 49.0, 10)
starts = [(lon, lat) for lon in lons for lat in lats]
```

### Esempio 3: Con parametri custom

```python
results_df, metrics = accessibility_inequality_to_target(
    G, starts, target,
    node_index=node_index,
    max_line_changes=0,  # Solo linea diretta
    change_penalty_min=5.0  # Penalità più alta per i cambi
)
```

---

**Creato per analizzare l'equità di accesso alla metropolitana parigina** 🚇
