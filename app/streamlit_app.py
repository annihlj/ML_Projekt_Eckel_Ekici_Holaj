"""
streamlit_app.py
================

Interaktive Demo-Oberfläche für das Kreditrisiko-Modell.

FUNKTIONEN (wie in der Aufgabenstellung gefordert):
- Beispielperson auswählen ODER zentrale Werte manuell anpassen
- vorhergesagte Ausfallwahrscheinlichkeit
- Risikoklasse (niedrig / mittel / hoch)
- zugehöriges Cluster
- wichtigste Einflussfaktoren
- deutlicher Disclaimer

START (im Projektwurzelverzeichnis, mit aktivierter virtueller Umgebung):
    streamlit run app/streamlit_app.py

Die App lädt die zuvor gespeicherten Artefakte (Modell + Clustering + Feature-Matrix)
und trainiert NICHTS neu.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Projektwurzel zum Pfad hinzufügen (App liegt in app/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app_utils as au  # noqa: E402


# ---------------------------------------------------------------------------
# Seitenkonfiguration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Kreditrisiko-Demo", page_icon="📊", layout="wide")

st.markdown("""
<style>
    /* Basis-Schriftgröße */
    html, body { font-size: 18px; }

    /* Allgemeiner Fließtext */
    p, li, div.stMarkdown { font-size: 1.05rem !important; }

    /* Sidebar & Widget-Labels */
    .stRadio label,
    .stNumberInput label,
    .stSelectbox label,
    [data-testid="stWidgetLabel"] { font-size: 1.1rem !important; }

    /* Metric-Titel (z. B. "Ausfallwahrscheinlichkeit") */
    [data-testid="stMetricLabel"] { font-size: 1.15rem !important; }

    /* Metric-Werte (große Zahlen) */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }

    /* Captions */
    .stCaption,
    [data-testid="stCaptionContainer"] { font-size: 0.95rem !important; }

    /* Überschriften */
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.6rem !important; }
    h3 { font-size: 1.35rem !important; }
</style>
""", unsafe_allow_html=True)


# Streamlit-Caching: Artefakte nur einmal laden (Performance).
@st.cache_resource
def _load():
    return au.load_artifacts()


def main() -> None:
    st.title("📊 Kreditausfallrisiko – Demo-Anwendung")

    # --- Disclaimer prominent ganz oben -----------------------------------
    st.error(f"⚠️ **Hinweis:** {au.DISCLAIMER}")

    # --- Artefakt-Prüfung --------------------------------------------------
    ok, missing = au.artifacts_exist()
    if not ok:
        st.warning(
            "Es fehlen noch trainierte Artefakte. Bitte zuerst die Modellierung "
            "ausführen (Notebook 04 für das Modell, Notebook 05 für das Clustering), "
            "sodass folgende Dateien existieren:"
        )
        for m in missing:
            st.write(f"- {m}")
        st.info(

        )
        st.stop()

    model, clustering, feature_matrix = _load()

    # --- Eingabemodus wählen ----------------------------------------------
    st.sidebar.header("Eingabe")
    mode = st.sidebar.radio(
        "Wie möchten Sie eine Person definieren?",
        ["Beispielperson auswählen", "Werte manuell anpassen"],
    )

    examples = au.get_example_people(feature_matrix, n=8)

    if mode == "Beispielperson auswählen":
        idx = st.sidebar.selectbox(
            "Beispielperson", options=list(range(len(examples))),
            format_func=lambda i: f"Person #{i + 1}",
        )
        person = examples.iloc[[idx]].copy()
    else:
        # Start von einer Beispielperson, dann zentrale Werte überschreiben.
        person = examples.iloc[[0]].copy()
        st.sidebar.caption(
            "Basis ist Beispielperson #1; die folgenden Kernwerte können Sie anpassen. "
            "Übrige (technische) Merkmale bleiben unverändert."
        )
        for col, (label, lo, hi, step, tip) in au.EDITABLE_FEATURES.items():
            if col in person.columns:
                current = float(person.iloc[0][col]) if pd.notna(person.iloc[0][col]) else lo
                current = min(max(current, lo), hi)
                person.iloc[0, person.columns.get_loc(col)] = st.sidebar.number_input(
                    label, min_value=lo, max_value=hi, value=current, step=step, help=tip,
                )
        # Abgeleitete Ratios konsistent neu berechnen, falls Basiswerte geändert wurden.
        if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(person.columns):
            inc = max(float(person.iloc[0]["AMT_INCOME_TOTAL"]), 1.0)
            person.iloc[0, person.columns.get_loc("credit_income_ratio")] = \
                float(person.iloc[0]["AMT_CREDIT"]) / inc
        if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(person.columns):
            inc = max(float(person.iloc[0]["AMT_INCOME_TOTAL"]), 1.0)
            person.iloc[0, person.columns.get_loc("annuity_income_ratio")] = \
                float(person.iloc[0]["AMT_ANNUITY"]) / inc

    # --- Vorhersage --------------------------------------------------------
    probability = au.predict_default_probability(model, person)
    cls, color = au.risk_class(probability)
    cluster = au.assign_cluster(clustering, person)

    # --- Ergebnisanzeige ---------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Ausfallwahrscheinlichkeit", f"{probability * 100:.1f} %")
    col2.markdown(
        f"<p style='font-size:1.15rem; margin-bottom:0.2rem;'>Risikoklasse</p>"
        f"<p style='color:{color}; font-size:2.5rem; font-weight:700; margin:0;'>"
        f"{cls.upper()}</p>",
        unsafe_allow_html=True,
    )
    cluster_tip = au.CLUSTER_DESCRIPTIONS.get(
        cluster, f"Cluster #{cluster} – keine Beschreibung verfügbar."
    )
    col3.metric("Zugeordnetes Cluster", f"#{cluster}", help=cluster_tip)

    st.progress(min(probability, 1.0))
    st.caption(
        "Die Risikoklassen-Schwellen (niedrig < 10 %, mittel < 25 %, sonst hoch) sind "
        "**beispielhaft** und nicht geschäftlich/regulatorisch validiert."
    )

    # --- Cluster-Profil ----------------------------------------------------
    st.subheader("Profil des zugeordneten Clusters")
    profiles = clustering.get("profiles")
    if profiles is not None and "cluster" in profiles.columns:
        row = profiles[profiles["cluster"] == cluster]
        if not row.empty:
            show = [c for c in ["cluster_size", "default_rate", "AMT_INCOME_TOTAL",
                                "credit_income_ratio", "external_score_mean", "age_years"]
                    if c in row.columns]
            display_df = row[show].reset_index(drop=True).rename(columns=au.COLUMN_LABELS)
            st.dataframe(display_df, use_container_width=True)
            st.caption("Cluster sind deskriptive Gruppen, **keine kausalen** Kategorien.")

    # --- Einflussfaktoren --------------------------------------------------
    st.subheader("Wichtigste Einflussfaktoren (global)")
    with st.spinner("Berechne Einflussfaktoren ..."):
        factors = au.top_influencing_factors(model, feature_matrix, top=6)
    st.bar_chart(factors.set_index("Merkmal"))
    st.caption(
        "Globale Permutation Importance – zeigt **assoziativen**, nicht kausalen Einfluss "
        "auf die Modellvorhersage. Nicht: 'Ursache des Ausfalls'."
    )

    # --- Ausgewählte Eingabewerte transparent anzeigen --------------------
    with st.expander("Verwendete Eingabewerte (Auszug)"):
        cols = [c for c in au.EDITABLE_FEATURES if c in person.columns]
        st.dataframe(person[cols].T.rename(columns={person.index[0]: "Wert"}),
                     use_container_width=True)

    st.divider()
    st.caption(f"⚠️ {au.DISCLAIMER}")


if __name__ == "__main__":
    main()
