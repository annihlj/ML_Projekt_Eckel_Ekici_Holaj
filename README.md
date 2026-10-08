# Kreditwürdigkeitsprüfung mit Machine Learning
### Vorhersage von Kreditausfallrisiken und Segmentierung von Antragstellern

> **Portfolio-Projekt – Machine Learning**  
> Dieses Projekt dient ausschließlich zu Demonstrations- und Lernzwecken.  
> Es darf **nicht** als alleinige Grundlage für reale Kreditentscheidungen verwendet werden.

**Status:** Alle sechs Notebooks vollständig implementiert und ausgeführt &nbsp;|&nbsp; Python 3.12 &nbsp;|&nbsp; (sklearn, LightGBM, Streamlit)

---

## Schnellstart

```bash
# 1. Virtuelle Umgebung anlegen und Abhängigkeiten installieren
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 2. Notebooks der Reihe nach ausführen (01 → 06)
jupyter lab notebooks/

# 3. Demo-App starten (nach Ausführung von Notebook 04 & 05)
streamlit run app/streamlit_app.py
```

> **Rohdaten:** Vor der ersten Ausführung müssen die CSV-Dateien manuell von
> [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) heruntergeladen
> und unter `data/raw/` abgelegt werden.

---

## Inhaltsverzeichnis

| # | Abschnitt |
|---|---|
| 1 | [Hintergrund & Problemstellung](#1-hintergrund--problemstellung) |
| 2 | [Projektziel & Forschungsfragen](#2-projektziel--forschungsfragen) |
| 3 | [Datenquelle](#3-datenquelle) |
| 4 | [Projektstruktur & Navigation](#4-projektstruktur--navigation) |
| 5 | [Installation & Ausführung](#5-installation--ausführung) |
| 6 | [Methodik (CRISP-DM)](#6-methodik-crisp-dm) |
| 7 | [Modelle & Metriken](#7-modelle--metriken) |
| 8 | [Zentrale Ergebnisse](#8-zentrale-ergebnisse) |
| 9 | [Beantwortung der Forschungsfragen](#9-beantwortung-der-forschungsfragen) |
| 10 | [Kritische Diskussion & Ethik](#10-kritische-diskussion--ethik) |
| 11 | [Ausblick & Zukünftige Forschung](#11-ausblick--zukünftige-forschung) |
| 12 | [KI-Nutzung & Literatur](#12-ki-nutzung--literatur) |

---

## 1. Hintergrund & Problemstellung

### 1.1 Einleitung & Relevanz

Die Kreditwürdigkeitsprüfung ist ein zentraler Prozess im Finanzwesen: Sie entscheidet, ob Antragsteller Zugang zu Krediten erhalten. Machine-Learning-Modelle können historische Kredit-, Antrags- und Zahlungsdaten nutzen, um Ausfallrisiken datenbasiert vorherzusagen und Antragsteller in Risikogruppen zu segmentieren. Aktuelle Forschung zeigt, dass ML im Credit Scoring leistungsfähige Ansätze bietet, zugleich aber Herausforderungen wie Datenqualität, Interpretierbarkeit und Verzerrungen bestehen (Ayari et al., 2026).

Das Ziel dieses Projekts ist daher nicht nur ein präzises Vorhersagemodell, sondern auch eine nachvollziehbare und verantwortungsvolle Einordnung der Ergebnisse. KI-Systeme zur Kreditwürdigkeitsprüfung gelten nach dem EU AI Act als **Hochrisiko-Systeme** und unterliegen besonderen Anforderungen an Transparenz, Kontrolle und Risikomanagement (European Banking Authority, 2025).

> *"The evaluation of a natural person's creditworthiness and credit scoring performed by*
> *institutions have been included in the list of high-risk use cases under the rationale that such an*
> *evaluation may affect the access to key financial resources and can thus have a significantly*
> *adverse impact on natural persons."*
> — Europäische Union, 2024; European Banking Authority, 2025

Die Forschung ist relevant, weil automatisierte Kreditentscheidungen sowohl wirtschaftliche als auch gesellschaftliche Auswirkungen haben. Erklärbare und faire Modelle helfen, Diskriminierung, Fehlentscheidungen und regulatorische Risiken zu reduzieren (FinRegLab, 2023).

---

### 1.2 Praxisbeispiel: Bias und Diskriminierung – Svea Ekonomi AB (Finnland, 2018)

Dieses Fallbeispiel veranschaulicht, welche realen Konsequenzen fehlerhaftes ML-Credit-Scoring haben kann.

**Ausgangssituation**

Einem Antragsteller wurde bei einem Online-Kauf die Kreditverlängerung verweigert, obwohl er keinerlei Zahlungsausfälle in seiner Historie hatte. Die Ablehnung erfolgte durch ein vollautomatisiertes, statistisches Scoring-System.

**Das Problem im Modell**

Das Modell wies einen systematischen Bias auf, da es ungeeignete Daten als Entscheidungsgrundlage nutzte:

- **Verwendung geschützter Merkmale:** Das Modell nutzte rechtlich geschützte Attribute wie Geschlecht, Muttersprache, Alter und Wohnort zur Risikobewertung.
- **Statistische Diskriminierung (Profiling):** Das System erkannte in den Trainingsdaten, dass Männer und finnischsprachige Personen im Durchschnitt häufiger Rückzahlungsprobleme hatten. Diese abstrakten Gruppenwahrscheinlichkeiten wurden dem Kläger negativ angerechnet – obwohl seine individuelle Bonität einwandfrei war.
- **Fehlende Kausalität / Proxy-Variablen:** Statt tatsächlicher Indikatoren für Zahlungsfähigkeit (Einkommen, Kredithistorie) verließ sich das Modell auf Ersatzvariablen. Der Antragsteller wurde nicht als Individuum, sondern als „Repräsentant eines statistischen Profils" bewertet.

**Urteil**

Das Tribunal verbot das System und verhängte eine Strafe von 100.000 €. Begründung: Effizienzargumente rechtfertigen keine Mehrfachdiskriminierung. Statistische Annahmen über Dritte dürfen nicht die einzige Basis einer Kreditablehnung sein.

**Takeaways für ML-Projekte**

| Lektion | Umsetzung in diesem Projekt |
|---|---|
| Geschützte Merkmale ausschließen | `config.SENSITIVE_OR_PROXY_FEATURES` markiert kritische Features |
| Fairness ≠ statistische Genauigkeit | Ethik-Kapitel in Notebook 06 |
| Human-in-the-Loop sicherstellen | Modell gibt Wahrscheinlichkeit – keine finale Entscheidung |

---

### 1.3 Was ist Credit Scoring?

Unter **Kreditrisiko** versteht man das Risiko, dass Kreditnehmer ihren Zahlungsverpflichtungen nicht nachkommen. Credit Scoring versucht, die Ausfallwahrscheinlichkeit eines Antragstellers zuverlässig einzuschätzen. Klassische Verfahren nutzen statistische Modelle; ML-Ansätze erkennen zusätzlich komplexe, nichtlineare Zusammenhänge in großen Datenmengen – mit zunehmender Bedeutung von Ensemble Learning, Klassifikation und Deep Learning (Ayari et al., 2026).

---

## 2. Projektziel & Forschungsfragen

### 2.1 Projektziel

Untersucht wird, ob sich das Kreditausfallrisiko anhand von Antragsdaten, früheren Kreditinformationen, Zahlungsdaten und weiteren Kundendaten vorhersagen lässt – und ob sich Antragsteller sinnvoll in Risikogruppen (Cluster) einteilen lassen.

### 2.2 Zentrale Forschungsfrage

> *Inwiefern können Machine-Learning-Modelle das Kreditausfallrisiko von Antragstellern*
> *anhand historischer Kredit- und Antragsdaten vorhersagen, und welche Gruppen von*
> *Antragstellern lassen sich durch Clustering identifizieren?*

### 2.3 Teilfragen

| # | Teilfrage | Methode |
|---|---|---|
| TF1 | Welche Merkmale haben den größten Einfluss auf das Ausfallrisiko? | Permutation Importance |
| TF2 | Wie stark verbessert Feature Engineering die Modellleistung? | Vorher-/Nachher-Vergleich |
| TF3 | Welche Modelle eignen sich für tabellarische Kreditrisikodaten? | Modellvergleich, Cross-Validation |
| TF4 | Wie lassen sich Antragsteller sinnvoll segmentieren? | K-Means Clustering |
| TF5 | Welche ethischen und methodischen Grenzen bestehen? | Kritische Analyse, Ethik-Review |

**Aufgabentypen:**
- (a) **Binäre Klassifikation** (überwacht) → Vorhersage von `TARGET`
- (b) **Clustering** (unüberwacht) → Segmentierung von Antragstellern

---

## 3. Datenquelle

**Kaggle – „Home Credit Default Risk"**
[(https://www.kaggle.com/c/home-credit-default-risk)](https://www.kaggle.com/c/home-credit-default-risk)

Relationaler Datensatz mit Haupt- und Nebentabellen. Verbindungsschlüssel: `SK_ID_CURR`.  
Zielvariable `TARGET`: `1` = Zahlungsschwierigkeiten / erhöhtes Ausfallrisiko, `0` = sonst.

Eine detaillierte Beschreibung aller Tabellen und Merkmale findet sich in [Data_explanation.md](Data_explanation.md).

### Verwendete Tabellen

| Priorität | Tabelle | Inhalt |
|---|---|---|
| Pflicht | `application_train.csv` & `application_test.csv` | Zielebene, demografische & finanzielle Kernmerkmale |
| Pflicht | `bureau.csv` & `bureau_balance.csv` | Externe Kredithistorie (andere Institute) |
| Pflicht | `previous_application.csv` | Interne Antragshistorie (Home Credit) |
| Pflicht | `installments_payments.csv` | Konkretes Zahlungsverhalten (Verzug/Unterzahlung) |
| Optional | `POS_CASH_balance.csv`, `credit_card_balance.csv` | Monatliche Verlaufsdetails |

---

## 4. Projektstruktur & Navigation

### 4.1 Verzeichnisbaum

```
credit-risk-ml/
│
├── data/
│   ├── raw/                    # Originaldaten von Kaggle (nicht im Repo – zu groß)
│   ├── interim/                # Aufbereitete Zwischentabellen nach Aggregation
│   └── processed/              # Fertige Feature-Matrix (feature_matrix.parquet)
│
├── notebooks/                  # Ausführbare Analyseschritte – in dieser Reihenfolge ausführen
│   ├── 01_data_understanding.ipynb          # Tabellenstruktur, Schlüssel, Datenqualität
│   ├── 02_data_engineering.ipynb            # Aggregation der Nebentabellen, Joins
│   ├── 03_eda.ipynb                         # Verteilungen, Korrelationen, Ausfall-Vergleich
│   ├── 04_modeling_classification.ipynb     # Klassifikationsmodelle & Evaluation
│   ├── 05_clustering.ipynb                  # K-Means, Silhouette, Cluster-Profile
│   └── 06_model_interpretation_and_discussion.ipynb  # Importance, PDPs, Ethik, Limitationen
│
├── src/                        # Wiederverwendbare Python-Module
│   ├── config.py               # Pfade, Konstanten, markierte sensible Merkmale
│   ├── load_data.py            # Laden der Rohtabellen aus data/raw/
│   ├── data_cleaning.py        # Bereinigung, Typenkorrektur, Ausreißerbehandlung
│   ├── feature_engineering.py  # Aggregation aus Nebentabellen, neue Kennzahlen
│   ├── preprocessing.py        # sklearn-Pipeline: Imputation, Encoding, Skalierung
│   ├── train_models.py         # Modelltraining, Cross-Validation, Hyperparameter-Suche
│   ├── evaluate_models.py      # Metriken, ROC/PR-Kurven, Konfusionsmatrix, Kalibrierung
│   ├── clustering.py           # K-Means, Silhouette-Analyse, Cluster-Profiling
│   ├── interpretation.py       # Permutation Importance, Partial Dependence Plots
│   ├── eda.py                  # Hilfsfunktionen für explorative Analysen
│   └── utils.py                # Logging, Speichern von Abbildungen & Tabellen
│
├── models/
│   ├── best_model.joblib       # Gespeicherter HistGradientBoostingClassifier (inkl. Pipeline)
│   └── clustering.joblib       # Gespeichertes K-Means-Modell
│
├── reports/
│   ├── figures/                # Erzeugte Abbildungen (PNG)
│   │   ├── eda/                # Verteilungsplots, Korrelationsmatrizen
│   │   ├── modeling/           # ROC-, PR-Kurven, Kalibrierungskurven
│   │   ├── clustering/         # Elbow-Kurve, Silhouette, Cluster-Profileplots
│   │   └── interpretation/     # Feature-Importance-Diagramme, PDPs
│   └── tables/                 # Erzeugte Auswertungstabellen (CSV)
│       ├── eda/, modeling/,  clustering/ ; interpretation/
│
├── app/
│   ├── streamlit_app.py        # Interaktive Demo-Webanwendung
│   └── app_utils.py            # Hilfsfunktionen für Laden & Vorhersage in der App
│
├── requirements.txt            # Python-Abhängigkeiten
├── Data_explanation.md         # Erklärung aller Kaggle-Tabellen und Merkmale
└── README.md
```

### 4.2 Schnellnavigation

| Ich möchte … | Einstiegspunkt |
|---|---|
| Verstehen, wie die Daten aussehen | [01_data_understanding.ipynb](notebooks/01_data_understanding.ipynb) |
| Nachvollziehen, wie Features erzeugt wurden | [02_data_engineering.ipynb](notebooks/02_data_engineering.ipynb) + [src/feature_engineering.py](src/feature_engineering.py) |
| Modelle und Kennzahlen vergleichen | [04_modeling_classification.ipynb](notebooks/04_modeling_classification.ipynb) |
| Verstehen, welche Merkmale wichtig sind | [06_model_interpretation_and_discussion.ipynb](notebooks/06_model_interpretation_and_discussion.ipynb) |
| Die Demo-App starten | [app/streamlit_app.py](app/streamlit_app.py) → siehe Abschnitt 5 |
| Pfade & Konstanten anpassen | [src/config.py](src/config.py) |
| Ethik und Grenzen des Modells lesen | Abschnitt 10 & Notebook 06 (Abschnitte 5–6) |

---

## 5. Installation & Ausführung

### 5.1 Installation

```bash
# 1) Repository klonen
cd credit-risk-ml                              # Optional, wenn über GitHub

# 2) Virtuelle Umgebung anlegen (empfohlen)
python -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate

# 3) Abhängigkeiten installieren  (Python 3.12 empfohlen)
pip install -r requirements.txt

# 4) Jupyter-Kernel registrieren (optional)
python -m ipykernel install --user --name credit-risk-ml
```

### 5.2 Ausführung

```bash
# A) Notebooks der Reihe nach ausführen (01 → 06):
jupyter lab notebooks/

# B) Demo-App starten (setzt trainiertes Modell unter models/ voraus):
streamlit run app/streamlit_app.py
```

Die App öffnet sich automatisch im Browser unter `http://localhost:8501`.  
Falls das Modell fehlt, zeigt die App eine klare Anleitung an, statt abzustürzen.

---

## 6. Methodik (CRISP-DM)

Das Projekt folgt dem **CRISP-DM**-Prozess *(Cross-Industry Standard Process for Data Mining; Wirth & Hipp, 2000)*:

```
Business Understanding → Data Understanding → Data Preparation
        → Modeling → Evaluation → Deployment
```

Die sechs Notebooks decken diesen Prozess schrittweise ab:

| Notebook | Phase | Inhalt |
|---|---|---|
| `01` Data Understanding | Datenverstehen | Struktur, Granularität, Schlüssel, Datenqualität |
| `02` Data Engineering | Datenvorbereitung | Aggregation der Nebentabellen, Joins auf `SK_ID_CURR` |
| `03` EDA | Datenverstehen | Verteilungen, Korrelationen, Ausfall- vs. Nicht-Ausfall-Vergleich |
| `04` Klassifikation | Modellierung + Evaluation | Baseline → Logistic Regression → Random Forest → Gradient Boosting |
| `05` Clustering | Modellierung | K-Means, Silhouette-Analyse, PCA-Visualisierung |
| `06` Interpretation & Diskussion | Evaluation + Deployment | Permutation Importance, PDPs, Ethik, Limitationen |

---

## 7. Modelle & Metriken

### 7.1 Modellvergleich

Fünf Modelle wurden mit **5-facher stratifizierter Cross-Validation** und einmalig auf dem **Testset** bewertet  
(307.511 Anträge, stratifizierter 80/20-Split, Ausfallrate konstant 8,1 % in beiden Splits):

| Modell | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|:---:|:---:|:---:|
| `DummyClassifier` (Baseline) | 0.081 | 0.081 | 0.500 |
| `LogisticRegression` | 0.219 | 0.246 | 0.763 |
| `RandomForestClassifier` | 0.198 | 0.217 | 0.740 |
| **`HistGradientBoostingClassifier`** | **0.226** | **0.269** | **0.775** |
| `LightGBMClassifier` | 0.229 | 0.268 | 0.775 |

> **Bestes Modell:** `HistGradientBoostingClassifier` (PR-AUC 0.269).  
> Nach Hyperparameter-Tuning (learning_rate=0.03, max_iter=200, max_leaf_nodes=31):  
> CV PR-AUC = 0.226, Test PR-AUC = 0.245.

### 7.2 Bewertungsmetriken

Wegen der stark **unausgewogenen** Zielvariable (11,4 : 1) sind Accuracy-basierte Metriken wenig aussagekräftig:

> Ein trivialer Klassifikator, der nie einen Ausfall vorhersagt, erreicht ~92 % Accuracy –
> ohne einen einzigen Ausfall zu erkennen.

Daher werden vorrangig folgende Metriken verwendet:

| Metrik | Warum relevant |
|---|---|
| **PR-AUC / Average Precision** | Hauptmetrik – robust bei Klassenungleichgewicht |
| **ROC-AUC** | Ergänzend – misst generelle Trennschärfe |
| **Recall & F1** | Wie viele Ausfälle werden erkannt? |
| Confusion Matrix | Visualisierung von False Positives/Negatives |
| Schwellenwert-Analyse | Precision/Recall-Trade-off bei verschiedenen Schwellen |
| Kalibrierungskurve | Ist die vorhergesagte Wahrscheinlichkeit realistisch? |

---

## 8. Zentrale Ergebnisse

**Datenbasis:** 307.511 Anträge; 161 Features nach Feature Engineering ; Ausfallrate 8,1 % (Ungleichgewicht 11,4 : 1)

### Klassifikation

- `HistGradientBoostingClassifier` und `LightGBM` sind die stärksten Modelle (PR-AUC ≈ 0.27, ROC-AUC ≈ 0.775).
- **Der Schwellenwert ist entscheidend:**

| Schwelle | Recall | Precision | F1 |
|:---:|:---:|:---:|:---:|
| 0.50 (Default) | 2,6 % | 61,6 % | — |
| **0.15** | **44,7 %** | **25,7 %** | **0.326** |

> Welcher Fehler schwerer wiegt – abgelehnte kreditwürdige Personen vs. bewilligte riskante Kredite –
> ist eine **fachliche und ethische Entscheidung**, keine technische Konstante.

### Feature Importance (Permutation Importance, Random Forest)

Die wichtigsten Einflussfaktoren:

1. `external_score_mean` ; `external_score_min` ; `external_score_max` ← stärkste Prädiktoren
2. `EXT_SOURCE_2` ; `EXT_SOURCE_3`
3. `annuity_credit_ratio` ; `NAME_EDUCATION_TYPE` ; `age_years`
4. `INST_LATE_PAYMENT_RATIO`

> Kein als sensibel markiertes Merkmal (Geschlecht, direkte Proxy-Variablen) erscheint unter den Top 5.

### Clustering (K-Means, k = 4)

Das Silhouette-Optimum lag bei k = 2 (Koeffizient 0.146); k = 4 wurde aus fachlichen Gründen gewählt.

| Cluster | Größe | Ausfallrate | Profil |
|:---:|:---:|:---:|---|
| **0** | 80.761 | **6,2 %** | Beste externe Bonität, niedrigster Verschuldungsgrad |
| 1 | 69.808 | 8,9 % | Mittleres Risiko |
| 2 | 156.941 | 8,6 % | Größter & jüngster Cluster |
| 3 | 1 | — | Statistischer Ausreißer |

> Niedrige Silhouette-Werte sind bei realen, graduell übergehenden Daten üblich.

---

## 9. Beantwortung der Forschungsfragen

### Gesamtantwort

Machine-Learning-Modelle können das Kreditausfallrisiko im Home-Credit-Datensatz **erkennbar besser** vorhersagen als eine naive Baseline. Gradient-Boosting-Modelle liefern die stärkste Leistung. Die Modellgüte hängt stark von Feature Engineering, geeigneten Metriken und einer bewussten Schwellenwertwahl ab. Das Clustering ergänzt die Klassifikation durch Risikoprofile – die Gruppen sind jedoch nicht vollständig trennscharf und sollten nicht als starre Entscheidungsklassen verwendet werden.

> **Fazit:** ML eignet sich gut zur datenbasierten *Unterstützung* der Kreditwürdigkeitsprüfung –
> nicht aber als alleinige Entscheidungsinstanz.

---

### TF1 ; Welche Merkmale haben den größten Einfluss?

Die Top-Einflussfaktoren sind die aggregierten externen Bonitätsscores (`external_score_mean/min/max`) sowie `EXT_SOURCE_2/3`. Diese bilden externe Risikoeinschätzungen ab und besitzen die höchste Erklärungskraft. Weitere wichtige Merkmale: `annuity_credit_ratio`, `NAME_EDUCATION_TYPE`, `age_years`, `INST_LATE_PAYMENT_RATIO`. Kein direkt als sensibel markiertes Merkmal erscheint unter den Top 5 – indirekte Proxy-Verzerrungen bleiben aber möglich.

> **Antwort:** Das Ausfallrisiko wird vor allem durch externe Bonitätsscores, Kredit-/Einkommensrelationen, Alter, Bildung und historisches Zahlungsverhalten beeinflusst.

---

### TF2 ; Wie stark verbessert Feature Engineering die Modellleistung?

Aus den Nebentabellen wurden aggregierte Merkmale erzeugt (Kredithistorie, Zahlungsverhalten, Verzüge, Verhältniskennzahlen). Die finale Basis umfasst 161 Features. Besonders wirksam: externe Score-Aggregationen, `annuity_credit_ratio`, `missing_values_count` und zahlungsbezogene Aggregationen.

> **Antwort:** Feature Engineering verbessert die Modellleistung wesentlich – ohne erweiterte Merkmale gingen wichtige Hinweise aus Kredithistorie und Zahlungsverhalten verloren.

---

### TF3 ; Welche Modelle eignen sich am besten?

Baumbasierte Ensemble-Modelle dominieren: `HistGradientBoostingClassifier` und `LightGBM` erzielen Test-PR-AUC ≈ 0.27 / ROC-AUC ≈ 0.775. `LogisticRegression` ist solide und bietet höhere Interpretierbarkeit. `RandomForestClassifier` liefert brauchbare, aber schwächere Ergebnisse.

> **Antwort:** Gradient-Boosting-Verfahren sind am geeignetsten. Für erklärungsorientierte Zwecke bietet die logistische Regression eine transparente Vergleichsbasis.

---

### TF4 ; Wie lassen sich Antragsteller segmentieren?

K-Means mit k = 4 liefert drei interpretierbare Hauptcluster + einen Ausreißer. Cluster 0 hat die beste Bonität und niedrigste Ausfallrate (6,2 %); Cluster 2 ist der größte und jüngste Cluster. Niedrige Silhouette-Werte zeigen, dass Antragsteller graduell ineinander übergehen.

> **Antwort:** Antragsteller lassen sich in Gruppen mit unterschiedlicher Bonität, Verschuldung und Ausfallrate segmentieren – geeignet für explorative Analyse, nicht für starre Entscheidungsklassen.

---

### TF5 ; Welche ethischen und methodischen Grenzen bestehen?

Methodisch problematisch sind: fehlende Werte, Klassenungleichgewicht, begrenztes Tuning, Aggregationsverluste und fehlender zeitlicher Split. Ethisch kritisch: Modelle lernen aus historisch verzerrten Entscheidungen und können Proxy-Variablen zur indirekten Diskriminierung nutzen.

> **Antwort:** Automatisierte Kreditwürdigkeitsprüfung ist keine objektive Entscheidung. Das Modell liefert eine Risikoeinschätzung – die finale Entscheidung muss transparent, überprüfbar, regulatorisch konform und durch menschliche Kontrolle abgesichert sein.

---

## 10. Kritische Diskussion & Ethik

*Ausführliche Behandlung in Notebook 06, Abschnitte 5–6.*

| Thema | Kernaussage |
|---|---|
| **Datenqualität** | 67/122 Spalten enthalten fehlende Werte; Fehlmechanismus (MCAR/MAR/MNAR) unbekannt. `missing_values_count` als zusätzliches Feature integriert. |
| **Historischer Bias** | Das Modell lernt aus vergangenen Kreditentscheidungen. Waren diese diskriminierend, reproduziert es die Diskriminierung. |
| **Proxy-Variablen** | Wohnregion, Familienstand, Berufsgruppe können indirekt sensible Attribute abbilden. Kritische Features sind in `config.SENSITIVE_OR_PROXY_FEATURES` markiert. |
| **Vorhersage ≠ Entscheidung** | Das Modell schätzt eine Wahrscheinlichkeit. Schwellenwahl, Härtefälle und menschliche Prüfung gehören zwingend dazu. |
| **Regulierung** | DSGVO Art. 22 (Recht auf Erklärung bei automatisierten Entscheidungen); EU AI Act → Kreditwürdigkeitssysteme sind Hochrisiko-Anwendungen. |
| **Limitationen** | Kein zeitlicher Split (Look-ahead Bias); begrenztes Hyperparameter-Tuning; zeitliche Dynamik durch Aggregation teilweise verloren. |

---

## 11. Ausblick & Zukünftige Forschung

Die vorliegende Arbeit zeigt vielversprechende Ansätze, lässt aber methodische und ethische Lücken offen. Folgende Richtungen könnten das Projekt sinnvoll erweitern:

### 11.1 Modellierung & Validierung

| Thema | Beschreibung |
|---|---|
| **Zeitlicher Split** *(höchste Priorität)* | Training auf älteren, Test auf neueren Anträgen – verhindert Look-ahead Bias |
| **Hyperparameter-Optimierung** | Bayesian Search (z. B. Optuna) über größeren Suchraum |
| **Klassenungleichgewicht** | SMOTE, Class-Weight-Anpassung oder Cost-Sensitive Learning |
| **Sequenzielle Modelle** | LSTM / Transformer für die zeitliche Dynamik der Nebentabellen |
| **Reject Inference** | Methoden zur Einbeziehung abgelehnter Anträge (Augmentation, Parcelling) |

### 11.2 Erklärbarkeit & Interpretierbarkeit

| Thema | Beschreibung |
|---|---|
| **SHAP** | Lokale Erklärungen für individuelle Entscheidungen (EU AI Act Anforderung) |
| **Counterfactual Explanations** | „Was müsste sich ändern, damit der Kredit genehmigt wird?" |
| **Kalibrierung** | Platt-Scaling / Isotone Regression für realistischere Wahrscheinlichkeiten |

### 11.3 Fairness & Bias

| Thema | Beschreibung |
|---|---|
| **Gruppenbasiertes Fairness-Audit** | Auswertung nach Alter, Geschlecht, Region (Fairlearn, AI Fairness 360) |
| **Debiasing-Strategien** | Pre-/In-/Post-Processing-Ansätze zur Fairness-Korrektur |
| **Proxy-Analyse** | Systematische Korrelationsanalyse zwischen Features und sensiblen Attributen |

### 11.4 Datenqualität & Quellen

| Thema | Beschreibung |
|---|---|
| **Robustere Imputation** | Multiple Imputation (MICE) statt Median/Modus-Annahme |
| **Externe Datenquellen** | Makroökonomische Indikatoren, Echtzeit-Auskunfteiscores |
| **Datensattalter** | Datensatz aus ~2018 – COVID-19, Zinswende, Inflation nicht abgebildet |

### 11.5 Produktiver Einsatz & Monitoring

| Thema | Beschreibung |
|---|---|
| **Modell-Monitoring** | Concept Drift / Data Drift Erkennung (Evidently AI, NannyML) |
| **Schwellenwertoptimierung** | Kosten-Nutzen-basierte Schwellenbestimmung statt Default 0.5 |
| **A/B-Testing** | Kontrollierte Experimente zur Messung des tatsächlichen Nutzens |

---

## 12. KI-Nutzung & Literatur

### Erklärung zur KI-Nutzung

Teile dieses Projekts (Code-Gerüst, Dokumentation, methodische Erläuterungen) wurden mit Unterstützung eines KI-Assistenzsystems erstellt und anschließend eigenständig geprüft, angepasst und inhaltlich verantwortet. Alle Modellergebnisse wurden selbstständig nachvollzogen.

Das Projekt wurde von allen Gruppenmitgliedern zusammen erarbeitet.
---

### Literaturverzeichnis

Ayari, H., Guetari, P. R., & Kraïem, P. N. (2026). *Machine learning powered financial credit scoring: A systematic literature review*. *Artificial Intelligence Review, 59*, Article 13. https://doi.org/10.1007/s10462-025-11416-2

Europäische Union. (2024). *Verordnung (EU) 2024/1689 – Gesetz über künstliche Intelligenz (EU AI Act)*. Amtsblatt der Europäischen Union. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

European Banking Authority. (2023). *Follow-up report on the use of machine learning for internal ratings-based models*. https://www.eba.europa.eu/publications-and-media/press-releases/eba-publishes-follow-report-use-machine-learning-internal

European Banking Authority. (2025). *AI Act: Implications for the EU banking and payments sector*. https://www.eba.europa.eu/

FinRegLab. (2023). *Explainability & fairness in machine learning for credit underwriting: Policy analysis*. https://finreglab.org/research/explainability-fairness-in-machine-learning-for-credit-underwriting-policy-analysis/

Kaggle. (2018). *Home Credit Default Risk* [Dataset]. https://www.kaggle.com/c/home-credit-default-risk

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions (SHAP). *NeurIPS.*

Orwat, Carsten (2020): Diskriminierungsrisiken durch Verwendung von Algorithmen, 1. Aufl., Baden-Baden: Nomos (S. 50-52)

Wirth, R., & Hipp, J. (2000). *CRISP-DM: Towards a standard process model for data mining.*