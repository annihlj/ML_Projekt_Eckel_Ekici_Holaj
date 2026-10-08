# Datenbasis: Home Credit Default Risk

Für die Beantwortung unserer Forschungsfragen nutzen wir den Datensatz der Kaggle-Challenge **"Home Credit Default Risk"**. Dieser Datensatz wurde von der *Home Credit Group* bereitgestellt, einem internationalen Finanzinstitut, das sich auf die Kreditvergabe an Menschen mit wenig oder gar keiner traditionellen Kredithistorie (sogenannte "Unbanked"-Population) spezialisiert hat.
Quelle: https://www.kaggle.com/competitions/home-credit-default-risk/overview

## 1. Was sind das für Daten? (Datenbeschreibung)

Der Datensatz besteht aus einer **relationalen Datenbankstruktur** mit mehreren miteinander verknüpften Tabellen. 
Er soll die Komplexität von Bankdaten zeigen. 

### Haupttabelle (`application_train.csv` / `application_test.csv`)
Dies ist der Kern des Datensatzes. Jede Zeile repräsentiert einen Kreditantrag.
* **Zielvariable (`TARGET`):** Zeigt an, ob der Kreditnehmer Zahlungsschwierigkeiten hatte (1 = Zahlungsausfall / Zahlungsverzug, 0 = Kredit pünktlich zurückgezahlt).
* **Demografische & Sozioökonomische Merkmale:** Alter, Geschlecht, Familienstand, Anzahl der Kinder, Wohnsituation, Bildungsabschluss.
* **Finanzielle Merkmale:** Jahresberufseinkommen, Kreditsumme, jährliche Kreditrate, Preis des finanzierten Gutes.

### Nebentabellen
Um die Kredithistorie der Antragsteller zu bewerten, liefert der Datensatz umfangreiche Vergangenheitsdaten (die über eine eindeutige ID `SK_ID_CURR` mit der Haupttabelle verknüpft werden können):
* **`bureau.csv` & `bureau_balance.csv`:** Daten aus externen Auskünften (Ähnlich wie eine Schufa). Zeigt Kredite an, die der Kunde bei *anderen* Finanzinstituten hat/hatte.
* **`previous_application.csv`:** Alle früheren Kreditanträge des Kunden bei der *Home Credit Group* selbst (genehmigte & abgelehnte).
* **`installments_payments.csv`:** Die genaue Historie der Ratenzahlungen (Wurde pünktlich gezahlt? Wurde der volle Betrag gezahlt?).
* **`POS_CASH_balance.csv` & `credit_card_balance.csv`:** Monatliche Salden aus früheren Konsumentenkrediten und Kreditkarten.
* **`HomeCredit_columns_description.csv`:** Das ist das sogenannte "Data Dictionary" (Daten-Wörterbuch) oder Codebook für das gesamte Projekt. In dieser Datei ist tabellarisch aufgeschlüsselt, aus welcher Tabelle eine Spalte stammt und was genau der Wert bedeutet.
---

## 2. Warum haben wir diesen Datensatz für unser Projekt ausgewählt?

Wir haben den Datensatz ausgewählt weil er optimal zu unsere zentralen Forschungsfragen passt und eine aktuelle realistische Challenge darstellt:

1. **Beantwortung der Hauptforschungsfrage (Kreditrisiko-Prädiktion):**
   Der Datensatz enthält eine klar definierte Zielvariable (`TARGET`), die es uns ermöglicht, überwachte Lernverfahren (Supervised Learning) zur binären Klassifikation des Ausfallrisikos zu trainieren.

2. **Potenzial für Feature Engineering (Teilfrage 2):**
   Da die Daten über sieben Tabellen verteilt sind, können wir das Modell nicht einfach "out-of-the-box" trainieren. Wir müssen aggregierte Merkmale bilden (z.B. *Durchschnittliche Verspätung bei Ratenzahlungen*, *Anteil abgelehnter Voranträge*). Dies erlaubt uns zu messen, wie stark Feature Engineering die Modellleistung verbessert.

3. **Umgang mit unbalancierten Daten (Teilfrage 3):**
   In der Realität fallen Kredite selten aus. Auch in diesem Datensatz liegt die Ausfallquote bei nur ca.  8 %. Diese starke Klassen-Imbalance (Imbalanced Data) erfordert fortschrittliche Machine-Learning-Algorithmen (wie XGBoost oder LightGBM) und spezielle Evaluierungsmetriken (wie den ROC-AUC-Score), was den methodischen Anspruch des Projekts hebt.
   Quellen: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html & https://www.geeksforgeeks.org/machine-learning/lightgbm-vs-xgboost-which-algorithm-is-better/

4. **Reichhaltigkeit für Kundensegmentierung (Teilfrage 4):**
   Mit über 100 Features in der Haupttabelle (von Bildungsgrad über Einkommensart bis hin zum Alter) bietet der Datensatz eine umfassende Grundlage, um unüberwachtes Lernen (Unsupervised Learning / Clustering) anzuwenden und verborgene Antragsteller-Typen zu identifizieren.

5. **Ethische & Soziale Relevanz (Teilfrage 5):**
   Die Home Credit Group richtet sich explizit an Menschen ohne Bankhistorie. Das Modell muss Entscheidungen auf Basis von alternativen Merkmalen treffen (z. B. Telefonnutzung, Wohngegend, familiäres Umfeld). Das bietet uns eine gute Diskussionsgrundlage, um Fairness, Bias (Voreingenommenheit von KI) und ethische Grenzen bei der Kreditvergabe kritisch zu beleuchten.