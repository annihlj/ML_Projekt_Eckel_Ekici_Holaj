"""Minimales Retrain-Skript: erzeugt best_model.joblib und clustering.joblib neu."""
import sys
import os

# Sicherstellen, dass src/ im Suchpfad liegt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(__import__("multiprocessing").cpu_count()))

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

import config
import preprocessing as prep
import utils
from train_models import ModelSpec, build_full_pipeline, save_model
from clustering import (
    run_clustering,
    select_cluster_features,
)

log = utils.get_logger()

# --- Klassifikationsmodell ---
log.info("Lade Feature-Matrix ...")
feat = utils.load_processed("feature_matrix")
X_train, X_test, y_train, y_test = prep.make_train_test_split(feat)

log.info("Trainiere HistGradientBoostingClassifier (%d Zeilen) ...", len(X_train))
numeric, categorical = prep.identify_feature_types(X_train)
spec = ModelSpec(
    "hist_gbdt",
    HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=config.RANDOM_STATE,
    ),
    needs_scaling=False,
)
pipe = build_full_pipeline(spec, numeric, categorical)
pipe.fit(X_train, y_train)
save_model(pipe, "best_model")
log.info("best_model.joblib gespeichert.")

# --- Clustering ---
log.info("Führe Clustering aus ...")
labels, profiles, km, pre = run_clustering(feat)
X_clust = select_cluster_features(feat)
clustering_bundle = {
    "kmeans": km,
    "preprocessor": pre,
    "features": list(X_clust.columns),
}
joblib.dump(clustering_bundle, config.MODELS_DIR / "clustering.joblib")
log.info("clustering.joblib gespeichert.")

log.info("Fertig! Starte die App mit: streamlit run app/streamlit_app.py")
