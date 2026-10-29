"""
Esperimento DI TEST su dati sintetici generati con sklearn.datasets.make_classification
-- VERSIONE ESTESA con scenari piu' sfidanti rispetto a synthetic_experiments.py.

QUESTO FILE E' COMPLETAMENTE AUTONOMO: non importa nulla dagli altri due script
(removal_strategies_experiments.py, synthetic_experiments.py). Tutte le removal
strategy, le metriche di valutazione e i wrapper per SHAP/KernelSHAP/LIME sono
ridefiniti qui sotto, identici a quelli degli altri script. I tre file possono
essere modificati indipendentemente l'uno dall'altro.

DIFFERENZA RISPETTO A synthetic_experiments.py: quello base copre solo due assi
(numero di feature x linearita' del confine di decisione). Qui aggiungiamo altre
dimensioni di difficolta' che negli esperimenti reali si presentano spesso e che
possono mettere in crisi gli explainer o le removal strategy in modi diversi:

    - segnale sparso in molto rumore (poche feature informative su molte)
    - feature ridondanti/correlate (classico caso difficile per le attribuzioni:
      l'importanza si "sparge" tra feature collineari)
    - sbilanciamento di classe (mite e severo)
    - rumore sulle label (flip_y alto)
    - separabilita' debole (classi quasi sovrapposte)
    - confine di decisione molto frastagliato (molti cluster per classe)
    - scale delle feature molto eterogenee (una feature in decine di migliaia,
      un'altra in centesimi -- mette alla prova strategie come "uniform" che
      dipendono dal range della feature)
    - geometria dei cluster irregolare (hypercube=False)
    - regime "quasi tutto informativo" (poche feature di puro rumore)

In ogni scenario, make_classification (con shuffle=False) mette le prime
n_informative + n_redundant + n_repeated colonne come feature che portano
segnale reale (le ultime sono rumore puro): e' la ground truth con cui
confrontiamo le attribuzioni generate dagli explainer.

Essendo un esperimento di test, i parametri restano piu' leggeri di un uso in
produzione, e il set complessivo comprende i 15 scenari riportati nell'appendice della tesi.
"""

import itertools
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import shap
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm

RNG = np.random.default_rng(0)

# Compatibilita' numpy 1.x / 2.x: trapz e' stato rinominato in trapezoid in numpy 2.0
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------------------
# 0. Configurazione centralizzata. Esperimento di test -> valori piu' leggeri
#    rispetto a un uso in produzione, ma stessa struttura per-metrica degli
#    altri due script (n_instances/n_runs configurabili separatamente).
# ---------------------------------------------------------------------------

CONFIG = {
    "n_samples": 5000,          # righe generate da make_classification per scenario
    "random_state": 0,           # seed per la generazione del dataset sintetico

    "faithfulness": {
        "n_instances": 1000,
        "n_runs": 1,
        "n_subsets": 20,
        "random_state": 0,
    },

    "deletion_insertion": {
        "n_instances": 1000,
        "n_runs": 1,
        "ranking_mode": "absolute",
        "compute_random_baseline": True,
        "n_random_rankings": 5,
        "random_state": 0,
        "accuracy_threshold": 0.5,
    },

    "complexity": {
        "n_instances": 1000,
        "n_runs": 1,
        "log_base": "e",
        "eps": 1e-12,
    },

    "ground_truth": {
        "n_instances": 1000,
        "top_k_for_precision": None,  # None -> usa n_informative+n_redundant+n_repeated dello scenario
    },

    "output_dir": "results",
    "curves_csv": "synthetic_extended_curves.csv",
    "metrics_csv": "synthetic_extended_instance_metrics.csv",
    "faithfulness_csv": "synthetic_extended_faithfulness_runs.csv",
    "complexity_csv": "synthetic_extended_complexity_runs.csv",
    "ground_truth_csv": "synthetic_extended_ground_truth_agreement.csv",
}

# ---------------------------------------------------------------------------
# 1. Scenari sintetici estesi. Ogni scenario e' un dict di parametri passati
#    (quasi) direttamente a make_classification, piu' un paio di chiavi extra
#    gestite da make_synthetic_dataset (feature_scale_multipliers).
#    Gli scenari coprono diverse dimensionalita'. La strategia separate_models
#    cresce esponenzialmente con n_features e puo' quindi essere molto costosa
#    negli scenari a dimensionalita' maggiore.
# ---------------------------------------------------------------------------

SCENARIOS = [
    # --- Scenari base: stesso generatore, diverso numero di istanze spiegate ---
    {
        "name": "linear_low_dim",
        "n_features": 6, "n_informative": 3, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 1, "class_sep": 2.0, "weights": None, "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 1000,
    },
    {
        "name": "baseline_linear",
        "n_features": 6, "n_informative": 3, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 1, "class_sep": 2.0, "weights": None, "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 500,
    },
    {
        "name": "nonlinear_low_dim",
        "n_features": 6, "n_informative": 3, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 3, "class_sep": 1.0, "weights": None, "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 1000,
    },
    {
        "name": "baseline_nonlinear",
        "n_features": 6, "n_informative": 3, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 3, "class_sep": 1.0, "weights": None, "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 500,
    },
    {
        "name": "linear_high_dim",
        "n_features": 10, "n_informative": 4, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 1, "class_sep": 2.0, "weights": None, "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 1000,
    },
    {
        "name": "nonlinear_high_dim",
        "n_features": 10, "n_informative": 4, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 3, "class_sep": 1.0, "weights": None, "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 1000,
    },

    # --- Sparsita' / densita' del segnale ---
    {
        "name": "sparse_signal_high_dim",
        "n_features": 15, "n_informative": 2, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 1, "class_sep": 2.0, "weights": None, "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 500,
    },
    {
        "name": "dense_signal_low_noise",
        "n_features": 10, "n_informative": 8, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 1, "class_sep": 2.0, "weights": None, "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 500,
    },

    # --- Sbilanciamento di classe ---
    {
        "name": "class_imbalance_mild",
        "n_features": 6, "n_informative": 3, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 1, "class_sep": 1.5, "weights": [0.8, 0.2], "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 500,
    },
    {
        "name": "class_imbalance_severe",
        "n_features": 6, "n_informative": 3, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 1, "class_sep": 1.5, "weights": [0.95, 0.05], "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 500,
    },

    # --- Rumore sulle label ---
    {
        "name": "noisy_labels",
        "n_features": 6, "n_informative": 3, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 1, "class_sep": 2.0, "weights": None, "flip_y": 0.15,
        "hypercube": True, "n_explained_instances": 500,
    },

    # --- Separabilita' debole ---
    {
        "name": "weak_separability",
        "n_features": 6, "n_informative": 3, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 1, "class_sep": 0.3, "weights": None, "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 500,
    },

    # --- Confine di decisione molto frammentato ---
    {
        "name": "highly_fragmented_boundary",
        "n_features": 6, "n_informative": 4, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 6, "class_sep": 1.2, "weights": None, "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 500,
    },

    # --- Geometria dei cluster irregolare ---
    {
        "name": "irregular_cluster_geometry",
        "n_features": 6, "n_informative": 3, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 2, "class_sep": 1.5, "weights": None, "flip_y": 0.01,
        "hypercube": False, "n_explained_instances": 500,
    },

    # --- Scale delle feature molto eterogenee ---
    {
        "name": "heterogeneous_feature_scales",
        "n_features": 6, "n_informative": 3, "n_redundant": 0, "n_repeated": 0,
        "n_clusters_per_class": 1, "class_sep": 2.0, "weights": None, "flip_y": 0.01,
        "hypercube": True, "n_explained_instances": 500,
        "feature_scale_multipliers": [10000.0, 1.0, 0.001, 1.0, 100.0, 1.0],
    },
]


def make_synthetic_dataset(scenario, n_samples, random_state=0):
    """Genera il dataset sintetico per uno scenario. shuffle=False garantisce
    che le prime (n_informative + n_redundant + n_repeated) colonne siano
    esattamente le feature che portano segnale reale (ground truth nota); le
    rimanenti sono puro rumore. Se lo scenario specifica
    "feature_scale_multipliers", le colonne vengono riscalate dopo la
    generazione (la ground truth sulle colonne informative non cambia)."""
    n_features = scenario["n_features"]
    n_informative = scenario["n_informative"]
    n_redundant = scenario.get("n_redundant", 0)
    n_repeated = scenario.get("n_repeated", 0)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_clusters_per_class=scenario["n_clusters_per_class"],
        class_sep=scenario["class_sep"],
        weights=scenario.get("weights"),
        flip_y=scenario["flip_y"],
        hypercube=scenario.get("hypercube", True),
        shuffle=False,
        random_state=random_state,
    )

    multipliers = scenario.get("feature_scale_multipliers")
    if multipliers is not None:
        X = X * np.array(multipliers, dtype=float)

    feature_cols = [f"f{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_cols)
    y_s = pd.Series(y.astype(int), name="target")

    n_signal = n_informative + n_redundant + n_repeated
    informative_mask = np.zeros(n_features, dtype=bool)
    informative_mask[:n_signal] = True  # informative + redundant + repeated portano tutte segnale reale
    return X_df, y_s, feature_cols, informative_mask


# ---------------------------------------------------------------------------
# 2. Modello base: Random Forest. n_jobs=1 -- su nodi multi-core, n_jobs=-1
#    introduce un overhead di dispatch che domina il tempo utile quando si
#    chiama predict_proba migliaia di volte su batch piccoli (verificato
#    empiricamente: ~8x piu' lento su un nodo a 64 core).
# ---------------------------------------------------------------------------

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=50, max_depth=6, random_state=0, n_jobs=1
    )
    model.fit(X_train.values, y_train.values)
    return model


def f_predict(model, X):
    """Probabilita' predetta della classe positiva (colonna 1)."""
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# 3. Le 6 strategie di rimozione: ognuna implementa F(x, S) -> predizione,
#    dove S e' l'insieme (maschera booleana) delle feature "presenti".
#    (identiche agli altri due script)
# ---------------------------------------------------------------------------

class RemovalStrategy:
    """Classe base: definisce F(x, mask) marginalizzando le feature assenti."""

    name = "base"

    def __call__(self, x, mask):
        raise NotImplementedError


class DefaultValues(RemovalStrategy):
    name = "default_values"

    def __init__(self, model, X_train):
        self.model = model
        self.default = X_train.mean(axis=0).values

    def __call__(self, x, mask):
        x_masked = x.copy()
        x_masked[~mask] = self.default[~mask]
        return f_predict(self.model, x_masked.reshape(1, -1))[0]


class Uniform(RemovalStrategy):
    name = "uniform"

    def __init__(self, model, X_train, n_samples=16):
        self.model = model
        self.lo = X_train.min(axis=0).values
        self.hi = X_train.max(axis=0).values
        self.n_samples = n_samples

    def __call__(self, x, mask):
        samples = RNG.uniform(self.lo, self.hi, size=(self.n_samples, len(x)))
        samples[:, mask] = x[mask]
        preds = f_predict(self.model, samples)
        return preds.mean()


class ProductMarginals(RemovalStrategy):
    """Marginalizza usando il prodotto delle marginali: ogni colonna assente e'
    campionata indipendentemente dalle altre, mischiando i valori osservati
    nel training set colonna per colonna."""

    name = "product"

    def __init__(self, model, X_train, n_samples=16):
        self.model = model
        self.X_train = X_train.values
        self.n_samples = n_samples

    def __call__(self, x, mask):
        idx = RNG.integers(0, len(self.X_train), size=(self.n_samples, len(x)))
        samples = self.X_train[idx, np.arange(len(x))]  # ogni colonna campionata a se'
        samples[:, mask] = x[mask]
        preds = f_predict(self.model, samples)
        return preds.mean()


class JointMarginal(RemovalStrategy):
    """Marginalizza usando la distribuzione marginale congiunta: si campionano
    righe intere dal training set, preservando le correlazioni tra le feature
    assenti."""

    name = "marginal"

    def __init__(self, model, X_train, n_samples=16):
        self.model = model
        self.X_train = X_train.values
        self.n_samples = n_samples

    def __call__(self, x, mask):
        idx = RNG.integers(0, len(self.X_train), size=self.n_samples)
        samples = self.X_train[idx].copy()
        samples[:, mask] = x[mask]
        preds = f_predict(self.model, samples)
        return preds.mean()


def RandomForestRegressorSafe():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0, n_jobs=1)


class Surrogate(RemovalStrategy):
    """Allena un modello surrogato g(x*mask, mask) a imitare f quando alcune
    feature sono mascherate (mascheramento casuale indipendente durante il
    training), come in L2X / REAL-X (Appendix E.3 del paper Covert et al.)."""

    name = "surrogate"

    def __init__(self, model, X_train, n_masks_per_row=4):
        self.model = model
        n, d = X_train.shape
        X_aug, y_aug = [], []
        Xv = X_train.values
        base_preds = f_predict(model, Xv)
        for i in range(n):
            for _ in range(n_masks_per_row):
                mask = RNG.integers(0, 2, size=d).astype(bool)
                x_masked = Xv[i].copy()
                x_masked[~mask] = 0.0  # valore "non osservato" -> 0 + indicatore
                X_aug.append(np.concatenate([x_masked, mask.astype(float)]))
                y_aug.append(base_preds[i])
        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)

        self.surrogate = RandomForestRegressorSafe()
        self.surrogate.fit(X_aug, y_aug)

    def __call__(self, x, mask):
        x_masked = x.copy()
        x_masked[~mask] = 0.0
        feat = np.concatenate([x_masked, mask.astype(float)]).reshape(1, -1)
        return self.surrogate.predict(feat)[0]


class SeparateModels(RemovalStrategy):
    """Allena un modello RF diverso per OGNI sottoinsieme di feature (fattibile
    solo perche' qui abbiamo poche feature: al piu' 2^10 = 1024 sottoinsiemi)."""

    name = "separate_models"

    def __init__(self, X_train, y_train):
        n, d = X_train.shape
        self.d = d
        self.models = {}
        Xv = X_train.values
        yv = y_train.values
        all_subsets = [subset for r in range(d + 1) for subset in itertools.combinations(range(d), r)]
        for subset in tqdm(all_subsets, desc="  separate_models: training per subset", leave=False):
            if len(subset) == 0:
                # nessuna feature: predici la prevalenza media della classe
                self.models[subset] = None
                self.prior = yv.mean()
                continue
            cols = list(subset)
            m = RandomForestClassifier(
                n_estimators=15, max_depth=5, random_state=0, n_jobs=1
            )
            m.fit(Xv[:, cols], yv)
            self.models[subset] = m

    def __call__(self, x, mask):
        subset = tuple(np.where(mask)[0])
        m = self.models[subset]
        if m is None:
            return self.prior
        return m.predict_proba(x[list(subset)].reshape(1, -1))[:, 1][0]


# ---------------------------------------------------------------------------
# 4. FAITHFULNESS (Bhatt et al., 2020). Identica agli altri due script.
# ---------------------------------------------------------------------------

def faithfulness_correlation(F, phi, x, n_features, n_subsets=50, min_size=1):
    full_mask = np.ones(n_features, dtype=bool)
    v_full = F(x, full_mask)

    attributed_sums = []
    actual_drops = []
    for _ in range(n_subsets):
        k = RNG.integers(min_size, n_features)
        S = RNG.choice(n_features, size=k, replace=False)

        mask = full_mask.copy()
        mask[S] = False
        v_removed = F(x, mask)

        attributed_sums.append(phi[S].sum())
        actual_drops.append(v_full - v_removed)

    attributed_sums = np.array(attributed_sums)
    actual_drops = np.array(actual_drops)

    if np.std(attributed_sums) == 0 or np.std(actual_drops) == 0:
        return np.nan
    return np.corrcoef(attributed_sums, actual_drops)[0, 1]


# ---------------------------------------------------------------------------
# 5. Ranking delle feature per Incremental Deletion / Addition.
#    Identico agli altri due script.
# ---------------------------------------------------------------------------

def rank_features(phi, ranking_mode):
    n_features = len(phi)
    if ranking_mode == "absolute":
        return np.argsort(-np.abs(phi))
    elif ranking_mode == "signed":
        return np.argsort(-phi)
    elif ranking_mode == "positive":
        idx = np.arange(n_features)
        pos_idx = idx[phi > 0]
        rest_idx = idx[phi <= 0]
        pos_sorted = pos_idx[np.argsort(-phi[pos_idx])]
        return np.concatenate([pos_sorted, rest_idx])
    else:
        raise ValueError(f"ranking_mode non riconosciuto: {ranking_mode!r}")


# ---------------------------------------------------------------------------
# 6. INCREMENTAL DELETION / ADDITION (Petsiuk et al., 2018 - RISE) e le loro
#    versioni ACCURACY (Kapishnikov et al., XRAI 2019). Identiche agli altri
#    due script.
# ---------------------------------------------------------------------------

def incremental_deletion(F, x, order, n_features):
    full_mask = np.ones(n_features, dtype=bool)
    mask = full_mask.copy()
    preds = [F(x, mask)]
    for feat_idx in order:
        mask = mask.copy()
        mask[feat_idx] = False
        preds.append(F(x, mask))
    preds = np.array(preds)
    fraction = np.arange(n_features + 1) / n_features
    auc = float(_trapz(preds, fraction))
    return preds, fraction, auc


def incremental_addition(F, x, order, n_features):
    empty_mask = np.zeros(n_features, dtype=bool)
    mask = empty_mask.copy()
    preds = [F(x, mask)]
    for feat_idx in order:
        mask = mask.copy()
        mask[feat_idx] = True
        preds.append(F(x, mask))
    preds = np.array(preds)
    fraction = np.arange(n_features + 1) / n_features
    auc = float(_trapz(preds, fraction))
    return preds, fraction, auc


def bind_target_class(strategy, target_class):
    """Restituisce un wrapper F(x, mask) -> P(classe = target_class), invece
    del comportamento originale (P(classe=1) fissa). Per classificazione
    binaria vale P(classe=0) = 1 - P(classe=1), quindi non serve ri-addestrare
    le strategie supervisionate (surrogate, separate_models)."""
    if target_class == 1:
        return strategy
    return lambda x, mask: 1.0 - strategy(x, mask)


def _is_correct(F, x, mask, threshold):
    """F e' gia' vincolata alla classe target (via bind_target_class):
    'corretto' = 'la classe target resta ancora quella favorita'."""
    return 1.0 if F(x, mask) >= threshold else 0.0


def incremental_deletion_accuracy(F, x, order, n_features, threshold=0.5):
    full_mask = np.ones(n_features, dtype=bool)
    mask = full_mask.copy()
    vals = [_is_correct(F, x, mask, threshold)]
    for feat_idx in order:
        mask = mask.copy()
        mask[feat_idx] = False
        vals.append(_is_correct(F, x, mask, threshold))
    vals = np.array(vals)
    fraction = np.arange(n_features + 1) / n_features
    auc = float(_trapz(vals, fraction))
    return vals, fraction, auc


def incremental_addition_accuracy(F, x, order, n_features, threshold=0.5):
    empty_mask = np.zeros(n_features, dtype=bool)
    mask = empty_mask.copy()
    vals = [_is_correct(F, x, mask, threshold)]
    for feat_idx in order:
        mask = mask.copy()
        mask[feat_idx] = True
        vals.append(_is_correct(F, x, mask, threshold))
    vals = np.array(vals)
    fraction = np.arange(n_features + 1) / n_features
    auc = float(_trapz(vals, fraction))
    return vals, fraction, auc


def random_ranking_baseline(F, x, n_features, n_random_rankings, random_state=0,
                             threshold=0.5, compute_accuracy=True):
    """Baseline opzionale: media di deletion_auc/addition_auc su ranking
    casuali delle feature. Se compute_accuracy=True, calcola anche la
    baseline delle versioni accuracy, riusando GLI STESSI ranking casuali.
    F: UNA SOLA funzione, gia' vincolata alla classe target, usata sia per
    le curve a probabilita' che per le versioni accuracy."""
    rng = np.random.default_rng(random_state)
    deletion_aucs = []
    addition_aucs = []
    deletion_acc_aucs = []
    addition_acc_aucs = []
    for _ in range(n_random_rankings):
        order = rng.permutation(n_features)
        _, _, d_auc = incremental_deletion(F, x, order, n_features)
        _, _, a_auc = incremental_addition(F, x, order, n_features)
        deletion_aucs.append(d_auc)
        addition_aucs.append(a_auc)
        if compute_accuracy:
            _, _, d_acc_auc = incremental_deletion_accuracy(F, x, order, n_features, threshold)
            _, _, a_acc_auc = incremental_addition_accuracy(F, x, order, n_features, threshold)
            deletion_acc_aucs.append(d_acc_auc)
            addition_acc_aucs.append(a_acc_auc)

    result = (float(np.mean(deletion_aucs)), float(np.mean(addition_aucs)))
    if compute_accuracy:
        result = result + (float(np.mean(deletion_acc_aucs)), float(np.mean(addition_acc_aucs)))
    return result


# ---------------------------------------------------------------------------
# 7. COMPLEXITY (Tseng, Shrikumar & Kundaje, NeurIPS 2020). Identica agli
#    altri due script.
# ---------------------------------------------------------------------------

def complexity(phi, log_base="e", eps=1e-12):
    """Entropia di Shannon della distribuzione normalizzata |phi_i| / sum(|phi|)."""
    abs_phi = np.abs(phi)
    total = abs_phi.sum()
    if total <= eps:
        return 0.0
    p = abs_phi / total
    p = p[p > eps]
    entropy = -(p * np.log(p)).sum()
    if log_base == "2":
        entropy = entropy / np.log(2)
    return float(entropy)


# ---------------------------------------------------------------------------
# 8. Spiegazioni SHAP, KernelSHAP e LIME, generate con le impostazioni di
#    DEFAULT delle rispettive librerie. Identiche agli altri due script.
# ---------------------------------------------------------------------------

def compute_shap_values(model, instances, n_features, target_classes):
    """SHAP con il suo esplicatore di default per modelli ad albero
    (shap.TreeExplainer), nessun parametro custom. Spiega, per ciascuna
    istanza, la classe PREDETTA dal modello (non piu' sempre la classe 1)."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(instances.values)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    else:
        sv = np.asarray(sv)  # gia' (n, d, 2)
    out = np.zeros(sv.shape[:2])
    for i in range(sv.shape[0]):
        out[i] = sv[i, :, int(target_classes[i])]
    return out


def compute_kernelshap_values(model, X_train, instances, target_classes):
    """KernelSHAP (shap.KernelExplainer), impostazioni di default (nsamples="auto").
    Spiega, per ciascuna istanza, la classe PREDETTA dal modello."""
    background = shap.sample(X_train, 50, random_state=0)
    explainer = shap.KernelExplainer(lambda data: model.predict_proba(data), background)
    sv = explainer.shap_values(instances.values, nsamples="auto", silent=True)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    else:
        sv = np.asarray(sv)  # gia' (n, d, 2)
    out = np.zeros(sv.shape[:2])
    for i in range(sv.shape[0]):
        out[i] = sv[i, :, int(target_classes[i])]
    return out


def compute_lime_values(model, X_train, instances, feature_cols, target_classes):
    """LIME con LimeTabularExplainer di default (discretize_continuous=True,
    num_samples=5000 di default internamente). Spiega, per ciascuna istanza,
    la classe PREDETTA dal modello."""
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_cols,
        class_names=["classe_0", "classe_1"],
        mode="classification",
        random_state=0,
    )
    n, d = len(instances), len(feature_cols)
    phi = np.zeros((n, d))
    for i in tqdm(range(n), desc="  LIME: spiegazione per istanza", leave=False):
        target = int(target_classes[i])
        exp = explainer.explain_instance(
            instances.iloc[i].values,
            lambda data: model.predict_proba(data),
            num_features=d,
            labels=(target,),
        )
        feat_weight = dict(exp.as_map()[target])
        for j in range(d):
            phi[i, j] = feat_weight.get(j, 0.0)
    return phi


# ---------------------------------------------------------------------------
# 9. Metrica NUOVA (non presente sui dataset reali, dove la ground truth non
#    e' nota): accordo tra le attribuzioni e le feature realmente informative
#    (informative + redundant + repeated) usate da make_classification.
# ---------------------------------------------------------------------------

def ground_truth_agreement(phi_matrix, informative_mask, top_k=None):
    """Aggrega |phi| sulle istanze (media) e la confronta con la ground truth.

    - auc: ROC-AUC trattando l'importanza media |phi| come score e
      "e' una feature che porta segnale" come etichetta binaria.
      1.0 = perfetto ordinamento, 0.5 = equivalente a un ranking casuale.
    - precision_at_k: frazione delle top-k feature (per |phi| medio) che
      portano davvero segnale, con k = numero di feature "segnale" dello
      scenario (o top_k se specificato esplicitamente).
    """
    mean_abs_phi = np.abs(phi_matrix).mean(axis=0)
    n_signal = int(informative_mask.sum())
    n_features = len(informative_mask)

    if n_signal == 0 or n_signal == n_features:
        auc = np.nan
    else:
        auc = float(roc_auc_score(informative_mask.astype(int), mean_abs_phi))

    k = top_k if top_k is not None else n_signal
    top_k_idx = np.argsort(-mean_abs_phi)[:k]
    precision_at_k = float(informative_mask[top_k_idx].mean())

    return auc, precision_at_k


# ---------------------------------------------------------------------------
# 10. Un esperimento completo per UNO scenario sintetico: stessa pipeline
#     degli altri due script (6 strategie, 3 explainer, Faithfulness,
#     Deletion/Addition + versioni Accuracy, Complexity), PIU' la metrica di
#     accordo con la ground truth.
# ---------------------------------------------------------------------------

def run_scenario(scenario):
    print("=" * 78)
    print(f"SCENARIO SINTETICO: {scenario['name']}")
    print(
        f"  n_features={scenario['n_features']}, n_informative={scenario['n_informative']}, "
        f"n_redundant={scenario.get('n_redundant', 0)}, n_repeated={scenario.get('n_repeated', 0)}"
    )
    print(
        f"  n_clusters_per_class={scenario['n_clusters_per_class']}, class_sep={scenario['class_sep']}, "
        f"weights={scenario.get('weights')}, flip_y={scenario['flip_y']}, "
        f"hypercube={scenario.get('hypercube', True)}"
    )
    if scenario.get("feature_scale_multipliers") is not None:
        print(f"  feature_scale_multipliers={scenario['feature_scale_multipliers']}")
    print("=" * 78)

    X, y, feature_cols, informative_mask = make_synthetic_dataset(
        scenario, CONFIG["n_samples"], random_state=CONFIG["random_state"]
    )
    n_features = len(feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    model = train_model(X_train, y_train)
    acc = model.score(X_test.values, y_test.values)
    print(f"{len(X)} righe, {n_features} feature ({informative_mask.sum()} portano segnale + "
          f"{n_features - informative_mask.sum()} rumore) -> {feature_cols}")
    print(f"Feature con segnale (ground truth): {[f for f, m in zip(feature_cols, informative_mask) if m]}")
    print(f"Bilanciamento classi nel training set: {y_train.mean():.3f}")
    print(f"Accuratezza RandomForest sul test set: {acc:.3f}\n")

    print("Costruzione delle strategie di rimozione (puo' richiedere qualche secondo)...")
    strategies = {
        "default_values": DefaultValues(model, X_train),
        "uniform": Uniform(model, X_train),
        "product": ProductMarginals(model, X_train),
        "marginal": JointMarginal(model, X_train),
        "surrogate": Surrogate(model, X_train),
        "separate_models": SeparateModels(X_train, y_train),
    }
    print("Strategie pronte.\n")

    faith_cfg = CONFIG["faithfulness"]
    delins_cfg = CONFIG["deletion_insertion"]
    complexity_cfg = CONFIG["complexity"]
    gt_cfg = CONFIG["ground_truth"]

    scenario_n = scenario.get("n_explained_instances", 500)
    n_faith = min(scenario_n, faith_cfg["n_instances"] if faith_cfg["n_instances"] is not None else scenario_n, len(X_test))
    n_delins = min(scenario_n, delins_cfg["n_instances"] if delins_cfg["n_instances"] is not None else scenario_n, len(X_test))
    n_complexity = min(scenario_n, complexity_cfg["n_instances"] if complexity_cfg["n_instances"] is not None else scenario_n, len(X_test))
    n_gt = min(scenario_n, gt_cfg["n_instances"] if gt_cfg["n_instances"] is not None else scenario_n, len(X_test))
    n_max = max(n_faith, n_delins, n_complexity, n_gt)

    # Le spiegazioni si generano UNA sola volta per le n_max istanze necessarie;
    # ogni metrica poi usa solo il prefisso di cui ha bisogno.
    instances = X_test.iloc[:n_max].reset_index(drop=True)
    y_instances = y_test.iloc[:n_max].reset_index(drop=True)

    print("Generazione spiegazioni SHAP, KernelSHAP e LIME (default di libreria) -- classe PREDETTA...")
    predicted_classes = (f_predict(model, instances.values) >= 0.5).astype(int)
    shap_phi = compute_shap_values(model, instances, n_features, predicted_classes)
    kernelshap_phi = compute_kernelshap_values(model, X_train, instances, predicted_classes)
    lime_phi = compute_lime_values(model, X_train, instances, feature_cols, predicted_classes)
    explanations = {
        "SHAP": shap_phi,
        "KernelSHAP": kernelshap_phi,
        "LIME": lime_phi,
    }

    # =========================================================================
    # GROUND TRUTH AGREEMENT: n_instances proprio, deterministica (nessun run).
    # =========================================================================
    print("Calcolo dell'accordo con la ground truth (feature con segnale reale)...")
    ground_truth_rows = []
    for method_name, phi_matrix in explanations.items():
        auc, precision_at_k = ground_truth_agreement(
            phi_matrix[:n_gt], informative_mask, top_k=gt_cfg["top_k_for_precision"]
        )
        ground_truth_rows.append({
            "scenario": scenario["name"],
            "explainer": method_name,
            "n_features": n_features,
            "n_signal_features": int(informative_mask.sum()),
            "n_instances": n_gt,
            "ground_truth_auc": auc,
            "ground_truth_precision_at_k": precision_at_k,
        })
    print(pd.DataFrame(ground_truth_rows).set_index("explainer")[
        ["ground_truth_auc", "ground_truth_precision_at_k"]
    ].round(4))
    print()

    # =========================================================================
    # COMPLEXITY: n_instances/n_runs propri. Deterministica.
    # =========================================================================
    complexity_rows = []
    for run in range(complexity_cfg["n_runs"]):
        for method_name, phi_matrix in explanations.items():
            for i in range(n_complexity):
                phi = phi_matrix[i]
                val = complexity(phi, log_base=complexity_cfg["log_base"], eps=complexity_cfg["eps"])
                complexity_rows.append({
                    "scenario": scenario["name"],
                    "run": run,
                    "instance_id": i,
                    "explainer": method_name,
                    "complexity": val,
                })

    print("--- Complexity (indipendente dalla removal strategy) ---")
    complexity_df = pd.DataFrame(complexity_rows)
    print(complexity_df.groupby("explainer")[["complexity"]].mean().round(4))
    print()

    # =========================================================================
    # FAITHFULNESS: n_instances/n_runs propri. Subset_size campionato
    # casualmente ad ogni prova (INVARIATO).
    # =========================================================================
    global RNG
    faithfulness_rows = []
    faithfulness_iter = [
        (method_name, strat_name)
        for method_name in explanations
        for strat_name in strategies
    ]
    for run in range(faith_cfg["n_runs"]):
        RNG = np.random.default_rng(faith_cfg["random_state"] + run)
        for method_name, strat_name in tqdm(
            faithfulness_iter, desc=f"Faithfulness [run {run}] [explainer x strategia]"
        ):
            strat = strategies[strat_name]
            phi_matrix = explanations[method_name]
            for i in range(n_faith):
                x = instances.iloc[i].values
                phi = phi_matrix[i]
                F = bind_target_class(strat, int(predicted_classes[i]))
                score = faithfulness_correlation(F, phi, x, n_features, n_subsets=faith_cfg["n_subsets"])
                faithfulness_rows.append({
                    "scenario": scenario["name"],
                    "run": run,
                    "instance_id": i,
                    "explainer": method_name,
                    "removal_strategy": strat_name,
                    "faithfulness": score,
                })

    print(f"\n--- Faithfulness correlation (n_instances={n_faith}, n_runs={faith_cfg['n_runs']}) ---")
    faithfulness_df = pd.DataFrame(faithfulness_rows)
    print(faithfulness_df.groupby(["explainer", "removal_strategy"])[["faithfulness"]].agg(
        ["mean", "std"]
    ).round(4))
    print()

    # =========================================================================
    # INCREMENTAL DELETION / ADDITION (+ versioni ACCURACY): n_instances/n_runs
    # propri.
    # =========================================================================
    curves_rows = []
    metrics_rows = []

    main_loop_iter = [
        (method_name, strat_name)
        for method_name in explanations
        for strat_name in strategies
    ]
    for run in range(delins_cfg["n_runs"]):
        RNG = np.random.default_rng(delins_cfg["random_state"] + run)
        random_baseline_cache = {}

        for method_name, strat_name in tqdm(
            main_loop_iter, desc=f"Deletion/Addition [run {run}] [explainer x strategia]"
        ):
            phi_matrix = explanations[method_name]
            strat = strategies[strat_name]
            for i in range(n_delins):
                x = instances.iloc[i].values
                phi = phi_matrix[i]
                F = bind_target_class(strat, int(predicted_classes[i]))

                order = rank_features(phi, delins_cfg["ranking_mode"])

                del_preds, del_fraction, del_auc = incremental_deletion(F, x, order, n_features)
                for step, (frac, pred) in enumerate(zip(del_fraction, del_preds)):
                    curves_rows.append({
                        "scenario": scenario["name"], "run": run, "instance_id": i, "explainer": method_name,
                        "removal_strategy": strat_name, "metric": "deletion",
                        "step": step, "fraction_features": frac, "prediction": pred,
                    })

                add_preds, add_fraction, add_auc = incremental_addition(F, x, order, n_features)
                for step, (frac, pred) in enumerate(zip(add_fraction, add_preds)):
                    curves_rows.append({
                        "scenario": scenario["name"], "run": run, "instance_id": i, "explainer": method_name,
                        "removal_strategy": strat_name, "metric": "addition",
                        "step": step, "fraction_features": frac, "prediction": pred,
                    })

                del_acc_vals, del_acc_fraction, del_acc_auc = incremental_deletion_accuracy(
                    F, x, order, n_features, threshold=delins_cfg["accuracy_threshold"]
                )
                for step, (frac, val) in enumerate(zip(del_acc_fraction, del_acc_vals)):
                    curves_rows.append({
                        "scenario": scenario["name"], "run": run, "instance_id": i, "explainer": method_name,
                        "removal_strategy": strat_name, "metric": "deletion_accuracy",
                        "step": step, "fraction_features": frac, "prediction": val,
                    })

                add_acc_vals, add_acc_fraction, add_acc_auc = incremental_addition_accuracy(
                    F, x, order, n_features, threshold=delins_cfg["accuracy_threshold"]
                )
                for step, (frac, val) in enumerate(zip(add_acc_fraction, add_acc_vals)):
                    curves_rows.append({
                        "scenario": scenario["name"], "run": run, "instance_id": i, "explainer": method_name,
                        "removal_strategy": strat_name, "metric": "addition_accuracy",
                        "step": step, "fraction_features": frac, "prediction": val,
                    })

                metrics_row = {
                    "scenario": scenario["name"],
                    "run": run,
                    "instance_id": i,
                    "explainer": method_name,
                    "removal_strategy": strat_name,
                    "deletion_auc": del_auc,
                    "addition_auc": add_auc,
                    "deletion_accuracy_auc": del_acc_auc,
                    "addition_accuracy_auc": add_acc_auc,
                }

                if delins_cfg["compute_random_baseline"]:
                    cache_key = (strat_name, i)
                    if cache_key not in random_baseline_cache:
                        random_baseline_cache[cache_key] = random_ranking_baseline(
                            F, x, n_features,
                            n_random_rankings=delins_cfg["n_random_rankings"],
                            random_state=delins_cfg["random_state"] + run * 1000 + i,
                            threshold=delins_cfg["accuracy_threshold"],
                            compute_accuracy=True,
                        )
                    rand_del_auc, rand_add_auc, rand_del_acc_auc, rand_add_acc_auc = random_baseline_cache[cache_key]
                    metrics_row["random_deletion_auc"] = rand_del_auc
                    metrics_row["random_addition_auc"] = rand_add_auc
                    metrics_row["deletion_auc_vs_random"] = del_auc - rand_del_auc
                    metrics_row["addition_auc_vs_random"] = add_auc - rand_add_auc
                    metrics_row["random_deletion_accuracy_auc"] = rand_del_acc_auc
                    metrics_row["random_addition_accuracy_auc"] = rand_add_acc_auc
                    metrics_row["deletion_accuracy_auc_vs_random"] = del_acc_auc - rand_del_acc_auc
                    metrics_row["addition_accuracy_auc_vs_random"] = add_acc_auc - rand_add_acc_auc

                metrics_rows.append(metrics_row)

    metrics_df = pd.DataFrame(metrics_rows)
    print(f"\n--- Deletion AUC / Addition AUC (n_instances={n_delins}, n_runs={delins_cfg['n_runs']}) ---")
    print(metrics_df.groupby(["explainer", "removal_strategy"])[
        ["deletion_auc", "addition_auc"]
    ].mean().round(4))
    print()

    return curves_rows, metrics_rows, faithfulness_rows, complexity_rows, ground_truth_rows


def _scenario_path(base_filename, scenario_name):
    """Nome del file di checkpoint per un singolo scenario, es.
    'synthetic_extended_curves.csv' -> 'synthetic_extended_curves_scenario_baseline_linear.csv'."""
    name, ext = base_filename.rsplit(".", 1)
    return f"{CONFIG['output_dir']}/{name}_scenario_{scenario_name}.{ext}"


_OUTPUT_BASE_FILENAMES = {
    "curves": (CONFIG["curves_csv"],),
    "metrics": (CONFIG["metrics_csv"],),
    "faithfulness": (CONFIG["faithfulness_csv"],),
    "complexity": (CONFIG["complexity_csv"],),
    "ground_truth": (CONFIG["ground_truth_csv"],),
}


def _scenario_already_done(scenario_name):
    """True se TUTTI i 5 file di checkpoint di questo scenario esistono gia'."""
    for (base,) in _OUTPUT_BASE_FILENAMES.values():
        if not os.path.exists(_scenario_path(base, scenario_name)):
            return False
    return True


def main():
    # CHECKPOINT/RESUME per singolo scenario: prima di calcolare uno scenario,
    # si controlla se i suoi 5 file di checkpoint esistono gia' su disco -- se
    # si', lo si salta (nessun ricalcolo). Se il job si interrompe a meta',
    # un rilancio successivo riparte esattamente dallo scenario giusto senza
    # rifare quelli gia' completati.
    #
    # RETROCOMPATIBILITA': alla fine, i checkpoint di TUTTI gli scenari
    # (quelli appena calcolati + quelli gia' presenti da run precedenti)
    # vengono concatenati negli stessi 5 CSV aggregati di sempre (stessi nomi
    # file di prima), cosi' qualunque analisi/script a valle che si aspetta
    # quei file continua a funzionare senza modifiche.
    for scenario in tqdm(SCENARIOS, desc="SCENARIO"):
        name = scenario["name"]
        if _scenario_already_done(name):
            print(f"[{name}] gia' completato (checkpoint trovato) -- SALTATO.")
            continue

        curves_rows, metrics_rows, faithfulness_rows, complexity_rows, ground_truth_rows = run_scenario(scenario)

        pd.DataFrame(curves_rows).to_csv(_scenario_path(CONFIG["curves_csv"], name), index=False)
        pd.DataFrame(metrics_rows).to_csv(_scenario_path(CONFIG["metrics_csv"], name), index=False)
        pd.DataFrame(faithfulness_rows).to_csv(_scenario_path(CONFIG["faithfulness_csv"], name), index=False)
        pd.DataFrame(complexity_rows).to_csv(_scenario_path(CONFIG["complexity_csv"], name), index=False)
        pd.DataFrame(ground_truth_rows).to_csv(_scenario_path(CONFIG["ground_truth_csv"], name), index=False)
        print(f"[{name}] checkpoint salvato.")

    # --- Concatenazione finale nei file aggregati (retrocompatibilita') -----
    def _concat_checkpoints(base_filename):
        frames = [
            pd.read_csv(_scenario_path(base_filename, scenario["name"]))
            for scenario in SCENARIOS
        ]
        return pd.concat(frames, ignore_index=True)

    curves_df = _concat_checkpoints(CONFIG["curves_csv"])
    metrics_df = _concat_checkpoints(CONFIG["metrics_csv"])
    faithfulness_df = _concat_checkpoints(CONFIG["faithfulness_csv"])
    complexity_df = _concat_checkpoints(CONFIG["complexity_csv"])
    ground_truth_df = _concat_checkpoints(CONFIG["ground_truth_csv"])

    curves_path = f"{CONFIG['output_dir']}/{CONFIG['curves_csv']}"
    metrics_path = f"{CONFIG['output_dir']}/{CONFIG['metrics_csv']}"
    faithfulness_path = f"{CONFIG['output_dir']}/{CONFIG['faithfulness_csv']}"
    complexity_path = f"{CONFIG['output_dir']}/{CONFIG['complexity_csv']}"
    ground_truth_path = f"{CONFIG['output_dir']}/{CONFIG['ground_truth_csv']}"
    curves_df.to_csv(curves_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    faithfulness_df.to_csv(faithfulness_path, index=False)
    complexity_df.to_csv(complexity_path, index=False)
    ground_truth_df.to_csv(ground_truth_path, index=False)

    print("=" * 78)
    print("RIEPILOGO: accordo con la ground truth per scenario x explainer")
    print("=" * 78)
    print(ground_truth_df.pivot_table(
        index="scenario", columns="explainer", values="ground_truth_auc"
    ).round(4))
    print()
    print(f"Curve complete salvate in:                 {curves_path}  ({len(curves_df)} righe)")
    print(f"Deletion/Addition AUC (instance-level) in:  {metrics_path}  ({len(metrics_df)} righe)")
    print(f"Faithfulness (raw, tutte le run) in:        {faithfulness_path}  ({len(faithfulness_df)} righe)")
    print(f"Complexity (raw, tutte le run) in:          {complexity_path}  ({len(complexity_df)} righe)")
    print(f"Accordo con la ground truth in:             {ground_truth_path}  ({len(ground_truth_df)} righe)")


if __name__ == "__main__":
    main()