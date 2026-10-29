"""
Replica di removal_strategies_experiments.py (le STESSE casistiche: varianti
di modello, esplorazione target/ranking separata, checkpoint/resume) ma su 3
dataset REALI MULTI-CLASSE invece di 4 binari:

    - Glass                (UCI, 6 classi reali sbilanciate, 9 feature)
    - Wine Quality (red)    (UCI, 6 classi ordinali molto sovrapposte, 8 feature su 11)
    - Digits                (sklearn built-in, 10 classi, 8 pixel su 64)

Glass e Wine Quality sono scaricati da un mirror GitHub (unico dominio
esterno raggiungibile in questo sandbox) e sono stati scelti al posto di
Iris/Wine (sklearn) perche' piu' "sfidanti": classi sbilanciate e sovrapposte,
problemi genuinamente difficili in letteratura -- non i classici dataset-
giocattolo quasi perfettamente separabili. Le loro label originali (non
contigue da 0) vengono rimappate con pd.Categorical(...).codes.

DIFFERENZA PRINCIPALE rispetto allo script binario: l'explanation_target
"positive" (classe 1 fissa) e' stato RIMOSSO. Con N>2 classi non esiste una
"classe 1" privilegiata -- fissare arbitrariamente una singola classe come
target universale non avrebbe alcun significato/giustificazione (a differenza
del binario, dove "positive" coincide spesso con la classe di interesse
convenzionale, es. "rischio", "recidiva"). Restano quindi solo:
    - "true_class":      spiega la classe VERA dell'istanza
    - "predicted_class":  spiega la classe PREDETTA dal modello

GENERALIZZAZIONE MULTI-CLASSE (identica a multiclass_experiments.py):
  - le 6 removal strategy restituiscono, a scelta, lo scalare P(classe=k)
    oppure l'intero vettore di probabilita' (necessario per il check di
    correttezza via argmax, che sostituisce la soglia fissa 0.5 -- valida
    solo in binario)
  - Surrogate e' un regressore MULTI-OUTPUT (un solo modello per tutte le
    classi, non uno per classe)
  - SeparateModels riallinea le colonne su m.classes_ (un sottomodello puo'
    non aver visto tutte le classi nel suo sottoinsieme di feature)
  - la variante "untrained" genera rumore su TUTTE le n_classes (non solo
    0/1)

QUESTO FILE E' COMPLETAMENTE AUTONOMO.
"""

import itertools
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

import shap
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm

RNG = np.random.default_rng(0)

_trapz = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------------------
# 0. Configurazione centralizzata.
# ---------------------------------------------------------------------------

CONFIG = {
    "base_path": "data",  # directory populated by scripts/download_data.py

    "faithfulness": {
        "n_instances": None,
        "n_runs": 1,
        "n_subsets": 20,
        "random_state": 0,
    },

    "deletion_insertion": {
        "n_instances": None,
        "n_runs": 1,
        "ranking_modes": ["signed", "absolute"],  # ranking, non c'entra col target di classe
        "compute_random_baseline": True,
        "n_random_rankings": 5,
        "random_state": 0,
    },

    # SOLO 2 target (niente "positive": nessuna classe privilegiata a N classi)
    "explanation_targets": ["true_class", "predicted_class"],

    "explanation_target_default_for_ranking_test": "predicted_class",
    "ranking_mode_default_for_target_test": "absolute",

    "model_variants": {
        "standard": {
            "model_type": "rf",
            "n_estimators": 50, "max_depth": 6,
            "explanation_targets": None,
            "ranking_modes": None,
        },
        "shallow": {
            "model_type": "rf",
            "n_estimators": 5, "max_depth": 1,
            "explanation_targets": ["predicted_class"],
            "ranking_modes": ["absolute"],
        },
        "untrained": {
            "model_type": "rf",
            "n_estimators": 50, "max_depth": 6,
            "untrained": True,
            "explanation_targets": ["predicted_class"],
            "ranking_modes": ["absolute"],
        },
    },

    "complexity": {
        "n_instances": None,
        "n_runs": 1,
        "log_base": "e",
        "eps": 1e-12,
    },

    "output_dir": "results",
    "curves_csv": "removal_strategies_multiclass_curves.csv",
    "metrics_csv": "removal_strategies_multiclass_instance_metrics.csv",
    "faithfulness_csv": "removal_strategies_multiclass_faithfulness_runs.csv",
    "complexity_csv": "removal_strategies_multiclass_complexity_runs.csv",
}

# ---------------------------------------------------------------------------
# 1. Caricamento dei 3 dataset reali multi-classe.
#
#    Glass e Wine Quality (red) sono presi da UCI (via mirror GitHub, unico
#    dominio esterno raggiungibile in questo sandbox) -- scelti perche' piu'
#    "sfidanti" di Iris/Wine (sklearn): entrambi hanno classi sbilanciate e
#    sovrapposte, problemi genuinamente difficili in letteratura (Wine
#    Quality in particolare e' un classico esempio di task con accuratezza
#    tipicamente sotto il 65% per la forte sovrapposizione tra classi
#    ordinali adiacenti). Digits resta il terzo, per il caso "molte classi".
#
#    Le label originali di Glass (1,2,3,5,6,7 -- la classe 4 non e' presente
#    nel dataset) e di Wine Quality (3..8) NON sono contigue da 0: le
#    rimappiamo a 0..n_classi-1 con pd.Categorical(...).codes, perche' il
#    resto della pipeline assume che l'indice di colonna di predict_proba
#    corrisponda direttamente al valore della classe.
# ---------------------------------------------------------------------------

def load_glass_data(path=None):
    """Glass Identification (UCI). 213 righe, 9 feature, 6 classi reali
    (sbilanciate: da 9 a 76 istanze per classe)."""
    path = os.path.join(CONFIG.get("base_path", ""), path or "glass.csv")
    columns = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
    df = pd.read_csv(path, header=None, names=columns)
    feature_cols = columns[:-1]
    X = df[feature_cols].astype(float).reset_index(drop=True)
    y_codes = pd.Categorical(df["Type"]).codes
    y = pd.Series(y_codes.astype(int), name="target").reset_index(drop=True)
    return X, y, feature_cols


def load_wine_quality_data(path=None, n_features=8):
    """Wine Quality - red (UCI). 1599 righe, 11 feature (ridotte a 8 piu'
    variabili), 6 classi ordinali (quality 3..8) molto sovrapposte -- task
    notoriamente difficile."""
    path = os.path.join(CONFIG.get("base_path", ""), path or "winequality-red.csv")
    columns = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
        "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates",
        "alcohol", "quality",
    ]
    df = pd.read_csv(path, header=None, names=columns)
    all_feature_cols = columns[:-1]
    var = df[all_feature_cols].var(axis=0)
    top_cols = var.sort_values(ascending=False).index[:n_features].tolist()
    top_cols = [c for c in all_feature_cols if c in top_cols]  # ripristina l'ordine originale
    X = df[top_cols].astype(float).reset_index(drop=True)
    y_codes = pd.Categorical(df["quality"]).codes
    y = pd.Series(y_codes.astype(int), name="target").reset_index(drop=True)
    return X, y, top_cols


def load_digits_data(n_features=8):
    """Digits (sklearn built-in). 1797 righe, 8 pixel (su 64) piu' variabili,
    10 classi -- il caso "molte classi"."""
    from sklearn.datasets import load_digits
    data = load_digits()
    var = data.data.var(axis=0)
    top_idx = np.argsort(-var)[:n_features]
    top_idx = sorted(top_idx)
    feature_cols = [f"pixel_{i}" for i in top_idx]
    X = pd.DataFrame(data.data[:, top_idx], columns=feature_cols)
    y = pd.Series(data.target.astype(int), name="target")
    return X, y, feature_cols


DATASETS = {
    "Glass": load_glass_data,
    "Wine Quality (red)": load_wine_quality_data,
    "Digits": load_digits_data,
}


# ---------------------------------------------------------------------------
# 2. Modello base. untrained=True genera rumore su TUTTE le n_classes (non
#    solo 0/1 come nel binario).
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, n_classes, n_estimators=50, max_depth=6,
                 untrained=False, random_state=0):
    if untrained:
        rng = np.random.default_rng(random_state)
        X_fit = rng.normal(size=X_train.shape)
        y_fit = rng.integers(0, n_classes, size=len(y_train))
        y_effective = pd.Series(y_fit, index=y_train.index, name=y_train.name)
    else:
        X_fit = X_train.values
        y_fit = y_train.values
        y_effective = y_train.copy()

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=1
    )
    model.fit(X_fit, y_fit)
    return model, y_effective


# ---------------------------------------------------------------------------
# 3. Le 6 removal strategy, GENERALIZZATE a N classi (identiche a
#    multiclass_experiments.py). __call__(x, mask, class_idx=None):
#      - class_idx=k   -> scalare P(classe=k)
#      - class_idx=None -> intero vettore di probabilita'
# ---------------------------------------------------------------------------

class RemovalStrategy:
    name = "base"

    def __call__(self, x, mask, class_idx=None):
        raise NotImplementedError


class DefaultValues(RemovalStrategy):
    name = "default_values"

    def __init__(self, model, X_train):
        self.model = model
        self.default = X_train.mean(axis=0).values

    def __call__(self, x, mask, class_idx=None):
        x_masked = x.copy()
        x_masked[~mask] = self.default[~mask]
        proba = self.model.predict_proba(x_masked.reshape(1, -1))[0]
        return proba if class_idx is None else proba[class_idx]


class Uniform(RemovalStrategy):
    name = "uniform"

    def __init__(self, model, X_train, n_samples=16):
        self.model = model
        self.lo = X_train.min(axis=0).values
        self.hi = X_train.max(axis=0).values
        self.n_samples = n_samples

    def __call__(self, x, mask, class_idx=None):
        samples = RNG.uniform(self.lo, self.hi, size=(self.n_samples, len(x)))
        samples[:, mask] = x[mask]
        mean_proba = self.model.predict_proba(samples).mean(axis=0)
        return mean_proba if class_idx is None else mean_proba[class_idx]


class ProductMarginals(RemovalStrategy):
    name = "product"

    def __init__(self, model, X_train, n_samples=16):
        self.model = model
        self.X_train = X_train.values
        self.n_samples = n_samples

    def __call__(self, x, mask, class_idx=None):
        idx = RNG.integers(0, len(self.X_train), size=(self.n_samples, len(x)))
        samples = self.X_train[idx, np.arange(len(x))]
        samples[:, mask] = x[mask]
        mean_proba = self.model.predict_proba(samples).mean(axis=0)
        return mean_proba if class_idx is None else mean_proba[class_idx]


class JointMarginal(RemovalStrategy):
    name = "marginal"

    def __init__(self, model, X_train, n_samples=16):
        self.model = model
        self.X_train = X_train.values
        self.n_samples = n_samples

    def __call__(self, x, mask, class_idx=None):
        idx = RNG.integers(0, len(self.X_train), size=self.n_samples)
        samples = self.X_train[idx].copy()
        samples[:, mask] = x[mask]
        mean_proba = self.model.predict_proba(samples).mean(axis=0)
        return mean_proba if class_idx is None else mean_proba[class_idx]


class Surrogate(RemovalStrategy):
    """Regressore MULTI-OUTPUT: un solo modello, allenato a predire l'intero
    vettore di probabilita' dato un input mascherato (non un surrogato per
    classe)."""

    name = "surrogate"

    def __init__(self, model, X_train, n_masks_per_row=4):
        self.model = model
        n, d = X_train.shape
        Xv = X_train.values
        base_proba = model.predict_proba(Xv)
        X_aug, y_aug = [], []
        for i in range(n):
            for _ in range(n_masks_per_row):
                mask = RNG.integers(0, 2, size=d).astype(bool)
                x_masked = Xv[i].copy()
                x_masked[~mask] = 0.0
                X_aug.append(np.concatenate([x_masked, mask.astype(float)]))
                y_aug.append(base_proba[i])
        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)

        self.surrogate = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0, n_jobs=1)
        self.surrogate.fit(X_aug, y_aug)

    def __call__(self, x, mask, class_idx=None):
        x_masked = x.copy()
        x_masked[~mask] = 0.0
        feat = np.concatenate([x_masked, mask.astype(float)]).reshape(1, -1)
        proba = self.surrogate.predict(feat)[0]
        return proba if class_idx is None else proba[class_idx]


class SeparateModels(RemovalStrategy):
    """Un modello per sottoinsieme di feature. Riallinea le colonne su
    m.classes_: un sottomodello puo' non aver visto tutte le N classi nel
    suo sottoinsieme di training (specie con molte classi e pochi dati)."""

    name = "separate_models"

    def __init__(self, X_train, y_train, n_classes):
        n, d = X_train.shape
        self.d = d
        self.n_classes = n_classes
        self.models = {}
        Xv = X_train.values
        yv = y_train.values
        all_subsets = [subset for r in range(d + 1) for subset in itertools.combinations(range(d), r)]
        for subset in tqdm(all_subsets, desc="  separate_models: training per subset", leave=False):
            if len(subset) == 0:
                self.models[subset] = None
                counts = np.bincount(yv, minlength=n_classes).astype(float)
                self.prior = counts / counts.sum()
                continue
            cols = list(subset)
            m = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=0, n_jobs=1)
            m.fit(Xv[:, cols], yv)
            self.models[subset] = m

    def __call__(self, x, mask, class_idx=None):
        subset = tuple(np.where(mask)[0])
        m = self.models[subset]
        if m is None:
            proba = self.prior
        else:
            raw = m.predict_proba(x[list(subset)].reshape(1, -1))[0]
            proba = np.zeros(self.n_classes)
            proba[m.classes_] = raw
        return proba if class_idx is None else proba[class_idx]


def bind_scalar(strategy, class_idx):
    return lambda x, mask: strategy(x, mask, class_idx=class_idx)


def bind_full(strategy):
    return lambda x, mask: strategy(x, mask, class_idx=None)


# ---------------------------------------------------------------------------
# 4. FAITHFULNESS (Bhatt et al., 2020). F qui e' gia' uno scalare vincolato
#    a una classe: il codice non deve sapere quante classi ci sono.
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
# 5. Ranking delle feature per Incremental Deletion / Addition. Identico al
#    binario -- opera su phi (gia' selezionato per la classe target), non
#    dipende da quante classi ci sono.
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


# ---------------------------------------------------------------------------
# 6. Versione ACCURACY, generalizzata a N classi: "corretto" = "la classe
#    target e' ancora l'argmax dell'intero vettore" (non piu' soglia 0.5).
# ---------------------------------------------------------------------------

def _is_correct_multiclass(F_full, x, mask, target_class):
    proba = F_full(x, mask)
    return 1.0 if int(np.argmax(proba)) == target_class else 0.0


def incremental_deletion_accuracy(F_full, x, target_class, order, n_features):
    full_mask = np.ones(n_features, dtype=bool)
    mask = full_mask.copy()
    vals = [_is_correct_multiclass(F_full, x, mask, target_class)]
    for feat_idx in order:
        mask = mask.copy()
        mask[feat_idx] = False
        vals.append(_is_correct_multiclass(F_full, x, mask, target_class))
    vals = np.array(vals)
    fraction = np.arange(n_features + 1) / n_features
    auc = float(_trapz(vals, fraction))
    return vals, fraction, auc


def incremental_addition_accuracy(F_full, x, target_class, order, n_features):
    empty_mask = np.zeros(n_features, dtype=bool)
    mask = empty_mask.copy()
    vals = [_is_correct_multiclass(F_full, x, mask, target_class)]
    for feat_idx in order:
        mask = mask.copy()
        mask[feat_idx] = True
        vals.append(_is_correct_multiclass(F_full, x, mask, target_class))
    vals = np.array(vals)
    fraction = np.arange(n_features + 1) / n_features
    auc = float(_trapz(vals, fraction))
    return vals, fraction, auc


def random_ranking_baseline(F_prob, F_full, x, n_features, n_random_rankings, target_class,
                             random_state=0, compute_accuracy=True):
    rng = np.random.default_rng(random_state)
    deletion_aucs, addition_aucs = [], []
    deletion_acc_aucs, addition_acc_aucs = [], []
    for _ in range(n_random_rankings):
        order = rng.permutation(n_features)
        _, _, d_auc = incremental_deletion(F_prob, x, order, n_features)
        _, _, a_auc = incremental_addition(F_prob, x, order, n_features)
        deletion_aucs.append(d_auc)
        addition_aucs.append(a_auc)
        if compute_accuracy:
            _, _, d_acc_auc = incremental_deletion_accuracy(F_full, x, target_class, order, n_features)
            _, _, a_acc_auc = incremental_addition_accuracy(F_full, x, target_class, order, n_features)
            deletion_acc_aucs.append(d_acc_auc)
            addition_acc_aucs.append(a_acc_auc)

    result = (float(np.mean(deletion_aucs)), float(np.mean(addition_aucs)))
    if compute_accuracy:
        result = result + (float(np.mean(deletion_acc_aucs)), float(np.mean(addition_acc_aucs)))
    return result


# ---------------------------------------------------------------------------
# 7. COMPLEXITY (Tseng, Shrikumar & Kundaje, NeurIPS 2020). Identica.
# ---------------------------------------------------------------------------

def complexity(phi, log_base="e", eps=1e-12):
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
# 8. Spiegazioni SHAP, KernelSHAP, LIME -- restituiscono TUTTE le classi in
#    un'unica chiamata, shape (n_istanze, n_feature, n_classi).
# ---------------------------------------------------------------------------

def compute_shap_values_all_classes(model, instances):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(instances.values)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    else:
        sv = np.asarray(sv)
    return sv


def compute_kernelshap_values_all_classes(model, X_train, instances):
    background = shap.sample(X_train, 50, random_state=0)
    explainer = shap.KernelExplainer(lambda data: model.predict_proba(data), background)
    sv = explainer.shap_values(instances.values, nsamples="auto", silent=True)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    else:
        sv = np.asarray(sv)
    return sv


def compute_lime_values_all_classes(model, X_train, instances, feature_cols, n_classes):
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_cols,
        class_names=[f"classe_{k}" for k in range(n_classes)],
        mode="classification",
        random_state=0,
    )
    n, d = len(instances), len(feature_cols)
    phi = np.zeros((n, d, n_classes))
    labels = tuple(range(n_classes))
    for i in tqdm(range(n), desc="  LIME: spiegazione per istanza", leave=False):
        exp = explainer.explain_instance(
            instances.iloc[i].values,
            lambda data: model.predict_proba(data),
            num_features=d,
            labels=labels,
        )
        for cls in labels:
            feat_weight = dict(exp.as_map()[cls])
            for j in range(d):
                phi[i, j, cls] = feat_weight.get(j, 0.0)
    return phi


def select_target_phi(phi_all, target_mode, y_instances, predicted_classes):
    """"true_class" o "predicted_class" -- "positive" non esiste qui."""
    n = phi_all.shape[0]
    out = np.zeros(phi_all.shape[:2])
    for i in range(n):
        if target_mode == "true_class":
            cls = int(y_instances.iloc[i])
        elif target_mode == "predicted_class":
            cls = int(predicted_classes[i])
        else:
            raise ValueError(f"explanation_target non riconosciuto: {target_mode!r}")
        out[i] = phi_all[i, :, cls]
    return out


def target_class_for_instance(target_mode, i, y_instances, predicted_classes):
    if target_mode == "true_class":
        return int(y_instances.iloc[i])
    elif target_mode == "predicted_class":
        return int(predicted_classes[i])
    else:
        raise ValueError(f"explanation_target non riconosciuto: {target_mode!r}")


def build_target_ranking_pairs(explanation_targets, ranking_modes, default_target, default_ranking):
    """Testa i due assi SEPARATAMENTE (non prodotto cartesiano completo)."""
    target_default = default_target if len(explanation_targets) > 1 else explanation_targets[0]
    ranking_default = default_ranking if len(ranking_modes) > 1 else ranking_modes[0]
    pairs = {(target_default, rm) for rm in ranking_modes}
    pairs |= {(tm, ranking_default) for tm in explanation_targets}
    return sorted(pairs)


# ---------------------------------------------------------------------------
# 9. Un esperimento completo per UN dataset x UNA model_variant.
# ---------------------------------------------------------------------------

def run_experiment(dataset_name, loader, model_variant_name="standard"):
    variant_cfg = CONFIG["model_variants"][model_variant_name]

    print("=" * 78)
    print(f"DATASET: {dataset_name}  |  MODEL VARIANT: {model_variant_name}")
    print("=" * 78)

    X, y, feature_cols = loader()
    n_features = len(feature_cols)
    n_classes = int(y.nunique())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    model, y_train_effective = train_model(
        X_train, y_train, n_classes,
        n_estimators=variant_cfg["n_estimators"],
        max_depth=variant_cfg["max_depth"],
        untrained=variant_cfg.get("untrained", False),
        random_state=0,
    )
    acc = model.score(X_test.values, y_test.values)
    print(f"{len(X)} righe, {n_features} feature, {n_classes} classi -> {feature_cols}")
    if variant_cfg.get("untrained", False):
        print("(RF allenato su RUMORE CASUALE: nessun segnale reale appreso)")
    print(f"Accuratezza RandomForest sul test set: {acc:.3f}\n")

    print("Costruzione delle strategie di rimozione...")
    strategies = {
        "default_values": DefaultValues(model, X_train),
        "uniform": Uniform(model, X_train),
        "product": ProductMarginals(model, X_train),
        "marginal": JointMarginal(model, X_train),
        "surrogate": Surrogate(model, X_train),
        "separate_models": SeparateModels(X_train, y_train_effective, n_classes),
    }
    print("Strategie pronte.\n")

    faith_cfg = CONFIG["faithfulness"]
    delins_cfg = CONFIG["deletion_insertion"]
    complexity_cfg = CONFIG["complexity"]

    def resolve_n(n_instances_cfg):
        return len(X_test) if n_instances_cfg is None else n_instances_cfg

    n_faith = resolve_n(faith_cfg["n_instances"])
    n_delins = resolve_n(delins_cfg["n_instances"])
    n_complexity = resolve_n(complexity_cfg["n_instances"])
    n_max = max(n_faith, n_delins, n_complexity)

    instances = X_test.iloc[:n_max].reset_index(drop=True)
    y_instances = y_test.iloc[:n_max].reset_index(drop=True)

    print("Generazione spiegazioni SHAP, KernelSHAP, LIME (tutte le classi in un colpo solo)...")
    shap_all = compute_shap_values_all_classes(model, instances)
    kernelshap_all = compute_kernelshap_values_all_classes(model, X_train, instances)
    lime_all = compute_lime_values_all_classes(model, X_train, instances, feature_cols, n_classes)
    explanations_all = {"SHAP": shap_all, "KernelSHAP": kernelshap_all, "LIME": lime_all}

    predicted_proba_full = model.predict_proba(instances.values)
    predicted_classes = np.argmax(predicted_proba_full, axis=1)
    true_classes = y_instances.values.astype(int)
    is_correct = (predicted_classes == true_classes)
    predicted_proba_max = predicted_proba_full.max(axis=1)

    def instance_meta(i):
        return {
            "predicted_class": int(predicted_classes[i]),
            "predicted_proba": float(predicted_proba_max[i]),
            "is_correct": bool(is_correct[i]),
        }

    explanation_targets = variant_cfg["explanation_targets"] or CONFIG["explanation_targets"]

    # =========================================================================
    # COMPLEXITY
    # =========================================================================
    complexity_rows = []
    for target_mode in explanation_targets:
        explanations = {
            name: select_target_phi(phi_all, target_mode, y_instances, predicted_classes)
            for name, phi_all in explanations_all.items()
        }
        for run in range(complexity_cfg["n_runs"]):
            for method_name, phi_matrix in explanations.items():
                for i in range(n_complexity):
                    val = complexity(phi_matrix[i], log_base=complexity_cfg["log_base"], eps=complexity_cfg["eps"])
                    complexity_rows.append({
                        "dataset": dataset_name, "model_variant": model_variant_name,
                        "explanation_target": target_mode, "run": run, "instance_id": i,
                        "explainer": method_name, "complexity": val, **instance_meta(i),
                    })

    print("--- Complexity (indipendente dalla removal strategy) ---")
    print(pd.DataFrame(complexity_rows).groupby(["explanation_target", "explainer"])[["complexity"]].mean().round(4))
    print()

    # =========================================================================
    # FAITHFULNESS
    # =========================================================================
    global RNG
    faithfulness_rows = []
    faithfulness_iter = [(mn, sn) for mn in explanations_all for sn in strategies]
    for target_mode in explanation_targets:
        explanations = {
            name: select_target_phi(phi_all, target_mode, y_instances, predicted_classes)
            for name, phi_all in explanations_all.items()
        }
        for run in range(faith_cfg["n_runs"]):
            RNG = np.random.default_rng(faith_cfg["random_state"] + run)
            for method_name, strat_name in tqdm(
                faithfulness_iter, desc=f"Faithfulness [{target_mode}] [run {run}]"
            ):
                strat = strategies[strat_name]
                phi_matrix = explanations[method_name]
                for i in range(n_faith):
                    x = instances.iloc[i].values
                    phi = phi_matrix[i]
                    tclass = target_class_for_instance(target_mode, i, y_instances, predicted_classes)
                    F = bind_scalar(strat, tclass)
                    score = faithfulness_correlation(F, phi, x, n_features, n_subsets=faith_cfg["n_subsets"])
                    faithfulness_rows.append({
                        "dataset": dataset_name, "model_variant": model_variant_name,
                        "explanation_target": target_mode, "run": run, "instance_id": i,
                        "explainer": method_name, "removal_strategy": strat_name,
                        "faithfulness": score, **instance_meta(i),
                    })

    print(f"\n--- Faithfulness correlation (n_instances={n_faith}) ---")
    fdf = pd.DataFrame(faithfulness_rows)
    print(fdf.groupby(["explanation_target", "explainer", "removal_strategy"])[["faithfulness"]].agg(
        ["mean", "std"]
    ).round(4))
    print()

    # =========================================================================
    # INCREMENTAL DELETION / ADDITION (+ versioni ACCURACY generalizzate).
    # I due assi (explanation_target, ranking_mode) testati separatamente.
    # =========================================================================
    curves_rows = []
    metrics_rows = []
    main_loop_iter = [(mn, sn) for mn in explanations_all for sn in strategies]
    ranking_modes = variant_cfg["ranking_modes"] or delins_cfg["ranking_modes"]
    target_ranking_pairs = build_target_ranking_pairs(
        explanation_targets, ranking_modes,
        CONFIG["explanation_target_default_for_ranking_test"],
        CONFIG["ranking_mode_default_for_target_test"],
    )

    explanations_by_target = {}
    for target_mode, ranking_mode in target_ranking_pairs:
        if target_mode not in explanations_by_target:
            explanations_by_target[target_mode] = {
                name: select_target_phi(phi_all, target_mode, y_instances, predicted_classes)
                for name, phi_all in explanations_all.items()
            }

    for target_mode, ranking_mode in target_ranking_pairs:
        explanations = explanations_by_target[target_mode]
        for run in range(delins_cfg["n_runs"]):
            RNG = np.random.default_rng(delins_cfg["random_state"] + run)
            random_baseline_cache = {}

            for method_name, strat_name in tqdm(
                main_loop_iter,
                desc=f"Deletion/Addition [{target_mode}/{ranking_mode}] [run {run}]"
            ):
                phi_matrix = explanations[method_name]
                strat = strategies[strat_name]
                for i in tqdm(range(n_delins), desc=f"  istanze [{method_name}/{strat_name}]", leave=False):
                    x = instances.iloc[i].values
                    phi = phi_matrix[i]
                    tclass = target_class_for_instance(target_mode, i, y_instances, predicted_classes)
                    F_prob = bind_scalar(strat, tclass)
                    F_full = bind_full(strat)

                    order = rank_features(phi, ranking_mode)

                    del_preds, del_fraction, del_auc = incremental_deletion(F_prob, x, order, n_features)
                    for step, (frac, pred) in enumerate(zip(del_fraction, del_preds)):
                        curves_rows.append({
                            "dataset": dataset_name, "model_variant": model_variant_name,
                            "explanation_target": target_mode, "ranking_mode": ranking_mode,
                            "run": run, "instance_id": i, "explainer": method_name,
                            "removal_strategy": strat_name, "metric": "deletion",
                            "step": step, "fraction_features": frac, "prediction": pred,
                        })

                    add_preds, add_fraction, add_auc = incremental_addition(F_prob, x, order, n_features)
                    for step, (frac, pred) in enumerate(zip(add_fraction, add_preds)):
                        curves_rows.append({
                            "dataset": dataset_name, "model_variant": model_variant_name,
                            "explanation_target": target_mode, "ranking_mode": ranking_mode,
                            "run": run, "instance_id": i, "explainer": method_name,
                            "removal_strategy": strat_name, "metric": "addition",
                            "step": step, "fraction_features": frac, "prediction": pred,
                        })

                    del_acc_vals, del_acc_fraction, del_acc_auc = incremental_deletion_accuracy(
                        F_full, x, tclass, order, n_features
                    )
                    for step, (frac, val) in enumerate(zip(del_acc_fraction, del_acc_vals)):
                        curves_rows.append({
                            "dataset": dataset_name, "model_variant": model_variant_name,
                            "explanation_target": target_mode, "ranking_mode": ranking_mode,
                            "run": run, "instance_id": i, "explainer": method_name,
                            "removal_strategy": strat_name, "metric": "deletion_accuracy",
                            "step": step, "fraction_features": frac, "prediction": val,
                        })

                    add_acc_vals, add_acc_fraction, add_acc_auc = incremental_addition_accuracy(
                        F_full, x, tclass, order, n_features
                    )
                    for step, (frac, val) in enumerate(zip(add_acc_fraction, add_acc_vals)):
                        curves_rows.append({
                            "dataset": dataset_name, "model_variant": model_variant_name,
                            "explanation_target": target_mode, "ranking_mode": ranking_mode,
                            "run": run, "instance_id": i, "explainer": method_name,
                            "removal_strategy": strat_name, "metric": "addition_accuracy",
                            "step": step, "fraction_features": frac, "prediction": val,
                        })

                    metrics_row = {
                        "dataset": dataset_name, "model_variant": model_variant_name,
                        "explanation_target": target_mode, "ranking_mode": ranking_mode,
                        "run": run, "instance_id": i, "explainer": method_name,
                        "removal_strategy": strat_name,
                        "deletion_auc": del_auc, "addition_auc": add_auc,
                        "deletion_accuracy_auc": del_acc_auc, "addition_accuracy_auc": add_acc_auc,
                        **instance_meta(i),
                    }

                    if delins_cfg["compute_random_baseline"]:
                        cache_key = (strat_name, i)
                        if cache_key not in random_baseline_cache:
                            random_baseline_cache[cache_key] = random_ranking_baseline(
                                F_prob, F_full, x, n_features,
                                n_random_rankings=delins_cfg["n_random_rankings"],
                                target_class=tclass,
                                random_state=delins_cfg["random_state"] + run * 1000 + i,
                                compute_accuracy=True,
                            )
                        rand_del_auc, rand_add_auc, rand_del_acc_auc, rand_add_acc_auc = (
                            random_baseline_cache[cache_key]
                        )
                        metrics_row["random_deletion_auc"] = rand_del_auc
                        metrics_row["random_addition_auc"] = rand_add_auc
                        metrics_row["deletion_auc_vs_random"] = del_auc - rand_del_auc
                        metrics_row["addition_auc_vs_random"] = add_auc - rand_add_auc
                        metrics_row["random_deletion_accuracy_auc"] = rand_del_acc_auc
                        metrics_row["random_addition_accuracy_auc"] = rand_add_acc_auc

                    metrics_rows.append(metrics_row)

    mdf = pd.DataFrame(metrics_rows)
    print(f"\n--- Deletion/Addition AUC (n_instances={n_delins}) ---")
    print(mdf.groupby(["explanation_target", "ranking_mode", "explainer", "removal_strategy"])[
        ["deletion_auc", "addition_auc", "deletion_accuracy_auc", "addition_accuracy_auc"]
    ].mean().round(4))
    print()

    return curves_rows, metrics_rows, faithfulness_rows, complexity_rows


# ---------------------------------------------------------------------------
# 10. main(): checkpoint/resume per (dataset, model_variant, explanation_target).
# ---------------------------------------------------------------------------

def _split_path(base_filename, variant_name, target_name):
    name, ext = base_filename.rsplit(".", 1)
    return f"{CONFIG['output_dir']}/{name}_{variant_name}_{target_name}.{ext}"


def _expected_targets_for_variant(variant_name):
    variant_cfg = CONFIG["model_variants"][variant_name]
    return variant_cfg["explanation_targets"] or CONFIG["explanation_targets"]


def _variant_already_done(variant_name):
    base_filenames = [
        CONFIG["curves_csv"], CONFIG["metrics_csv"],
        CONFIG["faithfulness_csv"], CONFIG["complexity_csv"],
    ]
    for target_name in _expected_targets_for_variant(variant_name):
        for base in base_filenames:
            if not os.path.exists(_split_path(base, variant_name, target_name)):
                return False
    return True


def _save_split_by_target(df, base_filename, variant_name, label):
    if len(df) == 0:
        return
    for target_name, sub_df in df.groupby("explanation_target"):
        path = _split_path(base_filename, variant_name, target_name)
        sub_df.to_csv(path, index=False)
        print(f"[{variant_name}/{target_name}] {label} salvate in: {path}  ({len(sub_df)} righe)")


def main():
    for variant_name in CONFIG["model_variants"]:
        if _variant_already_done(variant_name):
            print(f"[{variant_name}] gia' completata (tutti i file di output trovati) -- SALTATA.")
            continue

        variant_curves_rows = []
        variant_metrics_rows = []
        variant_faithfulness_rows = []
        variant_complexity_rows = []

        for dataset_name, loader in tqdm(list(DATASETS.items()), desc=f"DATASET [{variant_name}]"):
            curves_rows, metrics_rows, faithfulness_rows, complexity_rows = run_experiment(
                dataset_name, loader, model_variant_name=variant_name
            )
            variant_curves_rows.extend(curves_rows)
            variant_metrics_rows.extend(metrics_rows)
            variant_faithfulness_rows.extend(faithfulness_rows)
            variant_complexity_rows.extend(complexity_rows)

        curves_df = pd.DataFrame(variant_curves_rows)
        metrics_df = pd.DataFrame(variant_metrics_rows)
        faithfulness_df = pd.DataFrame(variant_faithfulness_rows)
        complexity_df = pd.DataFrame(variant_complexity_rows)

        print("=" * 78)
        _save_split_by_target(curves_df, CONFIG["curves_csv"], variant_name, "Curve complete")
        _save_split_by_target(metrics_df, CONFIG["metrics_csv"], variant_name, "Deletion/Addition AUC")
        _save_split_by_target(faithfulness_df, CONFIG["faithfulness_csv"], variant_name, "Faithfulness (raw)")
        _save_split_by_target(complexity_df, CONFIG["complexity_csv"], variant_name, "Complexity (raw)")


if __name__ == "__main__":
    main()