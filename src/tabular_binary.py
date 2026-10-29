"""
MODULO CORE (non eseguibile direttamente): contiene tutto il codice condiviso
degli esperimenti sulle tecniche di rimozione delle feature (Covert, Lundberg
& Lee, 2021 - "Explaining by Removing") sui 4 dataset tabellari:

    - Adult Income     (classificazione: reddito annuo > 50K)
    - COMPAS           (classificazione: recidiva a 2 anni)
    - German Credit    (classificazione: rischio di credito)
    - Breast Cancer    (classificazione: tumore maligno/benigno)

QUESTO FILE E' STATO SPEZZATO IN 4 RUNNER SOTTILI (uno per dataset):
run_adult_income.py, run_compas.py, run_german_credit.py,
run_breast_cancer.py -- ciascuno importa da qui e lancia SOLO il proprio
dataset. Motivo: Adult Income da solo puo' richiedere ~2 giorni di calcolo
(esperienza diretta su cluster), mentre gli altri 3 dataset sono molto piu'
rapidi. Tenendoli in un unico script/processo, un'interruzione su Adult
Income costringeva a rifare da capo anche il lavoro gia' completato sugli
altri dataset (il checkpoint/resume era per (model_variant, explanation_target),
non per dataset). Con un processo separato per dataset, ogni file scrive i
propri checkpoint indipendentemente: se Adult Income si interrompe, gli altri
3 dataset (gia' completati nei loro processi) non vengono minimamente toccati,
e rilanciando SOLO lo script di Adult Income si riprende dalla variante
interrotta.

Le SPIEGAZIONI sono generate con SHAP (TreeExplainer), KernelSHAP (KernelExplainer)
e LIME, tutte usando le impostazioni di DEFAULT delle rispettive librerie. Queste
spiegazioni sono fisse e non dipendono dalle 6 strategie di rimozione definite sotto.

Le 6 strategie di rimozione (default_values, uniform, product, marginal,
surrogate, separate_models) sono invece usate SOLO per calcolare la metrica
di valutazione (faithfulness): definiscono cosa significa "rimuovere" una
feature quando si misura il calo reale di predizione. In questo modo possiamo
osservare come la METRICA di faithfulness cambia a seconda della strategia
usata per valutare, a parita' di spiegazione (SHAP o LIME).

    1. default_values   -> feature mancanti sostituite con media/moda
    2. uniform           -> feature mancanti campionate da una uniforme sul dominio
    3. product            -> feature mancanti campionate dal prodotto delle marginali
                             (colonne indipendenti tra loro)
    4. marginal            -> feature mancanti campionate dalla marginale congiunta
                             (righe intere prese dal training set)
    5. surrogate          -> modello surrogato allenato a imitare f con feature mascherate
    6. separate_models     -> un modello RF diverso per ogni sottoinsieme di feature

FAITHFULNESS (Bhatt et al., 2020): si rimuovono sottoinsiemi casuali di feature
e si misura la correlazione tra la somma delle attribuzioni delle feature
rimosse e il calo reale della predizione causato dalla loro rimozione (calcolato
con la strategia in esame). Piu' alta e' la correlazione, piu' la spiegazione
riflette fedelmente il comportamento del modello secondo quella nozione di
rimozione.

INCREMENTAL DELETION / ADDITION (Petsiuk et al., 2018 - RISE) e la loro
versione ACCURACY (Kapishnikov et al., XRAI 2019): le feature vengono
rimosse/aggiunte una alla volta secondo il ranking di Φ, registrando la
predizione (o la correttezza rispetto alla vera label) ad ogni step.

COMPLEXITY (Tseng, Shrikumar & Kundaje, NeurIPS 2020): entropia di Shannon
della distribuzione |phi_i| normalizzata, non dipende dalla removal strategy.

NOTA: Infidelity e Sensitivity (Yeh et al., NeurIPS 2019) sono state RIMOSSE
da questo script. L'articolo originale le valida solo su dati immagine (MNIST,
CIFAR-10, ImageNet); non fornisce alcun esperimento ne' alcun default per
dati tabellari, quindi il loro utilizzo qui sarebbe stato un'estrapolazione
non validata dagli autori.

Per restare "molto semplice": poche feature (6), un modello RF piccolo, poche
istanze valutate, pochi sottoinsiemi per la stima Monte Carlo della metrica.
"""

import itertools
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import shap
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm

RNG = np.random.default_rng(0)

# Compatibilita' numpy 1.x / 2.x: trapz e' stato rinominato in trapezoid in numpy 2.0
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------------------
# 0. Configurazione centralizzata degli iperparametri (Incremental
#    Deletion/Addition, Complexity). Nessun valore hard-coded dentro le
#    funzioni: tutto passa da qui.
# ---------------------------------------------------------------------------

CONFIG = {
    "base_path": "data",  # directory populated by scripts/download_data.py

    "faithfulness": {
        "n_instances": None,  # None = usa tutte le istanze del test set; oppure un intero per limitarle
        "n_runs": 1,          # numero di run indipendenti (ciascuna ricampiona i sottoinsiemi casuali)
        "n_subsets": 20,      # numero di sottoinsiemi campionati per istanza per run
        "random_state": 0,
    },

    "deletion_insertion": {
        "n_instances": None,
        "n_runs": 1,          # numero di run indipendenti (rilevante per le strategie con campionamento interno)
        "ranking_modes": ["signed", "absolute"],  # TUTTI testati (non solo uno)
        "compute_random_baseline": True,
        "n_random_rankings": 5,
        "random_state": 0,
        # Versione "accuracy" (Accuracy Information Curve, Kapishnikov et al.,
        # XRAI 2019): ad ogni step si registra se la predizione (soglia sotto)
        # e' ancora corretta rispetto alla vera label, invece della probabilita'.
        "accuracy_threshold": 0.5,
    },

    "explanation_targets": ["positive", "true_class", "predicted_class"],
    # "positive":        spiega sempre la classe 1 (comportamento originale)
    # "true_class":      spiega la classe VERA dell'istanza (y_true)
    # "predicted_class":  spiega la classe PREDETTA dal modello (utile anche
    #                     per capire predizioni sbagliate)

    # Quando si esplorano PIU' valori su entrambi gli assi (variante "standard"),
    # NON si fa il prodotto cartesiano completo (3x3=9 combinazioni, costoso).
    # Si testano invece i due assi SEPARATAMENTE, tenendo l'altro asse fissato
    # su un default sensato:
    #   - si fa variare ranking_mode tenendo explanation_target fissato su...
    "explanation_target_default_for_ranking_test": "predicted_class",  # non "positive"
    #   - si fa variare explanation_target tenendo ranking_mode fissato su...
    "ranking_mode_default_for_target_test": "absolute",

    "model_variants": {
        # "standard": il modello usato finora in tutti gli esperimenti.
        #   explanation_targets/ranking_modes=None -> usa le liste globali sopra
        #   (esplorazione completa: 3 target x 3 ranking).
        "standard": {
            "model_type": "rf",
            "n_estimators": 50, "max_depth": 6,
            "explanation_targets": None,
            "ranking_modes": None,
        },
        # "shallow": stessa procedura, ma un RF a capacita' molto ridotta
        #   (pochi alberi, profondita' 1 -> praticamente degli stump). Testa
        #   se le metriche restano sensate quando il modello sottostante e'
        #   troppo semplice per il problema.
        "shallow": {
            "model_type": "rf",
            "n_estimators": 5, "max_depth": 1,
            "explanation_targets": ["predicted_class"],  # solo controllo di robustezza, un default sensato (non "positive")
            "ranking_modes": ["absolute"],
        },
        # "untrained": un vero RandomForestClassifier, stessa architettura di
        #   "standard", ma allenato su RUMORE CASUALE (feature e label
        #   indipendenti, generate da zero) invece che sui dati reali. Resta
        #   un RF vero e proprio -- compatibile con shap.TreeExplainer, a
        #   differenza di un DummyClassifier -- ma non ha appreso nulla sul
        #   problema reale: e' l'operativizzazione piu' letterale di "stesso
        #   modello, ma senza training" restando dentro l'API scikit-learn
        #   (un RF mai chiamato .fit() non risponde a predict_proba).
        "untrained": {
            "model_type": "rf",
            "n_estimators": 50, "max_depth": 6,
            "untrained": True,
            "explanation_targets": ["predicted_class"],
            "ranking_modes": ["absolute"],
        },
    },

    "complexity": {
        # Entropia di Shannon della distribuzione |phi_i| / sum(|phi|)
        # (Tseng, Shrikumar & Kundaje, NeurIPS 2020). Dipende solo da phi:
        # nessuna perturbazione, nessuna chiamata al modello o a F(x, mask).
        # n_runs incluso per uniformita' di interfaccia, ma essendo deterministica
        # (nessuna casualita' coinvolta) ogni run dara' lo stesso valore.
        "n_instances": None,
        "n_runs": 1,
        "log_base": "e",  # "e" (nats) oppure "2" (bit)
        "eps": 1e-12,
    },

    "output_dir": "results",
    "curves_csv": "removal_strategies_curves.csv",
    "metrics_csv": "removal_strategies_instance_metrics.csv",
    "faithfulness_csv": "removal_strategies_faithfulness_runs.csv",
    "complexity_csv": "removal_strategies_complexity_runs.csv",
}

# ---------------------------------------------------------------------------
# 1. Caricamento e preparazione dei 4 dataset. Per ognuno selezioniamo poche
#    feature semplici (numeriche o binarie), cosi' la strategia
#    "separate_models" resta trattabile (2^n_feature sottoinsiemi).
# ---------------------------------------------------------------------------

def load_compas(path=None):
    path = os.path.join(CONFIG["base_path"], path or "compas.csv")
    df = pd.read_csv(path)

    # Filtri standard usati da ProPublica per pulire i casi non affidabili
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    df["c_charge_degree"] = (df["c_charge_degree"] == "F").astype(int)  # 1=Felony, 0=Misdemeanor
    df["sex"] = (df["sex"] == "Male").astype(int)

    feature_cols = [
        "age",
        "priors_count",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "c_charge_degree",
        "sex",
    ]
    target_col = "two_year_recid"

    df = df[feature_cols + [target_col]].dropna()
    X = df[feature_cols].astype(float).reset_index(drop=True)
    y = df[target_col].astype(int).reset_index(drop=True)
    return X, y, feature_cols


def load_adult(path=None):
    """Adult Income (UCI/Census). Target: reddito annuo > 50K."""
    path = os.path.join(CONFIG["base_path"], path or "adult.csv")
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income",
    ]
    df = pd.read_csv(path, header=None, names=columns, na_values="?", skipinitialspace=True)
    df["sex"] = (df["sex"] == "Male").astype(int)
    df["income"] = (df["income"].str.strip(".") == ">50K").astype(int)

    feature_cols = [
        "age", "education_num", "capital_gain", "capital_loss",
        "hours_per_week", "sex",
    ]
    target_col = "income"

    df = df[feature_cols + [target_col]].dropna()
    X = df[feature_cols].astype(float).reset_index(drop=True)
    y = df[target_col].astype(int).reset_index(drop=True)
    return X, y, feature_cols


def load_german_credit(path=None):
    """German Credit (Statlog), formato UCI con attributi codificati (A11...).
    Target: 1 = rischio di credito cattivo (classe originale 2), 0 = buono."""
    path = os.path.join(CONFIG["base_path"], path or "german.csv")
    columns = [
        "checking_status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment_since", "installment_rate", "personal_status",
        "other_debtors", "residence_since", "property", "age", "other_installment_plans",
        "housing", "existing_credits", "job", "num_dependents", "telephone",
        "foreign_worker", "target",
    ]
    df = pd.read_csv(path, header=None, names=columns, sep=",")
    df["target"] = (df["target"] == 2).astype(int)  # 1 = "bad" credit risk

    feature_cols = [
        "duration", "credit_amount", "installment_rate", "residence_since",
        "age", "existing_credits", "num_dependents",
    ]
    target_col = "target"

    df = df[feature_cols + [target_col]].dropna()
    X = df[feature_cols].astype(float).reset_index(drop=True)
    y = df[target_col].astype(int).reset_index(drop=True)
    return X, y, feature_cols


def load_breast_cancer_data():
    """Breast Cancer Wisconsin (Diagnostic), incluso in scikit-learn.
    Target: 1 = maligno, 0 = benigno (nota: invertiamo la codifica di sklearn,
    dove 0=maligno/1=benigno, per leggibilita')."""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = 1 - data.target  # 1 = maligno

    feature_cols = [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean concavity",
    ]
    target_col = "target"

    df = df[feature_cols + [target_col]].dropna()
    X = df[feature_cols].astype(float).reset_index(drop=True)
    y = df[target_col].astype(int).reset_index(drop=True)
    return X, y, feature_cols


DATASETS = {
    "Adult Income": load_adult,
    "COMPAS": load_compas,
    "German Credit": load_german_credit,
    "Breast Cancer": load_breast_cancer_data,
}


# ---------------------------------------------------------------------------
# 2. Modello base: Random Forest
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, n_estimators=50, max_depth=6, untrained=False, random_state=0):
    """untrained=True: il RandomForest viene allenato su RUMORE CASUALE
    (feature e label generate da zero, indipendenti dal problema reale)
    invece che su (X_train, y_train). Resta un vero RF fittato -- compatibile
    con shap.TreeExplainer -- ma non ha appreso nulla sul problema reale.
    Restituisce anche le y_train "effettive" (quelle su cui si e' basato
    l'apprendimento, reali o rumore), perche' le removal strategy
    supervisionate (separate_models) devono restare coerenti con "quale
    funzione sta davvero spiegando" F."""
    if untrained:
        rng = np.random.default_rng(random_state)
        X_fit = rng.normal(size=X_train.shape)
        y_fit = rng.integers(0, 2, size=len(y_train))
        y_effective = pd.Series(y_fit, index=y_train.index, name=y_train.name)
    else:
        X_fit = X_train.values
        y_fit = y_train.values
        y_effective = y_train.copy()

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=1
    )
    # .values: evita il warning sklearn quando in seguito lo si interroga con array numpy
    model.fit(X_fit, y_fit)
    return model, y_effective


def f_predict(model, X):
    """Probabilita' predetta della classe positiva (colonna 1)."""
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# 3. Le 6 strategie di rimozione: ognuna implementa F(x, S) -> predizione,
#    dove S e' l'insieme (maschera booleana) delle feature "presenti".
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
        # media per continue, moda per binarie/intere: qui usiamo la media per
        # tutte le colonne numeriche (approssimazione semplice, come nel paper)
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


class Surrogate(RemovalStrategy):
    """Allena un modello surrogato g(x*mask, mask) a imitare f quando alcune
    feature sono mascherate (mascheramento casuale indipendente durante il
    training), come in L2X / REAL-X (Appendix E.3 del paper)."""

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


def RandomForestRegressorSafe():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0, n_jobs=1)


class SeparateModels(RemovalStrategy):
    """Allena un modello RF diverso per OGNI sottoinsieme di feature (fattibile
    solo perche' qui abbiamo poche feature: 2^6 = 64 sottoinsiemi)."""

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
# 4. Tecnica di sintesi comune a tutte le strategie: Shapley value stimato
#    con campionamento di permutazioni (Monte Carlo), come in IME/SAGE.
# ---------------------------------------------------------------------------

def shapley_values_mc(F, x, n_features, n_permutations=150):
    """Stima i valori di Shapley di F rispetto alle feature di x, campionando
    permutazioni casuali dell'ordine di inclusione delle feature."""
    phi = np.zeros(n_features)
    full_mask = np.ones(n_features, dtype=bool)
    empty_mask = np.zeros(n_features, dtype=bool)
    v_full = F(x, full_mask)
    v_empty = F(x, empty_mask)

    for _ in range(n_permutations):
        perm = RNG.permutation(n_features)
        mask = np.zeros(n_features, dtype=bool)
        prev_value = v_empty
        for feat_idx in perm:
            mask[feat_idx] = True
            new_value = F(x, mask)
            phi[feat_idx] += new_value - prev_value
            prev_value = new_value

    phi /= n_permutations
    return phi


# ---------------------------------------------------------------------------
# 4bis. Metrica di valutazione: faithfulness correlation (Bhatt et al., 2020).
#
# Idea: se le attribuzioni phi sono "fedeli" al modello, allora rimuovendo un
# sottoinsieme casuale S di feature, la somma delle attribuzioni di S dovrebbe
# correlare con il calo reale della predizione causato dalla loro rimozione.
# Usiamo la STESSA funzione F(x, mask) della strategia con cui e' stata
# generata la spiegazione, cosi' il confronto tra strategie e' end-to-end
# coerente: ogni strategia genera la spiegazione E definisce cosa significa
# "rimuovere" una feature per valutarla.
# ---------------------------------------------------------------------------

def faithfulness_correlation(F, phi, x, n_features, n_subsets=50, min_size=1):
    full_mask = np.ones(n_features, dtype=bool)
    v_full = F(x, full_mask)

    attributed_sums = []
    actual_drops = []
    for _ in range(n_subsets):
        k = RNG.integers(min_size, n_features)  # dimensione casuale del sottoinsieme rimosso
        S = RNG.choice(n_features, size=k, replace=False)

        mask = full_mask.copy()
        mask[S] = False  # rimuoviamo le feature in S
        v_removed = F(x, mask)

        attributed_sums.append(phi[S].sum())
        actual_drops.append(v_full - v_removed)

    attributed_sums = np.array(attributed_sums)
    actual_drops = np.array(actual_drops)

    if np.std(attributed_sums) == 0 or np.std(actual_drops) == 0:
        return np.nan
    return np.corrcoef(attributed_sums, actual_drops)[0, 1]


# ---------------------------------------------------------------------------
# 4bis-3. Ranking delle feature per Incremental Deletion / Addition.
#
#   - "absolute": ordina per |phi| decrescente (feature piu' influenti per
#                 magnitudine, a prescindere dal segno) - convenzione standard
#                 nei test di deletion/insertion (es. Petsiuk et al., RISE).
#   - "signed":   ordina per phi decrescente (dalla piu' positiva alla piu'
#                 negativa), preservando l'intero ordinamento con segno.
#   - "positive": isola le feature con contributo positivo (phi > 0) e le
#                 ordina in modo decrescente; le rimanenti (phi <= 0) sono
#                 accodate nell'ordine originale delle feature.
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
# 4bis-4. INCREMENTAL DELETION e INCREMENTAL ADDITION / INSERTION.
#
# Deletion: si parte dalla predizione con tutte le feature presenti e si
# rimuovono una alla volta, nell'ordine dato dal ranking, fino a rimuoverle
# tutte. Addition: procedura complementare, si parte da nessuna feature e si
# aggiungono una alla volta nello stesso ordine.
#
# Per un'istanza con d feature restituiamo tutti i d+1 valori della curva
# (non solo l'AUC), per poter confrontare le curve punto per punto tra le
# diverse removal strategy.
# ---------------------------------------------------------------------------

def incremental_deletion(F, x, order, n_features):
    full_mask = np.ones(n_features, dtype=bool)
    mask = full_mask.copy()
    preds = [F(x, mask)]  # step 0: nessuna feature rimossa
    for feat_idx in order:
        mask = mask.copy()
        mask[feat_idx] = False
        preds.append(F(x, mask))
    preds = np.array(preds)  # lunghezza d+1
    fraction = np.arange(n_features + 1) / n_features
    auc = float(_trapz(preds, fraction))
    return preds, fraction, auc


def incremental_addition(F, x, order, n_features):
    empty_mask = np.zeros(n_features, dtype=bool)
    mask = empty_mask.copy()
    preds = [F(x, mask)]  # step 0: nessuna feature presente
    for feat_idx in order:
        mask = mask.copy()
        mask[feat_idx] = True
        preds.append(F(x, mask))
    preds = np.array(preds)  # lunghezza d+1
    fraction = np.arange(n_features + 1) / n_features
    auc = float(_trapz(preds, fraction))
    return preds, fraction, auc


# ---------------------------------------------------------------------------
# 4bis-4bis. Versione ACCURACY di Incremental Deletion / Addition (Accuracy
# Information Curve, Kapishnikov et al., XRAI 2019).
#
# Identiche nella procedura (stesso F, stesso ranking) alle versioni a
# probabilita' sopra (che restano INVARIATE), ma ad ogni step si registra un
# indicatore 0/1 invece della probabilita' continua.
#
# COERENZA CON IL TARGET: F e' la STESSA funzione (gia' avvolta via
# bind_target_class) usata per le versioni a probabilita' -- quindi qui non
# si controlla piu' "la predizione coincide con la vera label y" in assoluto,
# ma "la classe TARGET (qualunque essa sia: positive/true_class/
# predicted_class) resta ancora quella favorita dal modello", cioe'
# F(x, mask) >= threshold. Per target="true_class" questo coincide
# esattamente con l'accuratezza classica (la classe target E' la vera
# label); per target="predicted_class" misura la STABILITA' della decisione
# originale; per target="positive" resta la versione fissa sulla classe 1.
# ---------------------------------------------------------------------------

def _is_correct(F, x, mask, threshold):
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
    casuali delle feature (indipendente dall'explainer, dipende solo dalla
    RemovalStrategy F e dall'istanza x). Se compute_accuracy=True, calcola
    anche la baseline delle versioni accuracy, riusando GLI STESSI ranking
    casuali (piu' efficiente e piu' corretto per il confronto).

    F: UNA SOLA funzione di rimozione, usata sia per le curve/AUC a
    probabilita' che per le versioni accuracy -- deve gia' restituire
    P(classe = target spiegato), es. via bind_target_class(). Nessuna
    distinzione F_prob/F_acc: la classe target e' usata ovunque, per piena
    coerenza con come e' stata generata la spiegazione."""
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
# 4bis-5. COMPLEXITY (Tseng, Shrikumar & Kundaje, NeurIPS 2020).
#
# A differenza di faithfulness / deletion / addition, NON usa la
# RemovalStrategy F(x, mask): dipende solo dal vettore phi (nessuna
# perturbazione, nessuna chiamata al modello). Per questo il suo valore e'
# lo stesso per tutte e 6 le removal strategy (viene calcolato una sola
# volta per ogni combinazione dataset x istanza x explainer e poi riportato
# su ogni riga).
# ---------------------------------------------------------------------------

def complexity(phi, log_base="e", eps=1e-12):
    """Entropia di Shannon della distribuzione normalizzata |phi_i| / sum(|phi|).
    Spiegazioni piu' "concentrate" su poche feature -> entropia bassa (piu'
    semplici); spiegazioni che spalmano l'importanza su tutte le feature ->
    entropia alta (piu' complesse)."""
    abs_phi = np.abs(phi)
    total = abs_phi.sum()
    if total <= eps:
        return 0.0
    p = abs_phi / total
    p = p[p > eps]  # solo termini con probabilita' positiva contribuiscono a p*log(p)
    entropy = -(p * np.log(p)).sum()
    if log_base == "2":
        entropy = entropy / np.log(2)
    return float(entropy)


# ---------------------------------------------------------------------------
# 4ter. Spiegazioni SHAP e LIME, generate con le impostazioni di DEFAULT delle
# rispettive librerie (nessuna delle 6 strategie custom entra nella loro
# generazione). Sono queste spiegazioni "chiavi in mano" che poi valuteremo
# con le 6 diverse strategie di rimozione come metrica di faithfulness.
# ---------------------------------------------------------------------------

def compute_shap_values(model, instances, n_features):
    """SHAP con il suo esplicatore di default per modelli ad albero
    (shap.TreeExplainer), nessun parametro custom. Restituisce le attribuzioni
    per ENTRAMBE le classi (shape n_istanze x n_features x 2), cosi' la stessa
    chiamata basta sia per il target "positive" che per "true_class"/
    "predicted_class" -- nessuna spiegazione viene rigenerata."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(instances.values)
    if isinstance(sv, list):  # sklearn RF binario: lista [classe0, classe1]
        sv = np.stack(sv, axis=-1)
    # altrimenti sv e' gia' (n_istanze, n_feature, n_classi)
    return sv  # shape (n_istanze, n_features, 2)


def compute_kernelshap_values(model, X_train, instances):
    """KernelSHAP (shap.KernelExplainer), il metodo model-agnostic generico
    di SHAP, usato con le sue impostazioni di default (nsamples="auto").
    Richiede comunque un dataset di background: usiamo un campione del
    training set, la scelta standard suggerita dalla libreria stessa.
    Spiega la funzione multi-output predict_proba (entrambe le classi in un
    'unica chiamata, non solo P(classe=1))."""
    background = shap.sample(X_train, 50, random_state=0)
    explainer = shap.KernelExplainer(lambda data: model.predict_proba(data), background)
    sv = explainer.shap_values(instances.values, nsamples="auto", silent=True)
    if isinstance(sv, list):  # alcune versioni di shap restituiscono una lista [classe0, classe1]
        sv = np.stack(sv, axis=-1)
    else:
        sv = np.asarray(sv)  # altre gia' (n_istanze, n_feature, n_classi)
    return sv  # shape (n_istanze, n_features, 2)


def compute_lime_values(model, X_train, instances, feature_cols):
    """LIME con LimeTabularExplainer di default (discretize_continuous=True,
    num_samples=5000 di default internamente). Richiede entrambe le label
    (0 e 1) in un'unica chiamata a explain_instance, cosi' non serve rigenerare
    la spiegazione per ciascun target di classe."""
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_cols,
        class_names=["classe_0", "classe_1"],
        mode="classification",
        random_state=0,
    )
    n, d = len(instances), len(feature_cols)
    phi = np.zeros((n, d, 2))
    for i in tqdm(range(n), desc="  LIME: spiegazione per istanza", leave=False):
        exp = explainer.explain_instance(
            instances.iloc[i].values,
            lambda data: model.predict_proba(data),
            num_features=d,
            labels=(0, 1),
        )
        for cls in (0, 1):
            feat_weight = dict(exp.as_map()[cls])  # {indice_feature: peso}
            for j in range(d):
                phi[i, j, cls] = feat_weight.get(j, 0.0)
    return phi  # shape (n_istanze, n_features, 2)


def select_target_phi(phi_both, target_mode, y_instances=None, predicted_classes=None):
    """Seleziona, per ciascuna istanza, quale delle due colonne di classe di
    phi_both (shape n x d x 2) usare come spiegazione, secondo il target:

      - "positive":         sempre la classe 1 (comportamento originale,
                             indipendente dalla predizione o dalla vera label).
      - "true_class":       la classe VERA dell'istanza (y_instances).
      - "predicted_class":  la classe PREDETTA dal modello per l'istanza
                             (predicted_classes) -- utile anche quando il
                             modello sbaglia, per capire "perche' ha deciso
                             cosi'", a differenza di "true_class" che spiega
                             "quali prove esistono per la risposta corretta".
    """
    n = phi_both.shape[0]
    if target_mode == "positive":
        return phi_both[:, :, 1].copy()
    elif target_mode == "true_class":
        out = np.zeros(phi_both.shape[:2])
        for i in range(n):
            out[i] = phi_both[i, :, int(y_instances.iloc[i])]
        return out
    elif target_mode == "predicted_class":
        out = np.zeros(phi_both.shape[:2])
        for i in range(n):
            out[i] = phi_both[i, :, int(predicted_classes[i])]
        return out
    else:
        raise ValueError(f"explanation_target non riconosciuto: {target_mode!r}")


def bind_target_class(strategy, target_class):
    """Restituisce un wrapper F(x, mask) -> P(classe = target_class), invece
    del comportamento originale delle 6 RemovalStrategy che restituiscono
    SEMPRE P(classe=1) (via f_predict). Per classificazione binaria vale
    l'identita' P(classe=0) = 1 - P(classe=1) (verificata anche a livello di
    singola feature per le attribuzioni SHAP: phi_classe0 = -phi_classe1),
    quindi non serve ri-addestrare o modificare le strategie supervisionate
    (surrogate, separate_models): basta invertire l'output quando si vuole
    spiegare la classe 0.

    QUESTO E' IL FIX del disallineamento descritto nel changelog: senza
    questo wrapper, Faithfulness/Deletion/Addition confrontavano le
    attribuzioni di predicted_class/true_class (che per le istanze con
    target=0 sono il negativo di quelle di classe 1) con un calo di
    predizione misurato SEMPRE su P(classe=1) -- da cui la correlazione
    sistematicamente negativa osservata per quei due target."""
    if target_class == 1:
        return strategy
    return lambda x, mask: 1.0 - strategy(x, mask)


def target_class_for_instance(target_mode, i, y_instances, predicted_classes):
    """Classe che l'esplicazione sta effettivamente spiegando per l'istanza i,
    secondo l'explanation_target in uso."""
    if target_mode == "positive":
        return 1
    elif target_mode == "true_class":
        return int(y_instances.iloc[i])
    elif target_mode == "predicted_class":
        return int(predicted_classes[i])
    else:
        raise ValueError(f"explanation_target non riconosciuto: {target_mode!r}")


def build_target_ranking_pairs(explanation_targets, ranking_modes, default_target, default_ranking):
    """Costruisce le coppie (explanation_target, ranking_mode) da valutare per
    Incremental Deletion/Addition, testando i due assi SEPARATAMENTE invece
    del prodotto cartesiano completo:
      - un target fissato su default_target, al variare di ranking_mode
      - un ranking_mode fissato su default_ranking, al variare di explanation_target
    Se una delle due liste ha un solo elemento (es. varianti "shallow"/
    "untrained"), quell'unico valore FA DA PROPRIO default -- il risultato e'
    semplicemente quella singola coppia, nessuna espansione indebita."""
    target_default = default_target if len(explanation_targets) > 1 else explanation_targets[0]
    ranking_default = default_ranking if len(ranking_modes) > 1 else ranking_modes[0]

    pairs = {(target_default, rm) for rm in ranking_modes}
    pairs |= {(tm, ranking_default) for tm in explanation_targets}
    return sorted(pairs)


# ---------------------------------------------------------------------------
# 5. Un esperimento completo per UN dataset: allena il modello, costruisce le
#    6 strategie, genera le spiegazioni SHAP/KernelSHAP/LIME (default) e ne
#    valuta la faithfulness al variare della strategia di rimozione.
# ---------------------------------------------------------------------------

def run_experiment(dataset_name, loader, model_variant_name="standard"):
    variant_cfg = CONFIG["model_variants"][model_variant_name]

    print("=" * 78)
    print(f"DATASET: {dataset_name}  |  MODEL VARIANT: {model_variant_name}")
    print("=" * 78)

    X, y, feature_cols = loader()
    n_features = len(feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    model, y_train_effective = train_model(
        X_train, y_train,
        n_estimators=variant_cfg["n_estimators"],
        max_depth=variant_cfg["max_depth"],
        untrained=variant_cfg.get("untrained", False),
        random_state=0,
    )
    acc = model.score(X_test.values, y_test.values)
    print(f"{len(X)} righe, {n_features} feature -> {feature_cols}")
    if variant_cfg.get("untrained", False):
        print("(RF allenato su RUMORE CASUALE: nessun segnale reale appreso dal problema)")
    print(f"Accuratezza RandomForest sul test set (contro le VERE label): {acc:.3f}\n")

    print("Costruzione delle strategie di rimozione (puo' richiedere qualche secondo)...")
    strategies = {
        "default_values": DefaultValues(model, X_train),
        "uniform": Uniform(model, X_train),
        "product": ProductMarginals(model, X_train),
        "marginal": JointMarginal(model, X_train),
        "surrogate": Surrogate(model, X_train),
        # separate_models usa y_train_effective (mescolate per la variante
        # "untrained"), cosi' resta coerente con "quale funzione sta
        # davvero spiegando" F -- niente segnale reale infiltrato di nascosto.
        "separate_models": SeparateModels(X_train, y_train_effective),
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

    # Le spiegazioni si generano UNA sola volta per le n_max istanze necessarie;
    # ogni metrica poi usa solo il prefisso di cui ha bisogno (stesso ordine di
    # istanze per tutte, dato che derivano tutte dallo stesso X_test).
    instances = X_test.iloc[:n_max].reset_index(drop=True)
    y_instances = y_test.iloc[:n_max].reset_index(drop=True)

    print("Generazione spiegazioni SHAP (TreeExplainer, default), KernelSHAP")
    print("(KernelExplainer, default) e LIME (default) -- entrambe le classi,")
    print("cosi' non serve rigenerarle per ogni explanation_target...")
    shap_phi_both = compute_shap_values(model, instances, n_features)
    kernelshap_phi_both = compute_kernelshap_values(model, X_train, instances)
    lime_phi_both = compute_lime_values(model, X_train, instances, feature_cols)
    explanations_both = {
        "SHAP": shap_phi_both,
        "KernelSHAP": kernelshap_phi_both,
        "LIME": lime_phi_both,
    }

    # Colonne derivate (predicted_class, predicted_proba, is_correct), calcolate
    # UNA volta, indipendenti da explainer/strategia/target -- servono sia per
    # costruire il target "predicted_class" sia per stratificare i risultati a
    # posteriori (es. "solo istanze mal classificate") senza ricalcolare nulla.
    predicted_proba = f_predict(model, instances.values)
    predicted_classes = (predicted_proba >= 0.5).astype(int)
    true_classes = y_instances.values.astype(int)
    is_correct = (predicted_classes == true_classes)

    def instance_meta(i):
        return {
            "predicted_class": int(predicted_classes[i]),
            "predicted_proba": float(predicted_proba[i]),
            "is_correct": bool(is_correct[i]),
        }

    explanation_targets = variant_cfg["explanation_targets"] or CONFIG["explanation_targets"]

    # =========================================================================
    # COMPLEXITY: n_instances/n_runs propri. Deterministica (nessuna casualita'),
    # quindi tutte le run danno lo stesso valore. Dipende dal target di
    # spiegazione (phi cambia), quindi si ripete per ciascun explanation_target.
    # =========================================================================
    complexity_rows = []
    for target_mode in explanation_targets:
        explanations = {
            name: select_target_phi(phi_both, target_mode, y_instances, predicted_classes)
            for name, phi_both in explanations_both.items()
        }
        for run in range(complexity_cfg["n_runs"]):
            for method_name, phi_matrix in explanations.items():
                for i in range(n_complexity):
                    phi = phi_matrix[i]
                    val = complexity(phi, log_base=complexity_cfg["log_base"], eps=complexity_cfg["eps"])
                    complexity_rows.append({
                        "dataset": dataset_name,
                        "model_variant": model_variant_name,
                        "explanation_target": target_mode,
                        "run": run,
                        "instance_id": i,
                        "explainer": method_name,
                        "complexity": val,
                        **instance_meta(i),
                    })

    print("--- Complexity (indipendente dalla removal strategy, dipende dall'explanation_target) ---")
    complexity_df = pd.DataFrame(complexity_rows)
    print(complexity_df.groupby(["explanation_target", "explainer"])[["complexity"]].mean().round(4))
    print()

    # =========================================================================
    # FAITHFULNESS: n_instances/n_runs propri, ripetuta per ciascun
    # explanation_target. Il subset_size resta campionato casualmente ad ogni
    # prova (INVARIATO); ogni run riazzera il generatore casuale globale con
    # un seed diverso, cosi' le run sono riproducibili ma tra loro distinte.
    # =========================================================================
    global RNG
    faithfulness_rows = []
    faithfulness_iter = [
        (method_name, strat_name)
        for method_name in explanations_both
        for strat_name in strategies
    ]
    for target_mode in explanation_targets:
        explanations = {
            name: select_target_phi(phi_both, target_mode, y_instances, predicted_classes)
            for name, phi_both in explanations_both.items()
        }
        for run in range(faith_cfg["n_runs"]):
            RNG = np.random.default_rng(faith_cfg["random_state"] + run)
            for method_name, strat_name in tqdm(
                faithfulness_iter,
                desc=f"Faithfulness [{target_mode}] [run {run}] [explainer x strategia]"
            ):
                strat = strategies[strat_name]
                phi_matrix = explanations[method_name]
                for i in range(n_faith):
                    x = instances.iloc[i].values
                    phi = phi_matrix[i]
                    tclass = target_class_for_instance(target_mode, i, y_instances, predicted_classes)
                    F_prob = bind_target_class(strat, tclass)
                    score = faithfulness_correlation(F_prob, phi, x, n_features, n_subsets=faith_cfg["n_subsets"])
                    faithfulness_rows.append({
                        "dataset": dataset_name,
                        "model_variant": model_variant_name,
                        "explanation_target": target_mode,
                        "run": run,
                        "instance_id": i,
                        "explainer": method_name,
                        "removal_strategy": strat_name,
                        "faithfulness": score,
                        **instance_meta(i),
                    })

    print(f"\n--- Faithfulness correlation (n_instances={n_faith}, n_runs={faith_cfg['n_runs']}, "
          f"n_subsets={faith_cfg['n_subsets']}) ---")
    faithfulness_df = pd.DataFrame(faithfulness_rows)
    faith_summary = faithfulness_df.groupby(["explanation_target", "explainer", "removal_strategy"])[
        ["faithfulness"]
    ].agg(["mean", "std"]).round(4)
    print(faith_summary)
    print()

    # =========================================================================
    # INCREMENTAL DELETION / ADDITION (+ versioni ACCURACY): n_instances/n_runs
    # propri. I due assi (explanation_target, ranking_mode) sono testati
    # SEPARATAMENTE (non prodotto cartesiano completo): quando si fa variare
    # ranking_mode, l'explanation_target resta fissato sul default
    # (predicted_class, non positive); quando si fa variare explanation_target,
    # ranking_mode resta fissato sul suo default (absolute). Ogni run riazzera
    # il generatore casuale globale, cosi' le strategie con campionamento
    # interno (uniform/product/marginal) danno valori F(x, mask) leggermente
    # diversi run per run.
    # =========================================================================
    curves_rows = []   # output A: curve complete (formato long)
    metrics_rows = []  # output B: metriche instance-level aggregate (deletion/addition)

    main_loop_iter = [
        (method_name, strat_name)
        for method_name in explanations_both
        for strat_name in strategies
    ]
    ranking_modes = variant_cfg["ranking_modes"] or delins_cfg["ranking_modes"]
    target_ranking_pairs = build_target_ranking_pairs(
        explanation_targets, ranking_modes,
        CONFIG["explanation_target_default_for_ranking_test"],
        CONFIG["ranking_mode_default_for_target_test"],
    )

    # Le spiegazioni-per-target si ricalcolano solo una volta per target_mode
    # (anche se compare in piu' coppie), non ad ogni coppia.
    explanations_by_target = {}
    for target_mode, ranking_mode in target_ranking_pairs:
        if target_mode not in explanations_by_target:
            explanations_by_target[target_mode] = {
                name: select_target_phi(phi_both, target_mode, y_instances, predicted_classes)
                for name, phi_both in explanations_both.items()
            }

    for target_mode, ranking_mode in target_ranking_pairs:
        explanations = explanations_by_target[target_mode]
        if True:
            for run in range(delins_cfg["n_runs"]):
                RNG = np.random.default_rng(delins_cfg["random_state"] + run)
                # Il random-ranking baseline non dipende dall'explainer ne' dal
                # ranking_mode (e' un ranking CASUALE per definizione): lo
                # calcoliamo una volta per (istanza, strategia) all'interno di
                # questa run e lo replichiamo su tutte le righe explainer.
                random_baseline_cache = {}

                for method_name, strat_name in tqdm(
                    main_loop_iter,
                    desc=f"Deletion/Addition [{target_mode}/{ranking_mode}] [run {run}] [explainer x strategia]"
                ):
                    phi_matrix = explanations[method_name]
                    strat = strategies[strat_name]
                    for i in tqdm(
                        range(n_delins), desc=f"  istanze [{method_name}/{strat_name}]", leave=False
                    ):
                        x = instances.iloc[i].values
                        phi = phi_matrix[i]

                        # Classe che questa spiegazione sta effettivamente spiegando per
                        # l'istanza i (dipende da target_mode). F e' UNA SOLA funzione,
                        # usata sia per le versioni a probabilita' che per quelle accuracy:
                        # piena coerenza con la classe che Φ sta spiegando.
                        tclass = target_class_for_instance(target_mode, i, y_instances, predicted_classes)
                        F = bind_target_class(strat, tclass)

                        # Ranking condiviso da Deletion e Addition (e dalle loro versioni accuracy)
                        order = rank_features(phi, ranking_mode)

                        # Incremental Deletion, versione a PROBABILITA' (curva completa + AUC)
                        del_preds, del_fraction, del_auc = incremental_deletion(F, x, order, n_features)
                        for step, (frac, pred) in enumerate(zip(del_fraction, del_preds)):
                            curves_rows.append({
                                "dataset": dataset_name,
                        "model_variant": model_variant_name,
                                "explanation_target": target_mode,
                                "ranking_mode": ranking_mode,
                                "run": run,
                                "instance_id": i,
                                "explainer": method_name,
                                "removal_strategy": strat_name,
                                "metric": "deletion",
                                "step": step,
                                "fraction_features": frac,
                                "prediction": pred,
                            })

                        # Incremental Addition, versione a PROBABILITA' (curva completa + AUC)
                        add_preds, add_fraction, add_auc = incremental_addition(F, x, order, n_features)
                        for step, (frac, pred) in enumerate(zip(add_fraction, add_preds)):
                            curves_rows.append({
                                "dataset": dataset_name,
                        "model_variant": model_variant_name,
                                "explanation_target": target_mode,
                                "ranking_mode": ranking_mode,
                                "run": run,
                                "instance_id": i,
                                "explainer": method_name,
                                "removal_strategy": strat_name,
                                "metric": "addition",
                                "step": step,
                                "fraction_features": frac,
                                "prediction": pred,
                            })

                        # Incremental Deletion, versione ACCURACY: stesso ranking, ma
                        # ad ogni step si registra 0/1 ("la classe target e' ancora quella
                        # favorita?", coerente con F usata per la versione a probabilita')
                        del_acc_vals, del_acc_fraction, del_acc_auc = incremental_deletion_accuracy(
                            F, x, order, n_features, threshold=delins_cfg["accuracy_threshold"]
                        )
                        for step, (frac, val) in enumerate(zip(del_acc_fraction, del_acc_vals)):
                            curves_rows.append({
                                "dataset": dataset_name,
                        "model_variant": model_variant_name,
                                "explanation_target": target_mode,
                                "ranking_mode": ranking_mode,
                                "run": run,
                                "instance_id": i,
                                "explainer": method_name,
                                "removal_strategy": strat_name,
                                "metric": "deletion_accuracy",
                                "step": step,
                                "fraction_features": frac,
                                "prediction": val,
                            })

                        # Incremental Addition, versione ACCURACY
                        add_acc_vals, add_acc_fraction, add_acc_auc = incremental_addition_accuracy(
                            F, x, order, n_features, threshold=delins_cfg["accuracy_threshold"]
                        )
                        for step, (frac, val) in enumerate(zip(add_acc_fraction, add_acc_vals)):
                            curves_rows.append({
                                "dataset": dataset_name,
                        "model_variant": model_variant_name,
                                "explanation_target": target_mode,
                                "ranking_mode": ranking_mode,
                                "run": run,
                                "instance_id": i,
                                "explainer": method_name,
                                "removal_strategy": strat_name,
                                "metric": "addition_accuracy",
                                "step": step,
                                "fraction_features": frac,
                                "prediction": val,
                            })

                        metrics_row = {
                            "dataset": dataset_name,
                        "model_variant": model_variant_name,
                            "explanation_target": target_mode,
                            "ranking_mode": ranking_mode,
                            "run": run,
                            "instance_id": i,
                            "explainer": method_name,
                            "removal_strategy": strat_name,
                            "deletion_auc": del_auc,
                            "addition_auc": add_auc,
                            "deletion_accuracy_auc": del_acc_auc,
                            "addition_accuracy_auc": add_acc_auc,
                            **instance_meta(i),
                        }

                        # Random-ranking baseline (opzionale) -- calcolata sia per le
                        # versioni a probabilita' che per quelle accuracy, riusando
                        # GLI STESSI ranking casuali internamente
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
                            rand_del_auc, rand_add_auc, rand_del_acc_auc, rand_add_acc_auc = (
                                random_baseline_cache[cache_key]
                            )
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
    print(f"ranking_modes={ranking_modes}\n")
    summary = metrics_df.groupby(["explanation_target", "ranking_mode", "explainer", "removal_strategy"])[
        ["deletion_auc", "addition_auc"]
    ].mean().round(4)
    print(summary)
    print()

    print(f"--- Deletion/Addition Accuracy AUC (versione AIC, soglia={delins_cfg['accuracy_threshold']}) ---")
    acc_summary = metrics_df.groupby(["explanation_target", "ranking_mode", "explainer", "removal_strategy"])[
        ["deletion_accuracy_auc", "addition_accuracy_auc"]
    ].mean().round(4)
    print(acc_summary)
    print()

    return curves_rows, metrics_rows, faithfulness_rows, complexity_rows


def _split_path(base_filename, dataset_slug, variant_name, target_name):
    """removal_strategies_curves.csv -> removal_strategies_curves_adult_income_shallow_positive.csv
    Include il dataset nel nome file: necessario ora che ogni dataset gira in
    un file/processo separato (altrimenti due dataset lanciati in parallelo
    si sovrascriverebbero a vicenda)."""
    name, ext = base_filename.rsplit(".", 1)
    return f"{CONFIG['output_dir']}/{name}_{dataset_slug}_{variant_name}_{target_name}.{ext}"


def _save_split_by_target(df, base_filename, dataset_slug, variant_name, label):
    for target_name, sub_df in df.groupby("explanation_target"):
        path = _split_path(base_filename, dataset_slug, variant_name, target_name)
        sub_df.to_csv(path, index=False)
        print(f"[{dataset_slug}/{variant_name}/{target_name}] {label} salvate in: {path}  ({len(sub_df)} righe)")


def _expected_targets_for_variant(variant_name):
    variant_cfg = CONFIG["model_variants"][variant_name]
    return variant_cfg["explanation_targets"] or CONFIG["explanation_targets"]


def _variant_already_done(dataset_slug, variant_name):
    """True se TUTTI i file attesi per questo dataset+variante (4 tipi di
    output x ogni suo explanation_target) esistono gia' su disco.
    Granularita' per (DATASET, VARIANTE): dato che ogni dataset gira nel suo
    file/processo separato, questo equivale in pratica a un checkpoint per
    dataset -- se un file grande (es. Adult Income) si interrompe a meta',
    gli altri dataset (gia' completati nei loro processi separati) non sono
    minimamente toccati."""
    base_filenames = [
        CONFIG["curves_csv"], CONFIG["metrics_csv"],
        CONFIG["faithfulness_csv"], CONFIG["complexity_csv"],
    ]
    for target_name in _expected_targets_for_variant(variant_name):
        for base in base_filenames:
            if not os.path.exists(_split_path(base, dataset_slug, variant_name, target_name)):
                return False
    return True


def run_all_variants_for_dataset(dataset_name, loader, dataset_slug=None):
    """Punto di ingresso per i runner sottili (uno per dataset): esegue TUTTE
    le model_variants (standard/shallow/untrained) per UN SOLO dataset, con
    checkpoint/resume per (dataset, variante, target). Se il file/processo si
    interrompe a meta', rilanciando SOLO QUESTO SCRIPT si riprende dalla
    variante interrotta -- gli altri dataset, girati in script separati, non
    vengono minimamente toccati."""
    dataset_slug = dataset_slug or dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    for variant_name in CONFIG["model_variants"]:
        if _variant_already_done(dataset_slug, variant_name):
            print(f"[{dataset_slug}/{variant_name}] gia' completata (tutti i file di output trovati) -- SALTATA.")
            continue

        curves_rows, metrics_rows, faithfulness_rows, complexity_rows = run_experiment(
            dataset_name, loader, model_variant_name=variant_name
        )

        curves_df = pd.DataFrame(curves_rows)
        metrics_df = pd.DataFrame(metrics_rows)
        faithfulness_df = pd.DataFrame(faithfulness_rows)
        complexity_df = pd.DataFrame(complexity_rows)

        print("=" * 78)
        _save_split_by_target(curves_df, CONFIG["curves_csv"], dataset_slug, variant_name, "Curve complete")
        _save_split_by_target(metrics_df, CONFIG["metrics_csv"], dataset_slug, variant_name, "Deletion/Addition AUC")
        _save_split_by_target(faithfulness_df, CONFIG["faithfulness_csv"], dataset_slug, variant_name, "Faithfulness (raw)")
        _save_split_by_target(complexity_df, CONFIG["complexity_csv"], dataset_slug, variant_name, "Complexity (raw)")
