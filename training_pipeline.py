"""
training_pipeline.py

Complete, reproducible ML pipeline that outputs a **report** (Markdown by default,
optional console or PDF summary) covering:

- EDA visualizations and insights
- Selected features and preprocessing steps
- Hyperparameter tuning results
- Model comparison table (train/test metrics)
- Calibration curves, ROC-AUC curves, Brier scores
- SHAP summary plots (bar + beeswarm)
- PSI results and drift analysis

Usage:
    python training_pipeline.py
    # or:
    python training_pipeline.py --data-path data.csv --artifacts-dir artifacts
    # optional formats:
    python training_pipeline.py --report-format console
    python training_pipeline.py --report-format pdf

Reproducibility:
- Fixed RNG seeds (Python, NumPy, sklearn CV)
- All dependencies/version info recorded in the report
- Deterministic preprocessing with train-fitted artifacts

Style:
- Type hints, docstrings, grouped imports (ruff/PEP 8 friendly)
"""

from __future__ import annotations

# ------------------------------- stdlib -------------------------------- #
import argparse
import json
import logging
import random
import re
import unicodedata
import warnings
from importlib import metadata
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

# ------------------------------ third-party --------------------------- #
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------ constants ----------------------------- #
RANDOM_STATE: int = 42
TARGET_COL: str = "Bankrupt?"

# --------------------------------------------------------------------- #
#                          Utility & helpers                            #
# --------------------------------------------------------------------- #


def set_global_seed(seed: int) -> None:
    """Set global RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def configure_logging(log_dir: Path) -> None:
    """Write logs to a file and to the console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,  # prevent duplicate handlers on re-run
    )


def ensure_dirs(dirs: Iterable[Path]) -> None:
    """Ensure a list of directories exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str, maxlen: int = 120) -> str:
    """
    Create a filesystem-safe filename from an arbitrary string.
    - Normalize accents to ASCII
    - Replace non [A-Za-z0-9._-] with '_'
    - Collapse repeats and trim leading/trailing dots/underscores
    """
    norm = unicodedata.normalize("NFKD", str(name))
    ascii_name = norm.encode("ascii", "ignore").decode("ascii", "ignore")
    ascii_name = re.sub(r"[^\w.\-]", "_", ascii_name)
    ascii_name = re.sub(r"__+", "_", ascii_name).strip("._")
    return (ascii_name or "feature")[:maxlen]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load CSV and validate target column and non-empty data."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Data not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found.")
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    return df


def package_versions(pkgs: List[str]) -> Dict[str, str]:
    """Return installed versions for a list of packages; 'n/a' if missing."""
    versions: Dict[str, str] = {}
    for p in pkgs:
        try:
            versions[p] = metadata.version(p)
        except Exception:
            versions[p] = "n/a"
    return versions


# --------------------------------------------------------------------- #
#                               EDA & PSI                               #
# --------------------------------------------------------------------- #


def plot_class_imbalance(y: pd.Series, out_path: Path) -> Mapping[str, float]:
    """Save a class distribution bar plot and return basic stats."""
    counts = y.value_counts().sort_index()
    ratios = counts / counts.sum()
    stats = {
        "pos_rate_pct": 100.0 * float(ratios.get(1, 0.0)),
        "neg_rate_pct": 100.0 * float(ratios.get(0, 0.0)),
        "n_pos": int(counts.get(1, 0)),
        "n_neg": int(counts.get(0, 0)),
        "n_total": int(counts.sum()),
    }
    logging.info(
        "Class distribution -> 0: %d (%.2f%%), 1: %d (%.2f%%)",
        stats["n_neg"],
        stats["neg_rate_pct"],
        stats["n_pos"],
        stats["pos_rate_pct"],
    )
    plt.figure(figsize=(5, 4))
    counts.plot(kind="bar")
    plt.title("Class Distribution: Bankrupt (1) vs Non-Bankrupt (0)")
    plt.xlabel("Bankruptcy Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return stats


def plot_corr_subset(
    X: pd.DataFrame,
    out_path: Path,
    n_cols: int = 12,
    high_corr_threshold: float = 0.90,
) -> List[Tuple[str, str, float]]:
    """
    Save a compact correlation heatmap (first N numeric features).
    Return top highly correlated feature pairs as insights.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    subset = numeric_cols[:n_cols] if len(numeric_cols) > n_cols else numeric_cols
    corr = X[subset].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Correlation Heatmap (first {len(subset)} features)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Extract top correlated pairs from *all* numeric columns for insights
    corr_full = X.select_dtypes(include=[np.number]).corr().abs()
    upper = corr_full.where(np.triu(np.ones_like(corr_full, dtype=bool), k=1))
    pairs: List[Tuple[str, str, float]] = []
    for col in upper.columns:
        for row in upper.index:
            val = float(upper.loc[row, col])
            if not np.isnan(val) and val >= high_corr_threshold:
                pairs.append((row, col, val))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:10]


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split to preserve class ratios."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def _hist_proportions(
    a: np.ndarray,
    bins: np.ndarray,
    min_prop: float = 1e-6,
) -> np.ndarray:
    """Normalized histogram bin proportions with a small non-zero floor."""
    counts, _ = np.histogram(a, bins=bins)
    total = counts.sum()
    if total == 0:
        props = np.zeros_like(counts, dtype=float)
        if props.size > 0:
            props[0] = 1.0
        return props
    props = counts.astype(float) / float(total)
    props = np.where(props == 0.0, min_prop, props)
    return props / props.sum()


def calculate_psi(
    train: pd.DataFrame,
    test: pd.DataFrame,
    bins: int = 10,
    min_prop: float = 1e-6,
) -> pd.DataFrame:
    """
    Calculate Population Stability Index (PSI) per numeric feature using
    quantile bins defined on the training data.
    """
    results: List[Dict[str, float]] = []
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        tr = train[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        te = test[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if tr.size == 0 or te.size == 0:
            continue
        edges = np.unique(np.quantile(tr, np.linspace(0.0, 1.0, bins + 1)))
        if edges.size < 2:
            results.append({"feature": col, "psi": 0.0})
            continue
        p = _hist_proportions(tr, edges, min_prop=min_prop)
        q = _hist_proportions(te, edges, min_prop=min_prop)
        psi = float(np.sum((p - q) * np.log(p / q)))
        results.append({"feature": col, "psi": psi})

    return (
        pd.DataFrame(results).sort_values("psi", ascending=False).reset_index(drop=True)
        if results
        else pd.DataFrame(columns=["feature", "psi"])
    )


def plot_psi_bar(psi_df: pd.DataFrame, out_path: Path, top_n: int = 30) -> None:
    """Bar chart of features sorted by PSI (top N)."""
    if psi_df.empty:
        return
    df = psi_df.head(top_n)
    plt.figure(figsize=(10, max(4, 0.25 * len(df))))
    sns.barplot(x="psi", y="feature", data=df)
    plt.axvline(0.10, linestyle="--", label="0.10 (monitor)")
    plt.axvline(0.25, linestyle="--", label="0.25 (investigate)")
    plt.title(f"Top {len(df)} PSI Features (Train vs Test)")
    plt.xlabel("PSI")
    plt.ylabel("Feature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_top_drift_histograms(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    psi_df: pd.DataFrame,
    out_dir: Path,
    top_k: int = 6,
    bins: int = 30,
) -> None:
    """
    For the top-k drifted features by PSI, save overlaid histograms (train vs test).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if psi_df.empty:
        return

    for feature in psi_df["feature"].head(top_k):
        if feature not in X_train.columns or feature not in X_test.columns:
            continue

        plt.figure(figsize=(6, 4))
        sns.histplot(X_train[feature].dropna(), bins=bins, stat="density", alpha=0.5, label="Train")
        sns.histplot(X_test[feature].dropna(), bins=bins, stat="density", alpha=0.5, label="Test")
        plt.title(f"Train vs Test — {feature}")
        plt.xlabel(str(feature))
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        fname = f"psi_hist_{safe_filename(feature)}.png"
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()


# --------------------------------------------------------------------- #
#                         Preprocessing & Select                        #
# --------------------------------------------------------------------- #


class IQRWinsorizer:
    """
    Clip outliers per numeric column using the IQR rule learned on the train set:
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
    """

    def __init__(self) -> None:
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame) -> "IQRWinsorizer":
        """Learn caps on numeric columns from X."""
        num_cols = X.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            q1, q3 = np.nanpercentile(X[c].to_numpy(), [25, 75])
            iqr = q3 - q1
            self.bounds_[c] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned caps to X."""
        Xc = X.copy()
        for c, (lower, upper) in self.bounds_.items():
            if c in Xc.columns:
                Xc[c] = Xc[c].clip(lower=lower, upper=upper)
        return Xc

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on X then transform X."""
        return self.fit(X).transform(X)


def correlation_filter(X: pd.DataFrame, threshold: float = 0.90) -> List[str]:
    """Remove highly correlated columns (keep one from each correlated group)."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
    to_drop = {col for col in upper.columns if (upper[col] > threshold).any()}
    return [c for c in X.columns if c not in to_drop]


def fit_preprocessing_and_select(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    List[str],
    SimpleImputer,
    IQRWinsorizer,
    StandardScaler,
]:
    """
    Impute (median), winsorize outliers, remove heavy correlations, and scale features.

    Returns:
        Xtr_sel_winz, Xte_sel_winz, Xtr_scaled, Xte_scaled, selected,
        imputer, winz, scaler
    """
    # 1) Impute numeric missing values based on train
    imputer = SimpleImputer(strategy="median")
    Xtr_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    Xte_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    # 2) Correlation filter learned on train (post-imputation)
    selected = correlation_filter(Xtr_imp, threshold=0.90)
    Xtr_sel = Xtr_imp[selected].copy()
    Xte_sel = Xte_imp[selected].copy()

    # 3) Winsorize outliers (caps learned on train)
    winz = IQRWinsorizer().fit(Xtr_sel)
    Xtr_sel_winz = winz.transform(Xtr_sel)
    Xte_sel_winz = winz.transform(Xte_sel)

    # 4) Standard scaling (train parameters)
    scaler = StandardScaler().fit(Xtr_sel_winz)
    Xtr_scaled = pd.DataFrame(
        scaler.transform(Xtr_sel_winz),
        columns=Xtr_sel_winz.columns,
        index=Xtr_sel_winz.index,
    )
    Xte_scaled = pd.DataFrame(
        scaler.transform(Xte_sel_winz),
        columns=Xte_sel_winz.columns,
        index=Xte_sel_winz.index,
    )

    return (
        Xtr_sel_winz,
        Xte_sel_winz,
        Xtr_scaled,
        Xte_scaled,
        selected,
        imputer,
        winz,
        scaler,
    )


def save_preproc_artifacts(
    out_dir: Path,
    selected: List[str],
    imputer: SimpleImputer,
    winz: IQRWinsorizer,
    scaler: StandardScaler,
    Xtr_sel_winz: pd.DataFrame,
    Xte_sel_winz: pd.DataFrame,
    Xtr_scaled: pd.DataFrame,
    Xte_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Persist selected features, transformers, and ready-to-train CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.Series(selected, name="feature").to_csv(out_dir / "selected_features.csv", index=False)

    bounds_df = pd.DataFrame.from_dict(
        {k: {"lower": v[0], "upper": v[1]} for k, v in winz.bounds_.items()},
        orient="index",
    ).reset_index(names="feature")
    bounds_df.to_csv(out_dir / "winsor_bounds.csv", index=False)

    joblib.dump(imputer, out_dir / "imputer.joblib")
    joblib.dump(winz, out_dir / "winsorizer.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")

    Xtr_sel_winz.to_csv(out_dir / "X_train_selected_winsorized.csv", index=False)
    Xte_sel_winz.to_csv(out_dir / "X_test_selected_winsorized.csv", index=False)
    Xtr_scaled.to_csv(out_dir / "X_train_selected_winsorized_scaled.csv", index=False)
    Xte_scaled.to_csv(out_dir / "X_test_selected_winsorized_scaled.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)


# --------------------------------------------------------------------- #
#                          Modeling, tuning, eval                       #
# --------------------------------------------------------------------- #


def pos_weight(y: pd.Series) -> float:
    """Return scale_pos_weight = (neg / pos) for XGBoost."""
    p = float(y.sum())
    n = float(len(y) - y.sum())
    return max(n / p, 1.0) if p > 0 else 1.0


def cv_splitter(n_splits: int = 5) -> StratifiedKFold:
    """Stratified K-Fold with fixed seed."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


class TuningSummary(Tuple[str, float, Dict[str, Any]]):
    """Typing helper for tuning summaries: (model_name, best_score, best_params)."""


def tune_and_fit_models(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    class_ratio_weight: float,
) -> Tuple[Dict[str, object], List[TuningSummary]]:
    """
    Randomized search for Logistic Regression, Random Forest, and (if available) XGBoost.
    Returns:
        models: dict of fitted best estimators
        tuning_summaries: list of (name, best_cv_ap, best_params)
    """
    models: Dict[str, object] = {}
    summaries: List[TuningSummary] = []

    # Logistic Regression
    lr = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    lr_grid = {"C": np.logspace(-3, 2, 20), "penalty": ["l1", "l2"]}
    lr_search = RandomizedSearchCV(
        lr,
        lr_grid,
        n_iter=20,
        scoring="average_precision",
        cv=cv_splitter(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    lr_search.fit(X_train_scaled, y_train)
    models["logreg"] = lr_search.best_estimator_
    summaries.append(("logreg", float(lr_search.best_score_), lr_search.best_params_))

    # Random Forest
    rf = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_grid = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [None, 6, 10, 14],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf_search = RandomizedSearchCV(
        rf,
        rf_grid,
        n_iter=25,
        scoring="average_precision",
        cv=cv_splitter(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    rf_search.fit(X_train_scaled, y_train)
    models["rf"] = rf_search.best_estimator_
    summaries.append(("rf", float(rf_search.best_score_), rf_search.best_params_))

    # XGBoost (optional)
    try:
        from xgboost import XGBClassifier  # type: ignore

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=6,
            reg_alpha=0.0,
            reg_lambda=1.0,
            scale_pos_weight=class_ratio_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": np.linspace(0.02, 0.2, 10),
            "subsample": np.linspace(0.6, 1.0, 5),
            "colsample_bytree": np.linspace(0.6, 1.0, 5),
            "reg_alpha": [0.0, 0.01, 0.1, 0.5],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "n_estimators": [300, 500, 700, 900],
        }
        xgb_search = RandomizedSearchCV(
            xgb,
            xgb_grid,
            n_iter=30,
            scoring="average_precision",
            cv=cv_splitter(),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        xgb_search.fit(X_train_scaled, y_train)
        models["xgb"] = xgb_search.best_estimator_
        summaries.append(("xgb", float(xgb_search.best_score_), xgb_search.best_params_))
    except Exception as exc:  # pragma: no cover - optional
        logging.warning("Skipping XGBoost (not installed or failed): %s", exc)

    return models, summaries


def predict_proba_safe(model: object, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a classifier."""
    proba = getattr(model, "predict_proba", None)
    if proba is not None:
        return proba(X)[:, 1]
    decision = getattr(model, "decision_function", None)
    if decision is not None:
        z = decision(X)
        return 1.0 / (1.0 + np.exp(-z))
    raise AttributeError("Model has neither predict_proba nor decision_function.")


def evaluate_model(
    name: str,
    model: object,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    out_dir: Path,
) -> Dict[str, float]:
    """Compute metrics and save ROC, Calibration, and PR plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    p_tr = predict_proba_safe(model, X_tr)
    p_te = predict_proba_safe(model, X_te)

    roc_tr = roc_auc_score(y_tr, p_tr)
    roc_te = roc_auc_score(y_te, p_te)

    pr_tr = average_precision_score(y_tr, p_tr)
    pr_te = average_precision_score(y_te, p_te)

    f1_tr = f1_score(y_tr, (p_tr >= 0.5).astype(int))
    f1_te = f1_score(y_te, (p_te >= 0.5).astype(int))

    brier_tr = brier_score_loss(y_tr, p_tr)
    brier_te = brier_score_loss(y_te, p_te)

    # ROC
    fpr_tr, tpr_tr, _ = roc_curve(y_tr, p_tr)
    fpr_te, tpr_te, _ = roc_curve(y_te, p_te)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_tr, tpr_tr, label=f"Train ROC (AUC={roc_tr:.3f})")
    plt.plot(fpr_te, tpr_te, label=f"Test ROC (AUC={roc_te:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_roc.png", dpi=150)
    plt.close()

    # Calibration
    frac_pos_tr, mean_pred_tr = calibration_curve(y_tr, p_tr, n_bins=10, strategy="quantile")
    frac_pos_te, mean_pred_te = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred_tr, frac_pos_tr, marker="o", label=f"Train (Brier={brier_tr:.3f})")
    plt.plot(mean_pred_te, frac_pos_te, marker="o", label=f"Test (Brier={brier_te:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_calibration.png", dpi=150)
    plt.close()

    # Precision–Recall
    prec_tr, rec_tr, _ = precision_recall_curve(y_tr, p_tr)
    prec_te, rec_te, _ = precision_recall_curve(y_te, p_te)
    plt.figure(figsize=(6, 5))
    plt.plot(rec_tr, prec_tr, label=f"Train PR (AP={pr_tr:.3f})")
    plt.plot(rec_te, prec_te, label=f"Test PR (AP={pr_te:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_pr.png", dpi=150)
    plt.close()

    return {
        "model": name,
        "roc_auc_train": float(roc_tr),
        "roc_auc_test": float(roc_te),
        "pr_auc_train": float(pr_tr),
        "pr_auc_test": float(pr_te),
        "f1_train": float(f1_tr),
        "f1_test": float(f1_te),
        "brier_train": float(brier_tr),
        "brier_test": float(brier_te),
    }


def save_models(models: Dict[str, object], out_dir: Path) -> None:
    """Persist fitted models to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, mdl in models.items():
        joblib.dump(mdl, out_dir / f"{name}.joblib")


def _to_jsonable(obj: Any) -> Any:
    """Convert common non-JSON-serializable types to JSON-friendly forms."""
    try:
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_best_params(models: Dict[str, object], out_path: Path) -> Path:
    """Serialize model hyperparameters to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Dict[str, Any]] = {}
    for name, mdl in models.items():
        params: Dict[str, Any] = {}
        if hasattr(mdl, "get_params"):
            try:
                raw = mdl.get_params()  # type: ignore[attr-defined]
            except Exception:
                raw = {}
            params = _to_jsonable(raw)
        payload[name] = {"class": mdl.__class__.__name__, "params": params}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------- #
#                                 SHAP                                  #
# --------------------------------------------------------------------- #


def shap_summary_best_model(
    best_name: str,
    best_model: object,
    X_for_shap: pd.DataFrame,
    out_dir: Path,
    sample_n: int = 200,
) -> None:
    """
    Compute and save SHAP summary bar & beeswarm plots for the best model.
    """
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - optional
        logging.warning("Skipping SHAP (package not installed): %s", exc)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(sample_n, len(X_for_shap))
    if n <= 1:
        logging.warning("Not enough samples for SHAP; skipping.")
        return
    X_sample = X_for_shap.sample(n=n, random_state=RANDOM_STATE)

    # Build an appropriate explainer
    name = best_model.__class__.__name__.lower()
    try:
        if "randomforest" in name or "xgb" in name:
            explainer = shap.TreeExplainer(best_model)
        elif "logisticregression" in name:
            explainer = shap.LinearExplainer(best_model, X_sample, feature_perturbation="interventional")  # type: ignore[arg-type]
        else:
            explainer = shap.Explainer(best_model, X_sample)
    except Exception:
        explainer = shap.Explainer(best_model, X_sample)

    # Silence SHAP's FutureWarning about NumPy global RNG seeding (keeps logs clean)
    warnings.filterwarnings(
        "ignore",
        message="The NumPy global RNG was seeded by calling `np.random.seed`",
        category=FutureWarning,
        module="shap",
    )

    shap_values = explainer(X_sample)

    # Bar summary
    try:
        shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(out_dir / f"{best_name}_shap_summary_bar.png", dpi=150)
        plt.close()
    except Exception as exc:
        logging.warning("Failed to save SHAP bar plot: %s", exc)

    # Beeswarm summary
    try:
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"{best_name}_shap_summary_beeswarm.png", dpi=150)
        plt.close()
    except Exception as exc:
        logging.warning("Failed to save SHAP beeswarm plot: %s", exc)


# --------------------------------------------------------------------- #
#                              Reporting                                #
# --------------------------------------------------------------------- #


def _format_pairs(pairs: List[Tuple[str, str, float]]) -> str:
    """Format correlated pairs for Markdown."""
    if not pairs:
        return "- No feature pairs exceeded the correlation threshold.\n"
    lines = []
    for a, b, r in pairs:
        lines.append(f"- **{a}** ↔ **{b}**: |r| = {r:.2f}")
    return "\n".join(lines) + "\n"


def write_markdown_report(
    artifacts_dir: Path,
    eda_dir: Path,
    preproc_dir: Path,
    models_dir: Path,
    eval_dir: Path,
    drift_dir: Path,
    shap_dir: Path,
    df_shape: Tuple[int, int],
    class_stats: Mapping[str, float],
    selected_features: List[str],
    high_corr_pairs: List[Tuple[str, str, float]],
    tuning_summaries: List[TuningSummary],
) -> Path:
    """
    Write a Markdown report with RELATIVE asset links (portable on GitHub / ZIP).
    """
    report_path = artifacts_dir / "report.md"

    def _rel(p: Path) -> Path:
        try:
            return p.relative_to(artifacts_dir)
        except ValueError:
            return p

    def _embed(p: Path) -> str:
        return _rel(p).as_posix()

    # Paths
    class_plot = eda_dir / "class_imbalance.png"
    corr_plot = eda_dir / "corr_subset_heatmap.png"

    psi_csv = drift_dir / "psi_results.csv"
    psi_bar = drift_dir / "psi_bar.png"
    psi_hist_dir = drift_dir / "psi_hists"

    cmp_csv = eval_dir / "model_comparison.csv"
    params_json = models_dir / "best_params.json"

    shap_bar = next(iter(shap_dir.glob("*_shap_summary_bar.png")), None)
    shap_beeswarm = next(iter(shap_dir.glob("*_shap_summary_beeswarm.png")), None)

    # Dependencies
    versions = package_versions(
        ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "xgboost", "shap"]
    )

    # Tuning summary table (Markdown)
    if tuning_summaries:
        tune_lines = ["| Model | Best CV PR-AUC | Best Params |", "|------:|---------------:|-------------|"]
        for name, score, params in tuning_summaries:
            tune_lines.append(
                f"| {name} | {score:.4f} | `{json.dumps(params)}` |"
            )
        tuning_md = "\n".join(tune_lines) + "\n"
    else:
        tuning_md = "_(No hyperparameter tuning details available)_\n"

    # Selected features (show top 20)
    top_feats = ", ".join(map(str, selected_features[:20])) if selected_features else "(none)"

    # Build report
    lines: List[str] = []
    lines.append("# Training Pipeline Report\n")
    lines.append("## Overview\n")
    lines.append(
        dedent(
            f"""
            - Rows × Columns: **{df_shape[0]} × {df_shape[1]}**
            - Target positive rate: **{class_stats['pos_rate_pct']:.2f}%** (1s={class_stats['n_pos']}, 0s={class_stats['n_neg']})
            - Random seed: **{RANDOM_STATE}**
            """
        ).strip()
        + "\n"
    )

    lines.append("\n## EDA (Visualizations & Insights)\n")
    if class_plot.exists():
        lines.append(f"![Class Balance]({_embed(class_plot)})\n")
    if corr_plot.exists():
        lines.append(f"![Correlation (subset)]({_embed(corr_plot)})\n")
    lines.append("**Highly correlated pairs (|r| ≥ 0.90):**\n")
    lines.append(_format_pairs(high_corr_pairs))

    lines.append("## Preprocessing & Selected Features\n")
    lines.append(
        "- Train-median imputation → correlation filter (|r|>0.90) → IQR winsorization → StandardScaler.\n"
        f"- Selected features: **{len(selected_features)}** (top 20 shown): {top_feats}\n"
        f"- Artifacts: `{_embed(preproc_dir / 'selected_features.csv')}`, "
        f"`{_embed(preproc_dir / 'winsor_bounds.csv')}`, scalers/imputers in `preproc/`.\n"
    )

    lines.append("## Hyperparameter Tuning (RandomizedSearchCV, CV=5, scoring=PR-AUC)\n")
    lines.append(tuning_md)
    lines.append(f"Full best-params JSON: `{_embed(params_json)}`\n")

    lines.append("## Model Evaluation & Comparison\n")
    lines.append(
        "- Metrics: ROC-AUC, PR-AUC, F1@0.5, Brier (train & test).  \n"
        f"- Comparison table (CSV): `{_embed(cmp_csv)}`.\n"
        "- Curves per model saved in `evaluation/`: `*_roc.png`, `*_pr.png`, `*_calibration.png`.\n"
    )

    lines.append("## Calibration, ROC & Brier\n")
    lines.append(
        "- **Calibration**: compare predicted probability vs observed frequency (lower Brier is better).  \n"
        "- **ROC-AUC**: threshold-insensitive separability.  \n"
        "- **Precision–Recall**: recommended for imbalanced targets.\n"
    )

    lines.append("## SHAP (Best Model)\n")
    if shap_bar and shap_beeswarm and shap_bar.exists() and shap_beeswarm.exists():
        lines.append(f"![SHAP Bar]({_embed(shap_bar)})\n")
        lines.append(f"![SHAP Beeswarm]({_embed(shap_beeswarm)})\n")
    else:
        lines.append("- SHAP plots not generated (package missing or no plots found).\n")

    lines.append("## Population Stability Index (PSI)\n")
    lines.append(
        f"- PSI results CSV: `{_embed(psi_csv)}`.  \n"
        f"- Bar chart: `{_embed(psi_bar)}`; histograms in `{_embed(psi_hist_dir)}/`.  \n"
        "- Reference thresholds: 0.10 (monitor), 0.25 (investigate).\n"
    )

    lines.append("## Reproducibility & Dependencies\n")
    lines.append("- Deterministic seed usage (Python/NumPy/sklearn CV).")
    lines.append("- Key packages:")
    for pkg, ver in versions.items():
        lines.append(f"  - {pkg}: {ver}")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Report written -> %s", report_path)
    return report_path


def write_console_report(
    df_shape: Tuple[int, int],
    class_stats: Mapping[str, float],
    selected_features: List[str],
    tuning_summaries: List[TuningSummary],
    comparison_csv: Path,
    psi_csv: Path,
) -> None:
    """Print a concise summary to the console."""
    print("\n=== Training Pipeline Report (Console) ===")
    print(f"Rows × Columns: {df_shape[0]} × {df_shape[1]}")
    print(
        f"Class balance: 1s={class_stats['n_pos']} ({class_stats['pos_rate_pct']:.2f}%), "
        f"0s={class_stats['n_neg']} ({class_stats['neg_rate_pct']:.2f}%)"
    )
    print(f"Selected features: {len(selected_features)} (top 10): {selected_features[:10]}")
    print("\nHyperparameter tuning (best CV PR-AUC):")
    if tuning_summaries:
        for name, score, params in tuning_summaries:
            print(f"- {name}: {score:.4f}, params: {params}")
    else:
        print("- (no tuning details)")
    if comparison_csv.exists():
        print(f"\nModel comparison CSV: {comparison_csv}")
        try:
            df = pd.read_csv(comparison_csv)
            print(df.to_string(index=False))
        except Exception:
            pass
    if psi_csv.exists():
        print(f"\nPSI results CSV: {psi_csv}")
        try:
            df = pd.read_csv(psi_csv).head(10)
            print("Top PSI features:")
            print(df.to_string(index=False))
        except Exception:
            pass
    print("=========================================\n")


def write_pdf_report(markdown_report: Path, out_pdf: Path) -> Optional[Path]:
    """
    Very lightweight PDF summary generator.
    Requires 'reportlab'. If unavailable, return None (caller can fall back).
    Note: This embeds the text content of the Markdown report, not images.
    """
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore
        from reportlab.lib.units import cm  # type: ignore
    except Exception:
        logging.warning("reportlab not installed; skipping PDF export.")
        return None

    text = markdown_report.read_text(encoding="utf-8")
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    width, height = A4
    left_margin = 2 * cm
    top = height - 2 * cm
    line_height = 0.5 * cm

    c.setFont("Helvetica", 10)
    y = top
    for line in text.splitlines():
        # naive wrap: split long lines
        while len(line) > 110:
            c.drawString(left_margin, y, line[:110])
            line = line[110:]
            y -= line_height
            if y < 2 * cm:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = top
        c.drawString(left_margin, y, line)
        y -= line_height
        if y < 2 * cm:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = top

    c.save()
    logging.info("PDF report written -> %s", out_pdf)
    return out_pdf


# --------------------------------------------------------------------- #
#                                 Main                                  #
# --------------------------------------------------------------------- #


def main() -> None:
    """Run the whole pipeline and generate a report."""
    parser = argparse.ArgumentParser(description="End-to-End Pipeline with Report (Markdown/Console/PDF)")
    parser.add_argument("--data-path", type=str, default="data.csv", help="Path to dataset CSV.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to store artifacts.")
    parser.add_argument(
        "--report-format",
        type=str,
        choices=["markdown", "console", "pdf"],
        default="markdown",
        help="Output report format (default: markdown).",
    )
    parser.add_argument("--corr-heatmap-cols", type=int, default=12, help="Columns shown in correlation heatmap.")
    parser.add_argument("--psi-bins", type=int, default=10, help="Bins for PSI computation.")
    args = parser.parse_args()

    # Resolve paths relative to script
    script_dir = Path(__file__).resolve().parent
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (script_dir / data_path).resolve()
    artifacts_dir = (script_dir / args.artifacts_dir).resolve()

    # Subdirs
    eda_dir = artifacts_dir / "eda"
    drift_dir = artifacts_dir / "drift"
    preproc_dir = artifacts_dir / "preproc"
    models_dir = artifacts_dir / "models"
    eval_dir = artifacts_dir / "evaluation"
    shap_dir = artifacts_dir / "shap"
    logs_dir = script_dir / "logs"

    ensure_dirs([artifacts_dir, eda_dir, drift_dir, preproc_dir, models_dir, eval_dir, shap_dir, logs_dir])
    configure_logging(logs_dir)
    set_global_seed(RANDOM_STATE)

    # Step 1 — Load + EDA + Split + PSI
    df = load_dataset(data_path)
    logging.info("Data shape: %s | Loaded from: %s", df.shape, data_path)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    class_stats = plot_class_imbalance(y, eda_dir / "class_imbalance.png")
    high_corr_pairs = plot_corr_subset(
        X, eda_dir / "corr_subset_heatmap.png", n_cols=args.corr_heatmap_cols, high_corr_threshold=0.90
    )

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
    logging.info(
        "Train: %s | Test: %s | Train 1s: %.2f%% | Test 1s: %.2f%%",
        X_train.shape,
        X_test.shape,
        100.0 * float(y_train.mean()),
        100.0 * float(y_test.mean()),
    )

    psi_df = calculate_psi(X_train, X_test, bins=args.psi_bins)
    psi_out = drift_dir / "psi_results.csv"
    psi_df.to_csv(psi_out, index=False)
    plot_psi_bar(psi_df, drift_dir / "psi_bar.png", top_n=30)
    plot_top_drift_histograms(X_train, X_test, psi_df, drift_dir / "psi_hists", top_k=6)
    if not psi_df.empty:
        n_mod = int((psi_df["psi"].between(0.10, 0.25)).sum())
        n_high = int((psi_df["psi"] > 0.25).sum())
        logging.info("PSI saved -> %s | moderate=%d | high=%d", psi_out, n_mod, n_high)
        logging.info("Top PSI:\n%s", psi_df.head(10).to_string(index=False))
    else:
        logging.info("PSI saved -> %s (no numeric columns or insufficient data)", psi_out)

    # Step 2 — Preprocess + Select + Save
    (
        Xtr_sel_winz,
        Xte_sel_winz,
        Xtr_scaled,
        Xte_scaled,
        selected,
        imputer,
        winz,
        scaler,
    ) = fit_preprocessing_and_select(X_train, X_test)

    save_preproc_artifacts(
        preproc_dir,
        selected,
        imputer,
        winz,
        scaler,
        Xtr_sel_winz,
        Xte_sel_winz,
        Xtr_scaled,
        Xte_scaled,
        y_train,
        y_test,
    )
    logging.info("Selected features: %d / %d", len(selected), X.shape[1])
    logging.info("Preprocessing artifacts saved -> %s", preproc_dir)

    # Step 3 — Train + Evaluate + Save
    spw = pos_weight(y_train)
    models, tuning_summaries = tune_and_fit_models(Xtr_scaled, y_train, spw)
    save_models(models, models_dir)
    save_best_params(models, models_dir / "best_params.json")

    rows: List[Dict[str, float]] = []
    for name, mdl in models.items():
        metrics = evaluate_model(
            name=name,
            model=mdl,
            X_tr=Xtr_scaled,
            y_tr=y_train,
            X_te=Xte_scaled,
            y_te=y_test,
            out_dir=eval_dir,
        )
        rows.append(metrics)
        logging.info(
            "%s -> ROC-AUC test=%.3f | PR-AUC test=%.3f | F1 test=%.3f | Brier test=%.3f",
            name,
            metrics["roc_auc_test"],
            metrics["pr_auc_test"],
            metrics["f1_test"],
            metrics["brier_test"],
        )

    cmp_df = pd.DataFrame(rows).sort_values("pr_auc_test", ascending=False)
    cmp_out = eval_dir / "model_comparison.csv"
    cmp_df.to_csv(cmp_out, index=False)
    logging.info("Comparison table saved -> %s", cmp_out)

    # SHAP for the best model by PR-AUC (Test)
    if not cmp_df.empty:
        best_row = cmp_df.iloc[0]
        best_name = str(best_row["model"])
        best_model = models[best_name]
        shap_summary_best_model(best_name, best_model, Xte_scaled, shap_dir)

    # Reports
    if args.report_format == "console":
        write_console_report(df.shape, class_stats, selected, tuning_summaries, cmp_out, psi_out)
    else:
        md_report = write_markdown_report(
            artifacts_dir=artifacts_dir,
            eda_dir=eda_dir,
            preproc_dir=preproc_dir,
            models_dir=models_dir,
            eval_dir=eval_dir,
            drift_dir=drift_dir,
            shap_dir=shap_dir,
            df_shape=df.shape,
            class_stats=class_stats,
            selected_features=selected,
            high_corr_pairs=high_corr_pairs,
            tuning_summaries=tuning_summaries,
        )
        if args.report_format == "pdf":
            out_pdf = artifacts_dir / "report.pdf"
            if write_pdf_report(md_report, out_pdf) is None:
                logging.info("Falling back to Markdown report at: %s", md_report)

    logging.info("All steps complete. Artifacts -> %s", artifacts_dir)


if __name__ == "__main__":
    main()
