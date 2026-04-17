import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GaussianNoise, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

NSL_URLS = {
    "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "test": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_nsl_kdd(data_dir: Path) -> tuple[Path, Path]:
    ensure_dir(data_dir)
    train_path = data_dir / "KDDTrain+.txt"
    test_path = data_dir / "KDDTest+.txt"

    if not train_path.exists():
        urlretrieve(NSL_URLS["train"], train_path)
    if not test_path.exists():
        urlretrieve(NSL_URLS["test"], test_path)

    return train_path, test_path


def load_nsl_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "label" in df.columns:
        return df
    return pd.read_csv(file_path, names=NSL_KDD_COLUMNS)


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    y = (df["label"].astype(str) != "normal").astype(int).values
    X = df.drop(columns=["label", "difficulty"], errors="ignore").copy()
    return X, y


def split_features_optional(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
    y = None
    if "label" in df.columns:
        y = (df["label"].astype(str) != "normal").astype(int).values
    X = df.drop(columns=["label", "difficulty"], errors="ignore").copy()
    return X, y


def fit_preprocessing(X_train: pd.DataFrame) -> tuple[OrdinalEncoder, StandardScaler, list[str], list[str]]:
    X_train = X_train.copy()
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [column for column in X_train.columns if column not in cat_cols]

    for column in cat_cols:
        X_train[column] = X_train[column].fillna("Unknown").astype(str)
    for column in num_cols:
        X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
        median = X_train[column].median()
        if pd.isna(median):
            median = 0.0
        X_train[column] = X_train[column].fillna(median)

    ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        X_train[cat_cols] = ordinal_encoder.fit_transform(X_train[cat_cols])

    scaler = StandardScaler()
    scaler.fit(X_train.values)
    feature_names = X_train.columns.tolist()
    return ordinal_encoder, scaler, feature_names, cat_cols


def transform_features(
    X: pd.DataFrame,
    ordinal_encoder: OrdinalEncoder,
    scaler: StandardScaler,
    feature_names: list[str],
    cat_cols: list[str],
) -> np.ndarray:
    X = X.copy()
    for column in feature_names:
        if column not in X.columns:
            X[column] = 0
    X = X[feature_names]

    num_cols = [column for column in feature_names if column not in cat_cols]
    for column in cat_cols:
        X[column] = X[column].fillna("Unknown").astype(str)
    for column in num_cols:
        X[column] = pd.to_numeric(X[column], errors="coerce")
        median = X[column].median()
        if pd.isna(median):
            median = 0.0
        X[column] = X[column].fillna(median)

    if cat_cols:
        X[cat_cols] = ordinal_encoder.transform(X[cat_cols])
    return scaler.transform(X.values)


def anomaly_score(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return -model.decision_function(X)
    if hasattr(model, "score_samples"):
        return -model.score_samples(X)
    raw = model.predict(X)
    return np.where(raw == 1, 0.0, 1.0)


def anomaly_pred_to_binary(raw_pred: np.ndarray) -> np.ndarray:
    return np.where(raw_pred == 1, 0, 1)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        roc_auc = float(roc_auc_score(y_true, score))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, score))
    except ValueError:
        pr_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def normalize_score(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    eps = 1e-12
    denom = max(upper - lower, eps)
    return np.clip((values - lower) / denom, 0.0, 1.0)


def select_f1_threshold(y_true: np.ndarray, y_score: np.ndarray, fallback: float = 0.5) -> float:
    if y_true.size == 0:
        return float(fallback)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return float(fallback)
    f1_values = (2.0 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def parse_int_csv(value: str, field_name: str) -> list[int]:
    values: list[int] = []
    for part in str(value).split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in {field_name}: '{token}'") from exc
    if not values:
        raise ValueError(f"{field_name} cannot be empty.")
    return values


def parse_float_csv(value: str, field_name: str) -> list[float]:
    values: list[float] = []
    for part in str(value).split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid float in {field_name}: '{token}'") from exc
    if not values:
        raise ValueError(f"{field_name} cannot be empty.")
    return values


def masked_recall(y_pred: np.ndarray, mask: np.ndarray) -> float:
    count = int(mask.sum())
    if count == 0:
        return float("nan")
    return float((y_pred[mask] == 1).sum() / count)


def normal_fpr(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    normal_mask = y_true == 0
    if int(normal_mask.sum()) == 0:
        return float("nan")
    return float(y_pred[normal_mask].mean())


def to_binary_labels(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        raw = pd.to_numeric(series, errors="coerce")
        if raw.notna().all():
            vals = raw.astype(int).values
            unique_vals = set(np.unique(vals).tolist())
            if unique_vals.issubset({0, 1}):
                return vals

    normalized = series.astype(str).str.strip().str.lower()
    if normalized.isin({"0", "1"}).all():
        return normalized.astype(int).values

    benign_alias = {"normal", "benign", "false", "non-attack", "0"}
    return (~normalized.isin(benign_alias)).astype(int).values


def safe_paired_tests(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size != y.size or x.size < 2:
        return float("nan"), float("nan")

    if np.isnan(x).any() or np.isnan(y).any():
        return float("nan"), float("nan")

    ttest_p = float("nan")
    wilcoxon_p = float("nan")

    try:
        _, ttest_p = ttest_rel(y, x)
    except Exception:
        ttest_p = float("nan")

    diff = y - x
    if not np.allclose(diff, diff[0]):
        try:
            _, wilcoxon_p = wilcoxon(diff)
        except Exception:
            wilcoxon_p = float("nan")

    return float(ttest_p), float(wilcoxon_p)


def build_prod_autoencoder(input_dim: int, latent_dim: int = 48, noise_std: float = 0.02) -> tuple[Model, Model]:
    inputs = Input(shape=(input_dim,), name="prod_ae_input")
    x = GaussianNoise(noise_std, name="noise")(inputs)
    x = Dense(256, activation="relu", name="enc_1")(x)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(0.15, name="drop_1")(x)
    x = Dense(128, activation="relu", name="enc_2")(x)
    latent = Dense(latent_dim, activation="relu", name="latent")(x)
    x = Dense(128, activation="relu", name="dec_1")(latent)
    x = Dropout(0.10, name="drop_2")(x)
    x = Dense(256, activation="relu", name="dec_2")(x)
    outputs = Dense(input_dim, activation="linear", name="prod_ae_output")(x)

    autoencoder = Model(inputs, outputs, name="prod_autoencoder")
    encoder = Model(inputs, latent, name="prod_encoder")
    autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return autoencoder, encoder


def run_train_prod(args) -> None:
    np.random.seed(args.seed)

    train_df = load_nsl_file(Path(args.train_path))
    test_df = load_nsl_file(Path(args.test_path))

    X_train_df, y_train = split_features(train_df)
    X_test_df, y_test = split_features(test_df)

    ordinal_encoder, scaler, feature_names, cat_cols = fit_preprocessing(X_train_df)
    X_train = transform_features(X_train_df, ordinal_encoder, scaler, feature_names, cat_cols)
    X_test = transform_features(X_test_df, ordinal_encoder, scaler, feature_names, cat_cols)

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=args.calibration_fraction,
        random_state=args.seed,
        stratify=y_train,
    )

    X_fit_normal = X_fit[y_fit == 0]

    autoencoder, encoder = build_prod_autoencoder(
        input_dim=X_train.shape[1],
        latent_dim=args.latent_dim,
        noise_std=args.noise_std,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    autoencoder.fit(
        X_fit_normal,
        X_fit_normal,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )

    fit_recon = autoencoder.predict(X_fit, verbose=0)
    cal_recon = autoencoder.predict(X_cal, verbose=0)
    test_recon = autoencoder.predict(X_test, verbose=0)
    fit_recon_error = np.mean((X_fit - fit_recon) ** 2, axis=1)
    cal_recon_error = np.mean((X_cal - cal_recon) ** 2, axis=1)
    test_recon_error = np.mean((X_test - test_recon) ** 2, axis=1)

    latent_fit = encoder.predict(X_fit, verbose=0)
    latent_cal = encoder.predict(X_cal, verbose=0)
    latent_test = encoder.predict(X_test, verbose=0)

    latent_lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    latent_lr.fit(latent_fit, y_fit)
    lr_score_cal = latent_lr.predict_proba(latent_cal)[:, 1]
    lr_score_test = latent_lr.predict_proba(latent_test)[:, 1]

    latent_if = IsolationForest(
        n_estimators=args.if_estimators,
        contamination="auto",
        random_state=args.seed,
        n_jobs=-1,
    )
    latent_if.fit(latent_fit[y_fit == 0])
    if_score_fit = -latent_if.decision_function(latent_fit)
    if_score_cal = -latent_if.decision_function(latent_cal)
    if_score_test = -latent_if.decision_function(latent_test)

    recon_scale_min = float(np.percentile(fit_recon_error, 1.0))
    recon_scale_max = float(np.percentile(fit_recon_error, 99.0))
    if_scale_min = float(np.percentile(if_score_fit, 1.0))
    if_scale_max = float(np.percentile(if_score_fit, 99.0))

    recon_norm_cal = normalize_score(cal_recon_error, recon_scale_min, recon_scale_max)
    recon_norm_test = normalize_score(test_recon_error, recon_scale_min, recon_scale_max)
    if_norm_cal = normalize_score(if_score_cal, if_scale_min, if_scale_max)
    if_norm_test = normalize_score(if_score_test, if_scale_min, if_scale_max)

    recon_iso = IsotonicRegression(out_of_bounds="clip")
    recon_iso.fit(recon_norm_cal, y_cal)
    if_iso = IsotonicRegression(out_of_bounds="clip")
    if_iso.fit(if_norm_cal, y_cal)
    lr_iso = IsotonicRegression(out_of_bounds="clip")
    lr_iso.fit(lr_score_cal, y_cal)

    recon_prob_cal = recon_iso.predict(recon_norm_cal)
    if_prob_cal = if_iso.predict(if_norm_cal)
    lr_prob_cal = lr_iso.predict(lr_score_cal)

    meta_features_cal = np.column_stack(
        [
            recon_prob_cal,
            lr_prob_cal,
            if_prob_cal,
            np.maximum(recon_prob_cal, if_prob_cal),
            (recon_prob_cal + lr_prob_cal + if_prob_cal) / 3.0,
        ]
    )

    idx_all = np.arange(y_cal.shape[0])
    idx_meta, idx_threshold = train_test_split(
        idx_all,
        test_size=args.threshold_fraction,
        random_state=args.seed,
        stratify=y_cal,
    )

    meta_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    meta_clf.fit(meta_features_cal[idx_meta], y_cal[idx_meta])

    threshold_scores = meta_clf.predict_proba(meta_features_cal[idx_threshold])[:, 1]
    threshold_labels = y_cal[idx_threshold]
    tuned_threshold = select_f1_threshold(threshold_labels, threshold_scores, fallback=0.5)

    normal_threshold_scores = threshold_scores[threshold_labels == 0]
    if normal_threshold_scores.size > 0:
        conformal_threshold = float(np.quantile(normal_threshold_scores, 1.0 - args.alpha_conformal))
    else:
        conformal_threshold = float(tuned_threshold)

    recon_prob_test = recon_iso.predict(recon_norm_test)
    if_prob_test = if_iso.predict(if_norm_test)
    lr_prob_test = lr_iso.predict(lr_score_test)

    meta_features_test = np.column_stack(
        [
            recon_prob_test,
            lr_prob_test,
            if_prob_test,
            np.maximum(recon_prob_test, if_prob_test),
            (recon_prob_test + lr_prob_test + if_prob_test) / 3.0,
        ]
    )
    prod_prob_test = meta_clf.predict_proba(meta_features_test)[:, 1]

    recon_threshold = float(np.percentile(fit_recon_error[y_fit == 0], args.recon_percentile))
    if_threshold = float(np.percentile(if_score_fit[y_fit == 0], args.if_percentile))

    pred_recon = (test_recon_error >= recon_threshold).astype(int)
    pred_lr = (lr_prob_test >= 0.5).astype(int)
    pred_if = (if_score_test >= if_threshold).astype(int)
    pred_prod_tuned = (prod_prob_test >= tuned_threshold).astype(int)
    pred_prod_conformal = (prod_prob_test >= conformal_threshold).astype(int)

    metrics = {
        "reconstruction": evaluate_predictions(y_test, pred_recon, test_recon_error),
        "latent_logistic": evaluate_predictions(y_test, pred_lr, lr_prob_test),
        "latent_isolation_forest": evaluate_predictions(y_test, pred_if, if_score_test),
        "production_tuned": evaluate_predictions(y_test, pred_prod_tuned, prod_prob_test),
        "production_conformal": evaluate_predictions(y_test, pred_prod_conformal, prod_prob_test),
    }

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    autoencoder.save((output_dir / "prod_autoencoder.keras").as_posix())
    joblib.dump(latent_lr, output_dir / "prod_latent_logreg.joblib")
    joblib.dump(latent_if, output_dir / "prod_latent_isolation_forest.joblib")
    joblib.dump(meta_clf, output_dir / "prod_meta_classifier.joblib")
    joblib.dump(recon_iso, output_dir / "prod_recon_isotonic.joblib")
    joblib.dump(lr_iso, output_dir / "prod_latent_logreg_isotonic.joblib")
    joblib.dump(if_iso, output_dir / "prod_latent_if_isotonic.joblib")

    joblib.dump(scaler, output_dir / "standard_scaler.joblib")
    joblib.dump(ordinal_encoder, output_dir / "ordinal_encoder.joblib")
    joblib.dump(feature_names, output_dir / "feature_names.joblib")
    joblib.dump(cat_cols, output_dir / "categorical_columns.joblib")

    thresholds = {
        "reconstruction_threshold": recon_threshold,
        "latent_if_threshold": if_threshold,
        "production_tuned_threshold": tuned_threshold,
        "production_conformal_threshold": conformal_threshold,
        "alpha_conformal": float(args.alpha_conformal),
        "reconstruction_score_scaling": {
            "min": recon_scale_min,
            "max": recon_scale_max,
        },
        "latent_if_score_scaling": {
            "min": if_scale_min,
            "max": if_scale_max,
        },
    }
    with open(output_dir / "prod_thresholds.json", "w", encoding="utf-8") as file:
        json.dump(thresholds, file, indent=2)

    metadata = {
        "pipeline_name": "Production Hybrid Ensemble",
        "version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "NSL-KDD",
        "label_mapping": {"normal": 0, "attack": 1},
        "data_split": {
            "train_rows": int(X_train.shape[0]),
            "calibration_fraction": float(args.calibration_fraction),
            "threshold_fraction": float(args.threshold_fraction),
        },
        "model_hyperparameters": {
            "latent_dim": int(args.latent_dim),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "noise_std": float(args.noise_std),
            "if_estimators": int(args.if_estimators),
        },
        "metrics": metrics,
    }
    with open(output_dir / "prod_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    monitor = {
        "reference_feature_mean": scaler.mean_.tolist(),
        "reference_feature_scale": scaler.scale_.tolist(),
        "reference_anomaly_rate_train": float(np.mean(y_train)),
        "reference_probability_quantiles": {
            "q50": float(np.quantile(prod_prob_test, 0.50)),
            "q90": float(np.quantile(prod_prob_test, 0.90)),
            "q99": float(np.quantile(prod_prob_test, 0.99)),
        },
    }
    with open(output_dir / "prod_monitor_baseline.json", "w", encoding="utf-8") as file:
        json.dump(monitor, file, indent=2)

    print(f"[production_tuned] F1={metrics['production_tuned']['f1']:.4f} ROC-AUC={metrics['production_tuned']['roc_auc']:.4f}")
    print(
        f"[production_conformal] F1={metrics['production_conformal']['f1']:.4f} "
        f"ROC-AUC={metrics['production_conformal']['roc_auc']:.4f}"
    )
    print(f"Saved production artifacts to {output_dir}")


def run_test_prod(args) -> None:
    model_dir = Path(args.model_dir)
    input_df = load_nsl_file(Path(args.input_path))
    X_input_df, _ = split_features_optional(input_df)

    scaler = joblib.load(model_dir / "standard_scaler.joblib")
    ordinal_encoder = joblib.load(model_dir / "ordinal_encoder.joblib")
    feature_names = joblib.load(model_dir / "feature_names.joblib")
    cat_cols = joblib.load(model_dir / "categorical_columns.joblib")

    autoencoder = load_model(model_dir / "prod_autoencoder.keras")
    latent_lr = joblib.load(model_dir / "prod_latent_logreg.joblib")
    latent_if = joblib.load(model_dir / "prod_latent_isolation_forest.joblib")
    meta_clf = joblib.load(model_dir / "prod_meta_classifier.joblib")

    recon_iso = joblib.load(model_dir / "prod_recon_isotonic.joblib")
    lr_iso = joblib.load(model_dir / "prod_latent_logreg_isotonic.joblib")
    if_iso = joblib.load(model_dir / "prod_latent_if_isotonic.joblib")

    with open(model_dir / "prod_thresholds.json", "r", encoding="utf-8") as file:
        thresholds = json.load(file)

    X_scaled = transform_features(X_input_df, ordinal_encoder, scaler, feature_names, cat_cols)
    recon = autoencoder.predict(X_scaled, verbose=0)
    recon_error = np.mean((X_scaled - recon) ** 2, axis=1)

    encoder = Model(autoencoder.input, autoencoder.get_layer("latent").output)
    latent = encoder.predict(X_scaled, verbose=0)
    lr_score = latent_lr.predict_proba(latent)[:, 1]
    if_score = -latent_if.decision_function(latent)

    recon_norm = normalize_score(
        recon_error,
        float(thresholds["reconstruction_score_scaling"]["min"]),
        float(thresholds["reconstruction_score_scaling"]["max"]),
    )
    if_norm = normalize_score(
        if_score,
        float(thresholds["latent_if_score_scaling"]["min"]),
        float(thresholds["latent_if_score_scaling"]["max"]),
    )

    recon_prob = recon_iso.predict(recon_norm)
    lr_prob = lr_iso.predict(lr_score)
    if_prob = if_iso.predict(if_norm)

    meta_features = np.column_stack(
        [
            recon_prob,
            lr_prob,
            if_prob,
            np.maximum(recon_prob, if_prob),
            (recon_prob + lr_prob + if_prob) / 3.0,
        ]
    )
    prod_prob = meta_clf.predict_proba(meta_features)[:, 1]

    pred_tuned = (prod_prob >= float(thresholds["production_tuned_threshold"])).astype(int)
    pred_conformal = (prod_prob >= float(thresholds["production_conformal_threshold"])).astype(int)

    stacked_components = np.column_stack([recon_prob, lr_prob, if_prob])
    component_names = np.array(["reconstruction", "latent_logreg", "latent_isolation_forest"])
    dominant_component = component_names[np.argmax(stacked_components, axis=1)]

    severity = pd.cut(
        prod_prob,
        bins=[-0.001, 0.30, 0.60, 0.80, 1.001],
        labels=["low", "medium", "high", "critical"],
    )

    output_df = input_df.copy()
    output_df["pred_prod_tuned"] = pred_tuned
    output_df["pred_prod_conformal"] = pred_conformal
    output_df["prod_anomaly_probability"] = prod_prob
    output_df["reconstruction_error"] = recon_error
    output_df["latent_logreg_score"] = lr_score
    output_df["latent_if_score"] = if_score
    output_df["reconstruction_prob"] = recon_prob
    output_df["latent_logreg_prob"] = lr_prob
    output_df["latent_if_prob"] = if_prob
    output_df["dominant_detector"] = dominant_component
    output_df["risk_tier"] = severity.astype(str)
    output_df.to_csv(args.output_csv, index=False)
    print(f"Saved production predictions to {args.output_csv}")


def run_train_ml(args) -> None:
    train_df = load_nsl_file(Path(args.train_path))
    test_df = load_nsl_file(Path(args.test_path))

    X_train_df, y_train = split_features(train_df)
    X_test_df, y_test = split_features(test_df)

    ordinal_encoder, scaler, feature_names, cat_cols = fit_preprocessing(X_train_df)
    X_train = transform_features(X_train_df, ordinal_encoder, scaler, feature_names, cat_cols)
    X_test = transform_features(X_test_df, ordinal_encoder, scaler, feature_names, cat_cols)
    X_train_normal = X_train[y_train == 0]

    models = {
        "isolation_forest": IsolationForest(
            n_estimators=args.if_estimators,
            contamination="auto",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "local_outlier_factor": LocalOutlierFactor(
            n_neighbors=args.lof_neighbors,
            novelty=True,
            contamination="auto",
            n_jobs=-1,
        ),
        "oneclass_svm": OneClassSVM(kernel="rbf", gamma="scale", nu=args.ocsvm_nu),
    }

    ensure_dir(Path(args.output_dir))

    summary = []
    for name, model in models.items():
        model.fit(X_train_normal)
        raw_pred = model.predict(X_test)
        y_pred = anomaly_pred_to_binary(raw_pred)
        score = anomaly_score(model, X_test)
        metrics = evaluate_predictions(y_test, y_pred, score)
        metrics["model"] = name
        summary.append(metrics)
        joblib.dump(model, Path(args.output_dir) / f"{name}.joblib")
        print(f"[{name}] F1={metrics['f1']:.4f} ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f}")

    best = sorted(summary, key=lambda row: (row["f1"], row["pr_auc"]), reverse=True)[0]

    joblib.dump(scaler, Path(args.output_dir) / "standard_scaler.joblib")
    joblib.dump(ordinal_encoder, Path(args.output_dir) / "ordinal_encoder.joblib")
    joblib.dump(feature_names, Path(args.output_dir) / "feature_names.joblib")
    joblib.dump(cat_cols, Path(args.output_dir) / "categorical_columns.joblib")

    metadata = {
        "dataset": "NSL-KDD",
        "label_mapping": {"normal": 0, "attack": 1},
        "models": list(models.keys()),
        "best_model": best["model"],
        "summary_metrics": summary,
    }
    with open(Path(args.output_dir) / "ml_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved ML artifacts to {args.output_dir}")


def run_test_ml(args) -> None:
    input_df = load_nsl_file(Path(args.input_path))
    model_dir = Path(args.model_dir)

    scaler = joblib.load(model_dir / "standard_scaler.joblib")
    ordinal_encoder = joblib.load(model_dir / "ordinal_encoder.joblib")
    feature_names = joblib.load(model_dir / "feature_names.joblib")
    cat_cols = joblib.load(model_dir / "categorical_columns.joblib")

    with open(model_dir / "ml_metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)
    model_name = args.model_name or metadata.get("best_model", "isolation_forest")

    normalized_name = str(model_name).strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "isolation_forest": "isolation_forest",
        "local_outlier_factor": "local_outlier_factor",
        "one_class_svm": "oneclass_svm",
        "oneclass_svm": "oneclass_svm",
    }
    file_stem = alias_map.get(normalized_name, normalized_name)

    candidate_paths = [
        model_dir / f"{file_stem}.joblib",
        model_dir / f"{model_name}.joblib",
    ]

    model = None
    selected_path = None
    for candidate in candidate_paths:
        if candidate.exists():
            model = joblib.load(candidate)
            selected_path = candidate
            break
    if model is None:
        available_files = sorted([path.name for path in model_dir.glob("*.joblib")])
        raise FileNotFoundError(
            f"Could not find model file for '{model_name}'. Tried: {[str(p.name) for p in candidate_paths]}. "
            f"Available: {available_files}"
        )
    X_input, _ = split_features(input_df)
    X_scaled = transform_features(X_input, ordinal_encoder, scaler, feature_names, cat_cols)

    raw_pred = model.predict(X_scaled)
    pred = anomaly_pred_to_binary(raw_pred)
    score = anomaly_score(model, X_scaled)

    output_df = input_df.copy()
    output_df["pred_label"] = pred
    output_df["anomaly_score"] = score
    output_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv} using model '{selected_path.stem}'")


def build_autoencoder(input_dim: int, latent_dim: int = 32) -> tuple[Model, Model]:
    inputs = Input(shape=(input_dim,), name="ae_input")
    encoded = Dense(128, activation="relu", name="enc_1")(inputs)
    latent = Dense(latent_dim, activation="relu", name="latent")(encoded)
    decoded = Dense(128, activation="relu", name="dec_1")(latent)
    outputs = Dense(input_dim, activation="linear", name="ae_output")(decoded)

    autoencoder = Model(inputs, outputs, name="autoencoder")
    encoder = Model(inputs, latent, name="encoder")
    autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return autoencoder, encoder


def run_train_dl(args) -> None:
    train_df = load_nsl_file(Path(args.train_path))
    test_df = load_nsl_file(Path(args.test_path))

    X_train_df, y_train = split_features(train_df)
    X_test_df, y_test = split_features(test_df)

    ordinal_encoder, scaler, feature_names, cat_cols = fit_preprocessing(X_train_df)
    X_train = transform_features(X_train_df, ordinal_encoder, scaler, feature_names, cat_cols)
    X_test = transform_features(X_test_df, ordinal_encoder, scaler, feature_names, cat_cols)
    X_train_normal = X_train[y_train == 0]

    autoencoder, encoder = build_autoencoder(X_train.shape[1], latent_dim=args.latent_dim)
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    autoencoder.fit(
        X_train_normal,
        X_train_normal,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )

    recon_train = autoencoder.predict(X_train_normal, verbose=0)
    train_errors = np.mean((X_train_normal - recon_train) ** 2, axis=1)
    threshold = float(np.percentile(train_errors, args.threshold_percentile))

    latent_train = encoder.predict(X_train, verbose=0)
    latent_test = encoder.predict(X_test, verbose=0)
    latent_clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    latent_clf.fit(latent_train, y_train)

    recon_test = autoencoder.predict(X_test, verbose=0)
    test_errors = np.mean((X_test - recon_test) ** 2, axis=1)
    ae_pred = (test_errors >= threshold).astype(int)
    ae_metrics = evaluate_predictions(y_test, ae_pred, test_errors)

    clf_score = latent_clf.predict_proba(latent_test)[:, 1]
    clf_pred = (clf_score >= 0.5).astype(int)
    clf_metrics = evaluate_predictions(y_test, clf_pred, clf_score)

    ensure_dir(Path(args.output_dir))
    autoencoder.save(Path(args.output_dir) / "autoencoder_major.keras")
    joblib.dump(latent_clf, Path(args.output_dir) / "latent_classifier.joblib")
    joblib.dump(scaler, Path(args.output_dir) / "standard_scaler.joblib")
    joblib.dump(ordinal_encoder, Path(args.output_dir) / "ordinal_encoder.joblib")
    joblib.dump(feature_names, Path(args.output_dir) / "feature_names.joblib")
    joblib.dump(cat_cols, Path(args.output_dir) / "categorical_columns.joblib")

    metadata = {
        "dataset": "NSL-KDD",
        "label_mapping": {"normal": 0, "attack": 1},
        "threshold_percentile": args.threshold_percentile,
        "reconstruction_threshold": threshold,
        "autoencoder_metrics": ae_metrics,
        "latent_classifier_metrics": clf_metrics,
    }
    with open(Path(args.output_dir) / "dl_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Autoencoder F1={ae_metrics['f1']:.4f} ROC-AUC={ae_metrics['roc_auc']:.4f} PR-AUC={ae_metrics['pr_auc']:.4f}")
    print(f"Latent-Classifier F1={clf_metrics['f1']:.4f} ROC-AUC={clf_metrics['roc_auc']:.4f} PR-AUC={clf_metrics['pr_auc']:.4f}")
    print(f"Saved DL artifacts to {args.output_dir}")


def load_first_existing(candidates: list[Path], artifact_label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = [path.name for path in candidates]
    raise FileNotFoundError(f"Could not find {artifact_label}. Tried: {tried}")


def run_test_dl(args) -> None:
    model_dir = Path(args.model_dir)
    input_df = load_nsl_file(Path(args.input_path))
    X_input_df, _ = split_features(input_df)

    scaler_path = load_first_existing(
        [
            model_dir / "standard_scaler.joblib",
            model_dir / "minmax_scaler.joblib",
        ],
        "DL scaler",
    )
    encoder_path = load_first_existing([model_dir / "ordinal_encoder.joblib"], "DL ordinal encoder")
    feature_names_path = load_first_existing([model_dir / "feature_names.joblib"], "DL feature names")
    cat_cols_path = load_first_existing([model_dir / "categorical_columns.joblib"], "DL categorical columns")
    latent_clf_path = load_first_existing(
        [
            model_dir / "latent_classifier.joblib",
            model_dir / "latent_logreg_classifier.joblib",
        ],
        "DL latent classifier",
    )
    autoencoder_path = load_first_existing(
        [
            model_dir / "autoencoder_major.keras",
            model_dir / "unsw_nb15_autoencoder.keras",
        ],
        "DL autoencoder model",
    )

    scaler = joblib.load(scaler_path)
    ordinal_encoder = joblib.load(encoder_path)
    feature_names = joblib.load(feature_names_path)
    cat_cols = joblib.load(cat_cols_path)
    latent_clf = joblib.load(latent_clf_path)
    autoencoder = load_model(autoencoder_path)

    metadata_path = model_dir / "dl_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if "reconstruction_threshold" in metadata:
        threshold = float(metadata["reconstruction_threshold"])
    else:
        threshold_path = model_dir / "reconstruction_threshold.json"
        with open(threshold_path, "r", encoding="utf-8") as file:
            threshold_json = json.load(file)
        threshold = float(threshold_json["threshold"])

    X_scaled = transform_features(X_input_df, ordinal_encoder, scaler, feature_names, cat_cols)
    recon = autoencoder.predict(X_scaled, verbose=0)
    recon_error = np.mean((X_scaled - recon) ** 2, axis=1)
    ae_pred = (recon_error >= threshold).astype(int)

    latent_layer = autoencoder.get_layer("latent").output
    encoder = Model(autoencoder.input, latent_layer)
    latent = encoder.predict(X_scaled, verbose=0)
    clf_score = latent_clf.predict_proba(latent)[:, 1]
    clf_pred = (clf_score >= 0.5).astype(int)

    gated_threshold_path = model_dir / "gated_ensemble_thresholds.json"
    latent_probability_threshold = 0.5
    if gated_threshold_path.exists():
        with open(gated_threshold_path, "r", encoding="utf-8") as file:
            gated_thresholds = json.load(file)
        latent_probability_threshold = float(
            gated_thresholds.get("latent_probability_threshold", latent_probability_threshold)
        )
    gated_pred = np.where((ae_pred == 1) | (clf_score >= latent_probability_threshold), 1, 0)

    output_df = input_df.copy()
    output_df["pred_ae"] = ae_pred
    output_df["pred_latent_clf"] = clf_pred
    output_df["pred_gated_ensemble"] = gated_pred
    output_df["reconstruction_error"] = recon_error
    output_df["latent_clf_score"] = clf_score
    output_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


def run_eval_publication(args) -> None:
    train_df = load_nsl_file(Path(args.train_path))
    test_df = load_nsl_file(Path(args.test_path))

    X_train_all_df, y_train_all = split_features(train_df)
    X_test_df, y_test = split_features(test_df)

    seeds = parse_int_csv(args.seeds, "seeds")
    target_fprs = parse_float_csv(args.target_fprs, "target_fprs")

    if not (0.0 < args.sample_fraction <= 1.0):
        raise ValueError("--sample_fraction must be in (0, 1].")
    if not (0.0 < args.normal_val_fraction < 1.0):
        raise ValueError("--normal_val_fraction must be in (0, 1).")
    if not (0.0 < args.latent_cal_fraction < 1.0):
        raise ValueError("--latent_cal_fraction must be in (0, 1).")
    if not (0.0 < args.reconstruction_quantile < 1.0):
        raise ValueError("--reconstruction_quantile must be in (0, 1).")
    if not (0.0 < args.latent_quantile < 1.0):
        raise ValueError("--latent_quantile must be in (0, 1).")
    if not (0.0 < args.alpha_conformal < 1.0):
        raise ValueError("--alpha_conformal must be in (0, 1).")

    for fpr in target_fprs:
        if not (0.0 < fpr < 1.0):
            raise ValueError(f"Invalid target FPR {fpr}. Each target FPR must be in (0, 1).")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    test_attack_labels = test_df["label"].astype(str).values
    train_attack_types = set(train_df.loc[train_df["label"] != "normal", "label"].astype(str).unique())
    seen_mask = np.array([(label != "normal") and (label in train_attack_types) for label in test_attack_labels])
    unseen_mask = np.array([(label != "normal") and (label not in train_attack_types) for label in test_attack_labels])

    external_df: pd.DataFrame | None = None
    y_external: np.ndarray | None = None
    external_mapping: dict[str, str] = {}
    if args.external_path:
        external_path = Path(args.external_path)
        if not external_path.exists():
            raise FileNotFoundError(f"External dataset file not found: {external_path}")
        external_df = pd.read_csv(external_path)
        if args.external_label_column and args.external_label_column in external_df.columns:
            y_external = to_binary_labels(external_df[args.external_label_column])

        if args.external_mapping_json:
            mapping_path = Path(args.external_mapping_json)
            if not mapping_path.exists():
                raise FileNotFoundError(f"External mapping json not found: {mapping_path}")
            with open(mapping_path, "r", encoding="utf-8") as file:
                loaded_mapping = json.load(file)
            if not isinstance(loaded_mapping, dict):
                raise ValueError("--external_mapping_json must contain a JSON object (external->nsl feature mapping).")
            external_mapping = {str(k): str(v) for k, v in loaded_mapping.items()}

    def build_external_feature_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        prepared = df.copy()
        if external_mapping:
            rename_map = {source: target for source, target in external_mapping.items() if source in prepared.columns}
            if rename_map:
                prepared = prepared.rename(columns=rename_map)
        if args.external_label_column and args.external_label_column in prepared.columns:
            prepared = prepared.drop(columns=[args.external_label_column], errors="ignore")

        for feature in feature_names:
            if feature not in prepared.columns:
                prepared[feature] = 0

        return prepared[feature_names]

    seed_rows: list[dict] = []
    attack_rows: list[dict] = []
    external_rows: list[dict] = []

    for seed in seeds:
        np.random.seed(seed)

        if args.sample_fraction < 1.0:
            X_seed_df, _, y_seed, _ = train_test_split(
                X_train_all_df,
                y_train_all,
                train_size=args.sample_fraction,
                random_state=seed,
                stratify=y_train_all,
            )
        else:
            X_seed_df = X_train_all_df.copy()
            y_seed = y_train_all.copy()

        ordinal_encoder, scaler, feature_names, cat_cols = fit_preprocessing(X_seed_df)
        X_seed = transform_features(X_seed_df, ordinal_encoder, scaler, feature_names, cat_cols)
        X_test = transform_features(X_test_df, ordinal_encoder, scaler, feature_names, cat_cols)

        X_seed_normal = X_seed[y_seed == 0]
        if X_seed_normal.shape[0] < 100:
            raise ValueError(
                f"Seed {seed}: not enough normal rows after sampling ({X_seed_normal.shape[0]}). "
                "Increase --sample_fraction."
            )

        x_norm_train, x_norm_val = train_test_split(
            X_seed_normal,
            test_size=args.normal_val_fraction,
            random_state=seed,
        )

        X_lat_train, X_lat_cal, y_lat_train, y_lat_cal = train_test_split(
            X_seed,
            y_seed,
            test_size=args.latent_cal_fraction,
            random_state=seed + 7,
            stratify=y_seed,
        )

        autoencoder, encoder = build_autoencoder(X_seed.shape[1], latent_dim=args.latent_dim)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ]
        autoencoder.fit(
            x_norm_train,
            x_norm_train,
            validation_data=(x_norm_val, x_norm_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=args.verbose,
        )

        val_recon = autoencoder.predict(x_norm_val, verbose=0)
        val_recon_error = np.mean((x_norm_val - val_recon) ** 2, axis=1)
        recon_threshold = float(np.quantile(val_recon_error, args.reconstruction_quantile))

        test_recon = autoencoder.predict(X_test, verbose=0)
        test_recon_error = np.mean((X_test - test_recon) ** 2, axis=1)
        ae_pred = (test_recon_error >= recon_threshold).astype(int)

        latent_train = encoder.predict(X_lat_train, verbose=0)
        latent_cal = encoder.predict(X_lat_cal, verbose=0)
        latent_test = encoder.predict(X_test, verbose=0)

        latent_clf = LogisticRegression(max_iter=3000, random_state=seed)
        latent_clf.fit(latent_train, y_lat_train)

        latent_prob_cal = latent_clf.predict_proba(latent_cal)[:, 1]
        latent_prob_test = latent_clf.predict_proba(latent_test)[:, 1]

        latent_prob_cal_normal = latent_prob_cal[y_lat_cal == 0]
        if latent_prob_cal_normal.size == 0:
            raise ValueError(f"Seed {seed}: no normal rows in latent calibration split.")
        latent_threshold = float(np.quantile(latent_prob_cal_normal, args.latent_quantile))

        latent_pred = (latent_prob_test >= 0.5).astype(int)
        gated_or_pred = np.where((ae_pred == 1) | (latent_prob_test >= latent_threshold), 1, 0)

        eps = 1e-12
        score_ae = test_recon_error
        score_latent = latent_prob_test
        score_gated = np.maximum(
            test_recon_error / max(recon_threshold, eps),
            latent_prob_test / max(latent_threshold, eps),
        )

        cal_recon = autoencoder.predict(X_lat_cal, verbose=0)
        cal_recon_error = np.mean((X_lat_cal - cal_recon) ** 2, axis=1)
        score_gated_cal = np.maximum(
            cal_recon_error / max(recon_threshold, eps),
            latent_prob_cal / max(latent_threshold, eps),
        )
        score_gated_cal_normal = score_gated_cal[y_lat_cal == 0]

        if score_gated_cal_normal.size > 0:
            conformal_threshold = float(np.quantile(score_gated_cal_normal, 1.0 - args.alpha_conformal))
        else:
            conformal_threshold = float(select_f1_threshold(y_lat_cal, score_gated_cal, fallback=1.0))
        gated_conformal_pred = (score_gated >= conformal_threshold).astype(int)

        ae_metrics = evaluate_predictions(y_test, ae_pred, score_ae)
        latent_metrics = evaluate_predictions(y_test, latent_pred, score_latent)
        gated_or_metrics = evaluate_predictions(y_test, gated_or_pred, score_gated)
        gated_conformal_metrics = evaluate_predictions(y_test, gated_conformal_pred, score_gated)

        seed_row = {
            "seed": int(seed),
            "train_rows_seed": int(X_seed.shape[0]),
            "ae_f1": ae_metrics["f1"],
            "ae_precision": ae_metrics["precision"],
            "ae_recall": ae_metrics["recall"],
            "ae_roc_auc": ae_metrics["roc_auc"],
            "ae_pr_auc": ae_metrics["pr_auc"],
            "ae_seen_recall": masked_recall(ae_pred, seen_mask),
            "ae_unseen_recall": masked_recall(ae_pred, unseen_mask),
            "ae_normal_fpr": normal_fpr(ae_pred, y_test),
            "latent_f1": latent_metrics["f1"],
            "latent_precision": latent_metrics["precision"],
            "latent_recall": latent_metrics["recall"],
            "latent_roc_auc": latent_metrics["roc_auc"],
            "latent_pr_auc": latent_metrics["pr_auc"],
            "gated_or_f1": gated_or_metrics["f1"],
            "gated_or_precision": gated_or_metrics["precision"],
            "gated_or_recall": gated_or_metrics["recall"],
            "gated_or_roc_auc": gated_or_metrics["roc_auc"],
            "gated_or_pr_auc": gated_or_metrics["pr_auc"],
            "gated_or_seen_recall": masked_recall(gated_or_pred, seen_mask),
            "gated_or_unseen_recall": masked_recall(gated_or_pred, unseen_mask),
            "gated_or_normal_fpr": normal_fpr(gated_or_pred, y_test),
            "gated_conformal_f1": gated_conformal_metrics["f1"],
            "gated_conformal_precision": gated_conformal_metrics["precision"],
            "gated_conformal_recall": gated_conformal_metrics["recall"],
            "gated_conformal_roc_auc": gated_conformal_metrics["roc_auc"],
            "gated_conformal_pr_auc": gated_conformal_metrics["pr_auc"],
            "gated_conformal_normal_fpr": normal_fpr(gated_conformal_pred, y_test),
            "delta_f1_gated_or_minus_ae": gated_or_metrics["f1"] - ae_metrics["f1"],
            "delta_recall_gated_or_minus_ae": gated_or_metrics["recall"] - ae_metrics["recall"],
            "delta_seen_recall_gated_or_minus_ae": (
                masked_recall(gated_or_pred, seen_mask) - masked_recall(ae_pred, seen_mask)
            ),
            "delta_unseen_recall_gated_or_minus_ae": (
                masked_recall(gated_or_pred, unseen_mask) - masked_recall(ae_pred, unseen_mask)
            ),
            "reconstruction_threshold": recon_threshold,
            "latent_probability_threshold": latent_threshold,
            "conformal_score_threshold": conformal_threshold,
        }

        for fpr_target in target_fprs:
            threshold = float(np.quantile(score_gated_cal_normal, 1.0 - fpr_target))
            pred_fixed_fpr = (score_gated >= threshold).astype(int)
            suffix = f"{fpr_target:.3f}".replace(".", "_")
            seed_row[f"gated_or_recall_at_fpr_{suffix}"] = float(
                precision_recall_fscore_support(y_test, pred_fixed_fpr, average="binary", zero_division=0)[1]
            )
            seed_row[f"gated_or_actual_fpr_at_fpr_{suffix}"] = normal_fpr(pred_fixed_fpr, y_test)

        seed_rows.append(seed_row)

        attack_df = pd.DataFrame(
            {
                "attack_type": test_attack_labels,
                "is_attack": y_test,
                "pred_ae": ae_pred,
                "pred_gated_or": gated_or_pred,
            }
        )
        attack_df = attack_df[attack_df["is_attack"] == 1]
        if not attack_df.empty:
            for attack_type, group in attack_df.groupby("attack_type"):
                ae_recall_attack = float(group["pred_ae"].mean())
                gated_recall_attack = float(group["pred_gated_or"].mean())
                attack_rows.append(
                    {
                        "seed": int(seed),
                        "attack_type": str(attack_type),
                        "count": int(group.shape[0]),
                        "ae_recall": ae_recall_attack,
                        "gated_or_recall": gated_recall_attack,
                        "delta_recall": gated_recall_attack - ae_recall_attack,
                    }
                )

        if external_df is not None:
            X_external_df = build_external_feature_frame(external_df, feature_names)
            X_external = transform_features(X_external_df, ordinal_encoder, scaler, feature_names, cat_cols)

            external_recon = autoencoder.predict(X_external, verbose=0)
            external_recon_error = np.mean((X_external - external_recon) ** 2, axis=1)
            external_ae_pred = (external_recon_error >= recon_threshold).astype(int)

            external_latent = encoder.predict(X_external, verbose=0)
            external_latent_prob = latent_clf.predict_proba(external_latent)[:, 1]
            external_gated_pred = np.where(
                (external_ae_pred == 1) | (external_latent_prob >= latent_threshold),
                1,
                0,
            )
            external_score_gated = np.maximum(
                external_recon_error / max(recon_threshold, eps),
                external_latent_prob / max(latent_threshold, eps),
            )

            external_row = {
                "seed": int(seed),
                "external_rows": int(X_external.shape[0]),
                "external_ae_anomaly_rate": float(external_ae_pred.mean()),
                "external_gated_or_anomaly_rate": float(external_gated_pred.mean()),
            }

            if y_external is not None and y_external.shape[0] == X_external.shape[0]:
                external_ae_metrics = evaluate_predictions(y_external, external_ae_pred, external_recon_error)
                external_gated_metrics = evaluate_predictions(y_external, external_gated_pred, external_score_gated)
                external_row.update(
                    {
                        "external_ae_f1": external_ae_metrics["f1"],
                        "external_ae_precision": external_ae_metrics["precision"],
                        "external_ae_recall": external_ae_metrics["recall"],
                        "external_gated_or_f1": external_gated_metrics["f1"],
                        "external_gated_or_precision": external_gated_metrics["precision"],
                        "external_gated_or_recall": external_gated_metrics["recall"],
                        "external_delta_f1_gated_or_minus_ae": external_gated_metrics["f1"] - external_ae_metrics["f1"],
                        "external_delta_recall_gated_or_minus_ae": (
                            external_gated_metrics["recall"] - external_ae_metrics["recall"]
                        ),
                    }
                )

            external_rows.append(external_row)

            if args.save_external_predictions:
                prediction_df = external_df.copy()
                prediction_df["pred_ae"] = external_ae_pred
                prediction_df["pred_gated_or"] = external_gated_pred
                prediction_df["reconstruction_error"] = external_recon_error
                prediction_df["latent_probability"] = external_latent_prob
                prediction_df.to_csv(output_dir / f"external_predictions_seed_{seed}.csv", index=False)

    seed_metrics_df = pd.DataFrame(seed_rows).sort_values("seed")
    seed_metrics_path = output_dir / "publication_seed_metrics.csv"
    seed_metrics_df.to_csv(seed_metrics_path, index=False)

    paired_rows = []
    paired_specs = [
        ("f1", "ae_f1", "gated_or_f1", "higher_better"),
        ("recall", "ae_recall", "gated_or_recall", "higher_better"),
        ("precision", "ae_precision", "gated_or_precision", "higher_better"),
        ("seen_recall", "ae_seen_recall", "gated_or_seen_recall", "higher_better"),
        ("unseen_recall", "ae_unseen_recall", "gated_or_unseen_recall", "higher_better"),
        ("normal_fpr", "ae_normal_fpr", "gated_or_normal_fpr", "lower_better"),
    ]

    for metric_name, baseline_col, proposed_col, direction in paired_specs:
        baseline_values = seed_metrics_df[baseline_col].to_numpy(dtype=float)
        proposed_values = seed_metrics_df[proposed_col].to_numpy(dtype=float)
        delta = proposed_values - baseline_values
        ttest_pvalue, wilcoxon_pvalue = safe_paired_tests(baseline_values, proposed_values)

        if direction == "higher_better":
            better_ratio = float(np.mean(delta > 0))
        else:
            better_ratio = float(np.mean(delta < 0))

        paired_rows.append(
            {
                "metric": metric_name,
                "baseline_mean": float(np.mean(baseline_values)),
                "proposed_mean": float(np.mean(proposed_values)),
                "delta_mean": float(np.mean(delta)),
                "delta_std": float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0,
                "direction": direction,
                "better_seed_ratio": better_ratio,
                "ttest_pvalue": ttest_pvalue,
                "wilcoxon_pvalue": wilcoxon_pvalue,
            }
        )

    paired_df = pd.DataFrame(paired_rows)
    paired_path = output_dir / "publication_paired_significance.csv"
    paired_df.to_csv(paired_path, index=False)

    attack_summary_records: list[dict] = []
    attack_seed_path = output_dir / "publication_attack_type_by_seed.csv"
    attack_summary_path = output_dir / "publication_attack_type_summary.csv"
    if attack_rows:
        attack_seed_df = pd.DataFrame(attack_rows)
        attack_seed_df.to_csv(attack_seed_path, index=False)

        attack_summary_df = (
            attack_seed_df.groupby("attack_type", as_index=False)
            .agg(
                count=("count", "mean"),
                seeds=("seed", "nunique"),
                ae_recall_mean=("ae_recall", "mean"),
                gated_or_recall_mean=("gated_or_recall", "mean"),
                delta_recall_mean=("delta_recall", "mean"),
                delta_recall_std=("delta_recall", "std"),
            )
            .sort_values("delta_recall_mean", ascending=False)
        )
        attack_summary_df.to_csv(attack_summary_path, index=False)
        attack_summary_records = attack_summary_df.to_dict(orient="records")

    external_summary_records: list[dict] = []
    external_path_csv = output_dir / "publication_external_metrics_by_seed.csv"
    if external_rows:
        external_df_metrics = pd.DataFrame(external_rows).sort_values("seed")
        external_df_metrics.to_csv(external_path_csv, index=False)
        external_summary_records = external_df_metrics.to_dict(orient="records")

    report = {
        "pipeline_name": "Publication Evaluation - DL Gated Ensemble",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "NSL-KDD",
        "train_path": str(args.train_path),
        "test_path": str(args.test_path),
        "config": {
            "seeds": seeds,
            "sample_fraction": float(args.sample_fraction),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "latent_dim": int(args.latent_dim),
            "normal_val_fraction": float(args.normal_val_fraction),
            "latent_cal_fraction": float(args.latent_cal_fraction),
            "reconstruction_quantile": float(args.reconstruction_quantile),
            "latent_quantile": float(args.latent_quantile),
            "alpha_conformal": float(args.alpha_conformal),
            "target_fprs": target_fprs,
            "external_path": str(args.external_path) if args.external_path else None,
            "external_label_column": args.external_label_column,
            "external_mapping_json": str(args.external_mapping_json) if args.external_mapping_json else None,
        },
        "seed_metrics_mean": seed_metrics_df.mean(numeric_only=True).to_dict(),
        "seed_metrics_std": seed_metrics_df.std(numeric_only=True).to_dict(),
        "paired_significance": paired_rows,
        "attack_type_summary_top_gain": attack_summary_records[:10],
        "attack_type_summary_top_drop": list(reversed(attack_summary_records[-10:])),
        "external_metrics_by_seed": external_summary_records,
    }

    with open(output_dir / "publication_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Saved publication seed metrics to {seed_metrics_path}")
    print(f"Saved publication paired significance to {paired_path}")
    if attack_rows:
        print(f"Saved publication attack-type metrics to {attack_seed_path}")
        print(f"Saved publication attack-type summary to {attack_summary_path}")
    if external_rows:
        print(f"Saved external publication metrics to {external_path_csv}")

    if not seed_metrics_df.empty:
        print(
            "[publication] "
            f"AE F1 mean={seed_metrics_df['ae_f1'].mean():.4f} | "
            f"Gated-OR F1 mean={seed_metrics_df['gated_or_f1'].mean():.4f} | "
            f"delta={seed_metrics_df['delta_f1_gated_or_minus_ae'].mean():.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Network anomaly detection CLI (notebook-aligned)")

    parser.add_argument("--mode", choices=["train", "test", "colab"], help="Legacy mode (mapped to DL train/test).")
    parser.add_argument("--data_path", type=str, help="Legacy path used with --mode.")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints", help="Legacy checkpoint root used with --mode.")
    parser.add_argument("--model_name", type=str, default="autoencoder", help="Legacy model folder name.")

    subparsers = parser.add_subparsers(dest="command")

    p_download = subparsers.add_parser("download-data", help="Download NSL-KDD train/test files.")
    p_download.add_argument("--data_dir", type=str, default="./data")

    p_train_ml = subparsers.add_parser("train-ml", help="Train ML anomaly models (IF, LOF, OCSVM).")
    p_train_ml.add_argument("--train_path", type=str, default="./data/KDDTrain+.txt")
    p_train_ml.add_argument("--test_path", type=str, default="./data/KDDTest+.txt")
    p_train_ml.add_argument("--output_dir", type=str, default="./checkpoints/ml_models_major")
    p_train_ml.add_argument("--if_estimators", type=int, default=400)
    p_train_ml.add_argument("--lof_neighbors", type=int, default=35)
    p_train_ml.add_argument("--ocsvm_nu", type=float, default=0.05)
    p_train_ml.add_argument("--seed", type=int, default=42)

    p_test_ml = subparsers.add_parser("test-ml", help="Run inference using trained ML artifacts.")
    p_test_ml.add_argument("--input_path", type=str, default="./data/KDDTest+.txt")
    p_test_ml.add_argument("--model_dir", type=str, default="./checkpoints/ml_models_major")
    p_test_ml.add_argument("--model_name", type=str, default=None, help="Optional: isolation_forest/local_outlier_factor/oneclass_svm")
    p_test_ml.add_argument("--output_csv", type=str, default="./ml_test_predictions.csv")

    p_train_dl = subparsers.add_parser("train-dl", help="Train autoencoder + latent classifier.")
    p_train_dl.add_argument("--train_path", type=str, default="./data/KDDTrain+.txt")
    p_train_dl.add_argument("--test_path", type=str, default="./data/KDDTest+.txt")
    p_train_dl.add_argument("--output_dir", type=str, default="./checkpoints/autoencoder_major")
    p_train_dl.add_argument("--epochs", type=int, default=30)
    p_train_dl.add_argument("--batch_size", type=int, default=256)
    p_train_dl.add_argument("--latent_dim", type=int, default=32)
    p_train_dl.add_argument("--threshold_percentile", type=float, default=95.0)
    p_train_dl.add_argument("--seed", type=int, default=42)

    p_test_dl = subparsers.add_parser("test-dl", help="Run inference using trained DL artifacts.")
    p_test_dl.add_argument("--input_path", type=str, default="./data/KDDTest+.txt")
    p_test_dl.add_argument("--model_dir", type=str, default="./checkpoints/autoencoder_major")
    p_test_dl.add_argument("--output_csv", type=str, default="./dl_test_predictions.csv")

    p_train_prod = subparsers.add_parser(
        "train-prod",
        help="Train production-grade hybrid ensemble (denoising AE + latent IF/LR + calibrated stacking + conformal threshold).",
    )
    p_train_prod.add_argument("--train_path", type=str, default="./data/KDDTrain+.txt")
    p_train_prod.add_argument("--test_path", type=str, default="./data/KDDTest+.txt")
    p_train_prod.add_argument("--output_dir", type=str, default="./checkpoints/production_hybrid")
    p_train_prod.add_argument("--epochs", type=int, default=25)
    p_train_prod.add_argument("--batch_size", type=int, default=256)
    p_train_prod.add_argument("--latent_dim", type=int, default=48)
    p_train_prod.add_argument("--noise_std", type=float, default=0.02)
    p_train_prod.add_argument("--if_estimators", type=int, default=400)
    p_train_prod.add_argument("--calibration_fraction", type=float, default=0.30)
    p_train_prod.add_argument("--threshold_fraction", type=float, default=0.40)
    p_train_prod.add_argument("--alpha_conformal", type=float, default=0.05)
    p_train_prod.add_argument("--recon_percentile", type=float, default=95.0)
    p_train_prod.add_argument("--if_percentile", type=float, default=95.0)
    p_train_prod.add_argument("--seed", type=int, default=42)

    p_test_prod = subparsers.add_parser("test-prod", help="Run inference using production-grade hybrid artifacts.")
    p_test_prod.add_argument("--input_path", type=str, default="./data/KDDTest+.txt")
    p_test_prod.add_argument("--model_dir", type=str, default="./checkpoints/production_hybrid")
    p_test_prod.add_argument("--output_csv", type=str, default="./prod_test_predictions.csv")

    p_eval_pub = subparsers.add_parser(
        "eval-publication",
        help=(
            "Run publication-grade evaluation for DL gated ensemble "
            "(multi-seed robustness, paired significance, and optional external validation)."
        ),
    )
    p_eval_pub.add_argument("--train_path", type=str, default="./data/KDDTrain+.txt")
    p_eval_pub.add_argument("--test_path", type=str, default="./data/KDDTest+.txt")
    p_eval_pub.add_argument("--output_dir", type=str, default="./checkpoints/autoencoder_major_project_gated")
    p_eval_pub.add_argument("--seeds", type=str, default="7,21,42,84,126")
    p_eval_pub.add_argument("--sample_fraction", type=float, default=1.0)
    p_eval_pub.add_argument("--epochs", type=int, default=30)
    p_eval_pub.add_argument("--batch_size", type=int, default=512)
    p_eval_pub.add_argument("--latent_dim", type=int, default=32)
    p_eval_pub.add_argument("--normal_val_fraction", type=float, default=0.2)
    p_eval_pub.add_argument("--latent_cal_fraction", type=float, default=0.2)
    p_eval_pub.add_argument("--reconstruction_quantile", type=float, default=0.95)
    p_eval_pub.add_argument("--latent_quantile", type=float, default=0.95)
    p_eval_pub.add_argument("--alpha_conformal", type=float, default=0.05)
    p_eval_pub.add_argument("--target_fprs", type=str, default="0.01,0.05,0.10")
    p_eval_pub.add_argument("--external_path", type=str, default=None)
    p_eval_pub.add_argument("--external_label_column", type=str, default="label")
    p_eval_pub.add_argument("--external_mapping_json", type=str, default=None)
    p_eval_pub.add_argument("--save_external_predictions", action="store_true")
    p_eval_pub.add_argument("--verbose", type=int, default=0, help="Keras fit verbosity (0 or 1).")

    return parser


def run_legacy(args) -> None:
    if args.mode == "colab":
        return
    if args.mode not in {"train", "test"}:
        raise ValueError("Unsupported legacy mode. Use --help for available commands.")

    ckpt_dir = Path(args.ckpt_path) / args.model_name
    data_path = Path(args.data_path) if args.data_path else None
    if not data_path:
        raise ValueError("--data_path is required when using legacy --mode.")

    if args.mode == "train":
        legacy_test_path = data_path.parent / "KDDTest+.txt"
        if not legacy_test_path.exists():
            legacy_test_path = data_path
        train_args = argparse.Namespace(
            train_path=str(data_path),
            test_path=str(legacy_test_path),
            output_dir=str(ckpt_dir),
            epochs=30,
            batch_size=256,
            latent_dim=32,
            threshold_percentile=95.0,
            seed=42,
        )
        run_train_dl(train_args)
    elif args.mode == "test":
        test_args = argparse.Namespace(
            input_path=str(data_path),
            model_dir=str(ckpt_dir),
            output_csv="test_output.csv",
        )
        run_test_dl(test_args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode:
        run_legacy(args)
        return

    if args.command == "download-data":
        train_path, test_path = download_nsl_kdd(Path(args.data_dir))
        print(f"Dataset ready:\n- {train_path}\n- {test_path}")
    elif args.command == "train-ml":
        run_train_ml(args)
    elif args.command == "test-ml":
        run_test_ml(args)
    elif args.command == "train-dl":
        run_train_dl(args)
    elif args.command == "test-dl":
        run_test_dl(args)
    elif args.command == "train-prod":
        run_train_prod(args)
    elif args.command == "test-prod":
        run_test_prod(args)
    elif args.command == "eval-publication":
        run_eval_publication(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
