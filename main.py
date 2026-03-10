import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
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


def fit_preprocessing(X_train: pd.DataFrame) -> tuple[OrdinalEncoder, StandardScaler, list[str], list[str]]:
    X_train = X_train.copy()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [column for column in X_train.columns if column not in cat_cols]

    for column in cat_cols:
        X_train[column] = X_train[column].fillna("Unknown").astype(str)
    for column in num_cols:
        median = X_train[column].median()
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
        median = X[column].median()
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
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


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


def run_test_dl(args) -> None:
    model_dir = Path(args.model_dir)
    input_df = load_nsl_file(Path(args.input_path))
    X_input_df, _ = split_features(input_df)

    scaler = joblib.load(model_dir / "standard_scaler.joblib")
    ordinal_encoder = joblib.load(model_dir / "ordinal_encoder.joblib")
    feature_names = joblib.load(model_dir / "feature_names.joblib")
    cat_cols = joblib.load(model_dir / "categorical_columns.joblib")
    latent_clf = joblib.load(model_dir / "latent_classifier.joblib")
    autoencoder = load_model(model_dir / "autoencoder_major.keras")

    with open(model_dir / "dl_metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)
    threshold = float(metadata["reconstruction_threshold"])

    X_scaled = transform_features(X_input_df, ordinal_encoder, scaler, feature_names, cat_cols)
    recon = autoencoder.predict(X_scaled, verbose=0)
    recon_error = np.mean((X_scaled - recon) ** 2, axis=1)
    ae_pred = (recon_error >= threshold).astype(int)

    latent_layer = autoencoder.get_layer("latent").output
    encoder = Model(autoencoder.input, latent_layer)
    latent = encoder.predict(X_scaled, verbose=0)
    clf_score = latent_clf.predict_proba(latent)[:, 1]
    clf_pred = (clf_score >= 0.5).astype(int)

    output_df = input_df.copy()
    output_df["pred_ae"] = ae_pred
    output_df["pred_latent_clf"] = clf_pred
    output_df["reconstruction_error"] = recon_error
    output_df["latent_clf_score"] = clf_score
    output_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


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
        train_args = argparse.Namespace(
            train_path=str(data_path),
            test_path=str(data_path),
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()