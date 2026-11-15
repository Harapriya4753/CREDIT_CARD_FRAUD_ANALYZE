import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
V_FEATURES = [f"V{i}" for i in range(1, 29)]
MODEL_FEATURES = V_FEATURES + ["Scaled_Amount", "Scaled_Time"]
REQUIRED_COLUMNS = set(V_FEATURES + ["Time", "Amount", "Class"])
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and save the credit-card fraud detection model."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "creditcard.csv",
        help="Path to the creditcard.csv dataset (default: project directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for the model and scaler artifacts (default: project directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.csv)

    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    data = data.drop_duplicates()
    if data.empty:
        raise ValueError("Dataset has no rows after duplicate removal.")
    if data["Class"].isna().any() or set(data["Class"].unique()) != {0, 1}:
        raise ValueError("The Class column must contain both labels 0 and 1.")
    if data[sorted(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")

    features = data[V_FEATURES + ["Time", "Amount"]]
    labels = data["Class"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    amount_scaler = StandardScaler().fit(x_train[["Amount"]])
    time_scaler = StandardScaler().fit(x_train[["Time"]])

    def prepare_features(rows):
        prepared = rows[V_FEATURES].copy()
        prepared["Scaled_Amount"] = amount_scaler.transform(rows[["Amount"]])[:, 0]
        prepared["Scaled_Time"] = time_scaler.transform(rows[["Time"]])[:, 0]
        return prepared[MODEL_FEATURES]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(prepare_features(x_train), y_train)

    predictions = model.predict(prepare_features(x_test))
    probabilities = model.predict_proba(prepare_features(x_test))[
        :, list(model.classes_).index(1)
    ]
    print(classification_report(y_test, predictions, digits=4))
    print(f"Average precision: {average_precision_score(y_test, probabilities):.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "fraud_rf_model.joblib")
    joblib.dump(amount_scaler, args.output_dir / "amount_scaler.joblib")
    joblib.dump(time_scaler, args.output_dir / "time_scaler.joblib")
    print(f"Model artifacts saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
