from pathlib import Path
from importlib import import_module

try:
    st = import_module("streamlit")
except ModuleNotFoundError as exc:
    if exc.name != "streamlit":
        raise
    raise SystemExit("Streamlit is required. Install it with `python -m pip install streamlit`.") from exc

try:
    joblib = import_module("joblib")
except ModuleNotFoundError as exc:
    if exc.name != "joblib":
        raise
    raise SystemExit("joblib is required. Install it with `python -m pip install joblib`.") from exc


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fraud_rf_model.joblib"
AMOUNT_SCALER_PATH = ROOT / "amount_scaler.joblib"
TIME_SCALER_PATH = ROOT / "time_scaler.joblib"
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + [
    "Scaled_Amount",
    "Scaled_Time",
]


st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")
st.title("Credit Card Fraud Detection")
st.write(
    "Enter transaction details to estimate whether the transaction is fraudulent."
)


@st.cache_resource
def load_artifacts():
    return (
        joblib.load(MODEL_PATH),
        joblib.load(AMOUNT_SCALER_PATH),
        joblib.load(TIME_SCALER_PATH),
    )


missing_artifacts = [
    path.name
    for path in (MODEL_PATH, AMOUNT_SCALER_PATH, TIME_SCALER_PATH)
    if not path.is_file()
]
if missing_artifacts:
    st.error(
        "Model files are missing: "
        + ", ".join(missing_artifacts)
        + ". Run `python train_model.py --csv creditcard.csv`, then include the "
        "generated `.joblib` files in the repository before deploying."
    )
    st.stop()

model, amount_scaler, time_scaler = load_artifacts()

with st.form("transaction"):
    time_column, amount_column = st.columns(2)
    with time_column:
        raw_time = st.number_input(
            "Transaction time (seconds)", min_value=0.0, value=100.0, step=1.0
        )
    with amount_column:
        raw_amount = st.number_input(
            "Transaction amount ($)", min_value=0.0, value=50.0, step=0.01
        )

    feature_columns = st.columns(3)
    feature_values = {}
    for index in range(1, 29):
        with feature_columns[(index - 1) % len(feature_columns)]:
            feature_values[f"V{index}"] = st.number_input(
                f"V{index}", value=0.0, step=0.1
            )

    submitted = st.form_submit_button("Check transaction", type="primary")

if submitted:
    feature_values["Scaled_Amount"] = amount_scaler.transform([[raw_amount]])[0][0]
    feature_values["Scaled_Time"] = time_scaler.transform([[raw_time]])[0][0]
    transaction = [[feature_values[column] for column in FEATURE_COLUMNS]]

    prediction = model.predict(transaction)[0]
    fraud_probability = model.predict_proba(transaction)[0][
        list(model.classes_).index(1)
    ]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"Potential fraud detected ({fraud_probability:.1%} probability).")
    else:
        st.success(f"No fraud detected ({fraud_probability:.1%} fraud probability).")
