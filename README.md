# Credit Card Fraud Detection

## Prepare the model

Obtain the `creditcard.csv` dataset and install the pinned dependencies in a
Python 3.12 environment. The pinned NumPy version does not support Python 3.15.

On Windows PowerShell, create and install the environment with:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

If the Python launcher is unavailable, install Python 3.12 and create the
environment with `uv` instead:

```powershell
uv python install 3.12
uv venv .venv312 --python 3.12
uv pip install --python .venv312\Scripts\python.exe -r requirements.txt
```

Train the model and generate the artifacts expected by the app:

```powershell
.\.venv312\Scripts\python.exe train_model.py --csv creditcard.csv
```

The training script writes `fraud_rf_model.joblib`, `amount_scaler.joblib`, and
`time_scaler.joblib` into the project directory. Keep these artifacts together
with `app.py`; do not commit the source dataset unless you intend to publish it.
Training fits each scaler on the training split only.

## Run locally

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Using `python -m streamlit` avoids relying on the `streamlit` command being on
`PATH`. If the environment already exists, skip the `py -3.12 -m venv` step.

## Deploy to Streamlit Community Cloud

Push `app.py`, `requirements.txt`, `train_model.py`, and all three generated
`.joblib` artifacts to a GitHub repository. In Streamlit Community Cloud, create
an app from that repository and set the main file to `app.py`. The artifacts
must be generated with the pinned dependencies in `requirements.txt`, because
scikit-learn model files are version-sensitive.
