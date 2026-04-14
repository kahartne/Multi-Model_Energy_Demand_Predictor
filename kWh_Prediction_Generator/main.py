import argparse
import pandas as pd
import joblib
import sys
from pathlib import Path

def load_model(model_name):
    # Map simple names to trained/saved .pkl files
    model_files = {
        "linear": "linear_model.pkl",
        "rf": "rf_model.pkl",
        "gbr": "gbr_model.pkl",
        "xgb": "xgb_model.pkl",
        "stack": "stack_model.pkl"
    }
    if model_name not in model_files:
        raise ValueError(f"Unknown model: {model_name}")

    # Make path relative to THIS script’s folder
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / model_files[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Working directory was: {Path.cwd()}"
        )

    return joblib.load(model_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="linear|rf|gbr|xgb|stack")
    parser.add_argument("--weather", required=True, help="Path to weather.csv")
    parser.add_argument("--out", required=True, help="Path to output predictions.csv")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Load weather features
    df = pd.read_csv(args.weather)

    # Keep columns in the same order used during training
    feature_cols = ['temperature', 'humidity', 'cloud_cover',
                    'hour', 'day_of_week', 'school_factor']

    X = df[feature_cols].values

    # Predict
    y_pred = model.predict(X)

    # Write baseline predictions
    out = pd.DataFrame({
        "slot": df["slot"],
        "baseline_kW": y_pred
    })
    out.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

if __name__ == "__main__":
    main()