import sys
import json
import pandas as pd
from logreg_train import standardize

def load_model(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 logreg_predict.py dataset_test.csv weights.json")
        sys.exit(1)
    try:
        df = pd.read_csv(sys.argv[1])
        model = load_model(sys.argv[2])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    features = model["features"]
    missing = [c for c in features + ["Index"] if c not in df.columns]
    if missing:
        print(f"Error: missing columns in dataset: {missing}")
        sys.exit(1)

    X = standardize(df, features, model["means"], model["stds"])
    houses = list(model["weights"].keys())


if __name__ == "__main__":
    main()