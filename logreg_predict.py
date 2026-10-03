import sys
import json
import pandas as pd
from logreg_train import standardize, hypothesis


def load_model(path):
    with open(path, "r") as f:
        return json.load(f)


def predict(X, weights, houses):
    predictions = []
    for x in X:
        best_house = None
        best_prob = -1.0
        for house in houses:
            prob = hypothesis(weights[house], x)
            if prob > best_prob:
                best_prob = prob
                best_house = house
        predictions.append(best_house)
    return predictions


def save_predictions(indexes, predictions, path):
    with open(path, "w") as f:
        f.write("Index,Hogwarts House\n")
        for i in range(len(predictions)):
            f.write(f"{indexes[i]},{predictions[i]}\n")


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
    predictions = predict(X, model["weights"], houses)
    save_predictions(list(df["Index"]), predictions, "houses.csv")
    print("Predictions saved to houses.csv")


if __name__ == "__main__":
    main()