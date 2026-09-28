import sys
import pandas as pd

FEATURES = [
    "Astronomy", "Herbology", "Divination", "Muggle Studies",
    "Ancient Runes", "History of Magic", "Transfiguration",
    "Potions", "Charms", "Flying",
]
HOUSES = ["Gryffindor" , "Hufflepuff", "Ravenclaw", "Slytherin"]


def mean(values):
    total = 0.0
    count = 0
    for v in values:
        if v == v:
            total += v
            count += 1
    return total / count


def std(values, mu):
    total = 0.0
    count = 0
    for v in values:
        if v == v:
            total += (v - mu) ** 2
            count += 1
    return (total / (count - 1)) ** 0.5


def standardize(df, features, means, stds):
    X = []
    for _, row in df.iterrows():
        x = [1.0]
        for f in features:
            v = row[f]
            if v != v:
                x.append(0.0)
            else:
                x.append((v - means[f]) / stds[f])
        X.append(x)
    return X


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 logreg_train.py dataset_train.csv")
        sys.exit(1)
    try:
        df = pd.read_csv(sys.argv[1])
    except Exception as e:
        print(f"Error: could not read {sys.argv[1]}: {e}")
        sys.exit(1)

    missing = [c for c in FEATURES + ["Hogwarts House"] if c not in df.columns]
    if missing:
        print(f"Error: missing columns in dataset: {missing}")
        sys.exit(1)

    means = {f: mean(df[f]) for f in FEATURES}
    stds = {f: std(df[f], means[f]) for f in FEATURES}
    X = standardize(df, FEATURES, means, stds)

if __name__ == "__main__":
    main()