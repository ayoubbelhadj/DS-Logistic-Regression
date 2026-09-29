import sys
import pandas as pd
import json
import math

FEATURES = [
    "Astronomy", "Herbology", "Divination", "Muggle Studies",
    "Ancient Runes", "History of Magic", "Transfiguration",
    "Potions", "Charms", "Flying",
]
HOUSES = ["Gryffindor" , "Hufflepuff", "Ravenclaw", "Slytherin"]

LEARNING_RATE = 0.5
ITERATIONS = 1000

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


def sigmoid(z):
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def dot(theta, x):
    total = 0.0
    for i in range(len(theta)):
        total += theta[i] * x[i]
    return total


def hypothesis(theta, x):
    return sigmoid(dot(theta, x))


def cost(theta, X, y):
    m = len(X)
    eps = 1e-15
    total = 0.0
    for i in range(m):
        h = hypothesis(theta, X[i])
        h = min(max(h, eps), 1 - eps)
        total += y[i] * math.log(h) + (1 - y[i]) * math.log(1 - h)
    return -total / m


def gradient(theta, X, y):
    m = len(X)
    n = len(theta)
    grad = [0.0] * n
    for i in range(m):
        error = hypothesis(theta, X[i]) - y[i]
        for j in range(n):
            grad[j] += error * X[i][j]
    for j in range(n):
        grad[j] /= m
    return grad


def train(X, y, learning_rate, iterations):
    theta = [0.0] * len(X[0])
    for it in range(iterations):
        grad = gradient(theta, X, y)
        for j in range(len(theta)):
            theta[j] -= learning_rate * grad[j]
        if it % 100 == 0 or it == iterations - 1:
            print(f"  iteration {it}: cost = {cost(theta, X, y):.6f}")
    return theta


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

    weights = {}
    for house in HOUSES:
        print(f"Training {house} vs all")
        y = [1 if h == house else 0 for h in df["Hogwarts House"]]
        weights[house] = train(X, y, LEARNING_RATE, ITERATIONS)

    model = {
    "features": FEATURES,
    "means": means,
    "stds": stds,
    "weights": weights,
    }
    with open("weights.json", "w") as f:
        json.dump(model, f, indent=4)
    print("Weights saved to weights.json")

if __name__ == "__main__":
    main()