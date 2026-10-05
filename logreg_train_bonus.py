import sys
import pandas as pd
import json
import math
import random

FEATURES = [
    "Astronomy", "Herbology", "Divination", "Muggle Studies",
    "Ancient Runes", "History of Magic", "Transfiguration",
    "Potions", "Charms", "Flying",
]
HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

OPTIMIZERS = {
    "batch": {"learning_rate": 0.5, "epochs": 1000, "batch_size": None},
    "sgd": {"learning_rate": 0.01, "epochs": 10, "batch_size": 1},
    "minibatch": {"learning_rate": 0.1, "epochs": 50, "batch_size": 32},
}


def error(msg):
    print(f"Error: {msg}")
    sys.exit(1)


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


def gradient(theta, X, y, indexes):
    m = len(indexes)
    n = len(theta)
    grad = [0.0] * n
    for i in indexes:
        error = hypothesis(theta, X[i]) - y[i]
        for j in range(n):
            grad[j] += error * X[i][j]
    for j in range(n):
        grad[j] /= m
    return grad


def train(X, y, learning_rate, epochs, batch_size):
    m = len(X)
    if batch_size is None:
        batch_size = m
    theta = [0.0] * len(X[0])
    indexes = list(range(m))
    print_every = max(1, epochs // 10)
    for epoch in range(epochs):
        if batch_size < m:
            random.shuffle(indexes)
        for start in range(0, m, batch_size):
            batch = indexes[start:start + batch_size]
            grad = gradient(theta, X, y, batch)
            for j in range(len(theta)):
                theta[j] -= learning_rate * grad[j]
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch}: cost = {cost(theta, X, y):.6f}")
    return theta


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python3 logreg_train.py dataset_train.csv [batch|sgd|minibatch]")
        sys.exit(1)
    optimizer = sys.argv[2] if len(sys.argv) == 3 else "batch"
    if optimizer not in OPTIMIZERS:
        error(f"unknown optimizer '{optimizer}'. Choose from: batch, sgd, minibatch")
    config = OPTIMIZERS[optimizer]
    try:
        df = pd.read_csv(sys.argv[1])
        if len(df) == 0:
            error("dataset is empty")
    except Exception as e:
        error(f"could not read {sys.argv[1]}: {e}")

    missing = [c for c in FEATURES + ["Hogwarts House"] if c not in df.columns]
    if missing:
        error(f"missing columns in dataset: {missing}")

    means = {f: mean(df[f]) for f in FEATURES}
    stds = {f: std(df[f], means[f]) for f in FEATURES}
    X = standardize(df, FEATURES, means, stds)

    random.seed(42)
    print(f"Optimizer: {optimizer} {config}")
    weights = {}
    for house in HOUSES:
        print(f"Training {house} vs all")
        y = [1 if h == house else 0 for h in df["Hogwarts House"]]
        weights[house] = train(X, y, config["learning_rate"],
                                config["epochs"], config["batch_size"])

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