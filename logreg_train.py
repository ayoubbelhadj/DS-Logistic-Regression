import sys
import pandas as pd

FEATURES = [
    "Astronomy", "Herbology", "Divination", "Muggle Studies",
    "Ancient Runes", "History of Magic", "Transfiguration",
    "Potions", "Charms", "Flying",
]
HOUSES = ["Gryffindor" , "Hufflepuff", "Ravenclaw", "Slytherin"]

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


if __name__ == "__main__":
    main()