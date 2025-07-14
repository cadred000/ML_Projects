#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle

def extract_features(series):
    mean_val = np.mean(series)
    std_val = np.std(series)
    min_val = np.min(series)
    max_val = np.max(series)
    slope, _ = np.polyfit(np.arange(len(series)), series, 1)
    return [mean_val, std_val, min_val, max_val, slope]

def main():
    test_df = pd.read_csv("test.csv", header=None)
    test_array = test_df.values
    X_test = []
    for row in test_array:
        features = extract_features(row)
        X_test.append(features)
    X_test = np.array(X_test)
    with open("trained_model.pkl", "rb") as f:
        clf = pickle.load(f)
    predictions = clf.predict(X_test)
    pd.DataFrame(predictions).to_csv("Result.csv", header=False, index=False)

if __name__ == '__main__':
    main()

