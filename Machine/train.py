#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
import pickle
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

def handle_missing(segment, threshold=5):
    missing = np.sum(pd.isnull(segment))
    if missing > threshold:
        return None
    if missing > 0:
        s = pd.Series(segment)
        s = s.interpolate(limit_direction='both')
        return s.values
    return segment

def extract_meal_data(insulin_df, cgm_df):
    insulin_df['Time'] = pd.to_datetime(insulin_df['Time'], errors='coerce')
    cgm_df['Time'] = pd.to_datetime(cgm_df['Time'], errors='coerce')
    carb_col = "BWZ Carb Input (grams)"
    gl_col = "Sensor Glucose (mg/dL)"
    meals = insulin_df[(insulin_df[carb_col].notnull()) & (insulin_df[carb_col] != 0)]
    meal_times = sorted(meals['Time'].tolist())
    meal_segments = []
    used_until = pd.NaT
    tolerance = timedelta(minutes=5)
    for tm in meal_times:
        if pd.notnull(used_until) and tm < used_until:
            continue
        future_meals = [t for t in meal_times if t > tm and t < tm + timedelta(hours=2)]
        if future_meals:
            exact_meal = any(abs(t - (tm + timedelta(hours=2))) <= tolerance for t in future_meals)
            if exact_meal:
                start_time = tm + timedelta(minutes=90)
                end_time = tm + timedelta(hours=4)
            else:
                continue
        else:
            start_time = tm - timedelta(minutes=30)
            end_time = tm + timedelta(hours=2)
        window_df = cgm_df[(cgm_df['Time'] >= start_time) & (cgm_df['Time'] <= end_time)]
        required_points = int((end_time - start_time).total_seconds() / 300) + 1
        if len(window_df) >= required_points:
            segment = window_df.iloc[:required_points][gl_col].values.astype(float)
            segment = handle_missing(segment)
            if segment is not None and len(segment) == required_points:
                meal_segments.append(segment)
                used_until = end_time
    return np.array(meal_segments)

def extract_no_meal_data(insulin_df, cgm_df):
    insulin_df['Time'] = pd.to_datetime(insulin_df['Time'], errors='coerce')
    cgm_df['Time'] = pd.to_datetime(cgm_df['Time'], errors='coerce')
    carb_col = "BWZ Carb Input (grams)"
    gl_col = "Sensor Glucose (mg/dL)"
    meals = insulin_df[(insulin_df[carb_col].notnull()) & (insulin_df[carb_col] != 0)]
    meal_times = meals['Time'].tolist()
    cgm_df = cgm_df.sort_values('Time').reset_index(drop=True)
    no_meal_segments = []
    window_size = 24
    for i in range(len(cgm_df) - window_size + 1):
        window = cgm_df.iloc[i : i + window_size]
        ws = window['Time'].iloc[0]
        we = window['Time'].iloc[-1]
        if any((mt >= ws) and (mt <= we) for mt in meal_times):
            continue
        segment = window[gl_col].values.astype(float)
        segment = handle_missing(segment)
        if segment is not None and len(segment) == window_size:
            no_meal_segments.append(segment)
    return np.array(no_meal_segments)

def extract_features(data):
    features = []
    for sample in data:
        mean_val = np.mean(sample)
        std_val = np.std(sample)
        min_val = np.min(sample)
        max_val = np.max(sample)
        slope, _ = np.polyfit(np.arange(len(sample)), sample, 1)
        features.append([mean_val, std_val, min_val, max_val, slope])
    return np.array(features)

def main():
    insulin1 = pd.read_csv("InsulinData.csv")
    cgm1 = pd.read_csv("CGMData.csv")
    insulin2 = pd.read_csv("Insulin_patient2.csv")
    cgm2 = pd.read_csv("CGM_patient2.csv")
    meal_data1 = extract_meal_data(insulin1, cgm1)
    meal_data2 = extract_meal_data(insulin2, cgm2)
    if meal_data1.size and meal_data2.size:
        meal_data = np.vstack([meal_data1, meal_data2])
    elif meal_data1.size:
        meal_data = meal_data1
    else:
        meal_data = meal_data2
    no_meal_data1 = extract_no_meal_data(insulin1, cgm1)
    no_meal_data2 = extract_no_meal_data(insulin2, cgm2)
    if no_meal_data1.size and no_meal_data2.size:
        no_meal_data = np.vstack([no_meal_data1, no_meal_data2])
    elif no_meal_data1.size:
        no_meal_data = no_meal_data1
    else:
        no_meal_data = no_meal_data2
    X_meal = extract_features(meal_data)
    X_no_meal = extract_features(no_meal_data)
    y_meal = np.ones(X_meal.shape[0])
    y_no_meal = np.zeros(X_no_meal.shape[0])
    X = np.vstack([X_meal, X_no_meal])
    y = np.concatenate([y_meal, y_no_meal])
    clf = DecisionTreeClassifier(random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=kf)
    clf.fit(X, y)
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(clf, f)

if __name__ == '__main__':
    main()

