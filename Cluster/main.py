#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import math
import csv

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
    
    meals = insulin_df[(insulin_df[carb_col].notna()) & (insulin_df[carb_col] > 0)]
    meal_times = sorted(meals['Time'].tolist())
    
    meal_segments = []
    carb_values = []
    
    used_until = pd.NaT
    tolerance = timedelta(minutes=5)
    
    for tm in meal_times:
        if pd.notnull(used_until) and tm < used_until:
            continue
            
        meal_row = meals[meals['Time'] == tm].iloc[0]
        carb_value = meal_row[carb_col]
        
        start_time = tm
        end_time = tm + timedelta(hours=2)
        
        window_df = cgm_df[(cgm_df['Time'] >= start_time) & (cgm_df['Time'] <= end_time)]
        required_points = 30  
        
        if len(window_df) >= required_points:
            window_df = window_df.sort_values('Time')
            segment = window_df.iloc[:required_points][gl_col].values
            
            try:
                segment = segment.astype(float)
                segment = handle_missing(segment)
                
                if segment is not None and len(segment) == required_points:
                    meal_segments.append(segment)
                    carb_values.append(carb_value)
                    used_until = end_time
            except:
                continue
    
    return np.array(meal_segments), np.array(carb_values)

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

def compute_sse(data, labels, centers=None):
    n_clusters = len(np.unique(labels))
    if centers is None:
        centers = np.zeros((n_clusters, data.shape[1]))
        for i in range(n_clusters):
            if np.sum(labels == i) > 0:
                centers[i] = np.mean(data[labels == i], axis=0)
    
    sse = 0
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            cluster_center = centers[i]
            cluster_sse = np.sum(np.square(cluster_points - cluster_center))
            sse += cluster_sse
    
    return sse

def compute_entropy(confusion_matrix):
    total_samples = np.sum(confusion_matrix)
    if total_samples == 0:
        return 0
        
    entropy = 0
    
    for i in range(confusion_matrix.shape[0]): 
        cluster_size = np.sum(confusion_matrix[i])
        if cluster_size > 0:
            cluster_entropy = 0
            for j in range(confusion_matrix.shape[1]): 
                p_ij = confusion_matrix[i, j] / cluster_size
                if p_ij > 0:
                    cluster_entropy -= p_ij * np.log2(p_ij)
            entropy += (cluster_size / total_samples) * cluster_entropy
    
    return entropy

def compute_purity(confusion_matrix):
    total_samples = np.sum(confusion_matrix)
    if total_samples == 0:
        return 0
        
    purity = 0
    
    for i in range(confusion_matrix.shape[0]):  
        if np.sum(confusion_matrix[i]) > 0:
            max_class = np.max(confusion_matrix[i])
            purity += max_class
    
    return purity / total_samples

def compute_confusion_matrix(true_labels, cluster_labels, n_bins, n_clusters):
    confusion_matrix = np.zeros((n_clusters, n_bins))
    
    for i in range(len(true_labels)):
        bin_idx = true_labels[i]
        cluster_idx = cluster_labels[i]
        confusion_matrix[cluster_idx, bin_idx] += 1
    
    return confusion_matrix

def main():
    insulin_df = pd.read_csv("InsulinData.csv", low_memory=False)
    cgm_df = pd.read_csv("CGMData.csv", low_memory=False)
    
    meal_data, carb_amounts = extract_meal_data(insulin_df, cgm_df)
    

    if len(carb_amounts) == 0:
        print("Error: No valid meal data found that meets criteria")
        
        results = [0, 0, 0, 0, 0, 0]
        with open('Result.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(results)
        return
    
    min_carbs = np.min(carb_amounts)
    max_carbs = np.max(carb_amounts)
    bin_size = 20
    n_bins = math.ceil((max_carbs - min_carbs) / bin_size)
    
    n_bins = max(2, n_bins)
    
    bin_assignments = np.floor((carb_amounts - min_carbs) / bin_size).astype(int)
    
    feature_matrix = extract_features(meal_data)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled_features)
    
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=min(5, len(scaled_features)-1))
    neighbors.fit(scaled_features)
    distances, _ = neighbors.kneighbors(scaled_features)
    
    eps = np.mean(distances[:, -1]) * 0.75
    min_samples = min(5, len(scaled_features) // 10)
    min_samples = max(2, min_samples)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(scaled_features)
    
    if -1 in dbscan_labels:
        dbscan_labels = dbscan_labels.copy()  
        dbscan_labels[dbscan_labels == -1] = max(dbscan_labels) + 1
    
    n_kmeans_clusters = len(np.unique(kmeans_labels))
    n_dbscan_clusters = len(np.unique(dbscan_labels))
    
    kmeans_sse = compute_sse(scaled_features, kmeans_labels, kmeans.cluster_centers_)
    
    dbscan_centers = np.zeros((n_dbscan_clusters, scaled_features.shape[1]))
    for i in range(n_dbscan_clusters):
        cluster_points = scaled_features[dbscan_labels == i]
        if len(cluster_points) > 0:
            dbscan_centers[i] = np.mean(cluster_points, axis=0)
    
    dbscan_sse = compute_sse(scaled_features, dbscan_labels, dbscan_centers)

    kmeans_confusion = compute_confusion_matrix(bin_assignments, kmeans_labels, n_bins, n_kmeans_clusters)
    dbscan_confusion = compute_confusion_matrix(bin_assignments, dbscan_labels, n_bins, n_dbscan_clusters)

    kmeans_entropy = compute_entropy(kmeans_confusion)
    dbscan_entropy = compute_entropy(dbscan_confusion)
    kmeans_purity = compute_purity(kmeans_confusion)
    dbscan_purity = compute_purity(dbscan_confusion)

    results = [kmeans_sse, dbscan_sse, kmeans_entropy, dbscan_entropy, kmeans_purity, dbscan_purity]
    with open('Result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results)

if __name__ == "__main__":
    main()
