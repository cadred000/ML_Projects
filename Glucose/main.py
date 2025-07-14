#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys

def parse_timestamps(df, date_col="Date", time_col="Time"):
    df[date_col] = df[date_col].astype(str)
    df[time_col] = df[time_col].astype(str)
    df["Timestamp"] = pd.to_datetime(
        df[date_col] + " " + df[time_col],
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce"
    )
    return df

def find_auto_mode_start_time(df_insulin, alarm_col="Alarm", event="AUTO MODE ACTIVE PLGM OFF"):
    rows = df_insulin[df_insulin[alarm_col] == event]
    if rows.empty:
        return None
    auto_start = rows["Timestamp"].min()
    return auto_start

def find_nearest_cgm_time(df_cgm, auto_time):
    if auto_time is None:
        return None
    mask = df_cgm["Timestamp"] >= auto_time
    if not mask.any():
        return None
    nearest_time = df_cgm.loc[mask, "Timestamp"].min()
    return nearest_time

def reindex_day_and_interpolate(df, cgm_col, min_valid=1):
    if df.empty:
        return pd.DataFrame(columns=["Timestamp", cgm_col, "Date"])
    
    df_days = []
    for date_val, group in df.groupby(df["Timestamp"].dt.date):
        valid_count = group[cgm_col].notna().sum()
        if valid_count < min_valid:
            continue
        group = group.copy()
        group["TimestampRounded"] = group["Timestamp"].dt.round("5min")
        # Restrict to the numeric column we care about for averaging
        group_numeric = group[[cgm_col]].copy()
        group_numeric.index = group["TimestampRounded"]
        start = pd.Timestamp(year=group["Timestamp"].iloc[0].year, month=group["Timestamp"].iloc[0].month, day=group["Timestamp"].iloc[0].day, hour=0, minute=0)
        end = start + pd.Timedelta(minutes=23*60+55)
        idx_range = pd.date_range(start, end, freq="5min")
        g2 = group_numeric.resample("5min").mean()
        g2 = g2.reindex(idx_range)
        g2[cgm_col] = g2[cgm_col].interpolate(method="linear", limit_direction="both", limit=12, inplace=False)
        g2["Date"] = date_val
        g2["Timestamp"] = g2.index
        df_days.append(g2.reset_index(drop=True))
    if not df_days:
        return pd.DataFrame(columns=["Timestamp", cgm_col, "Date"])
    concatenated = pd.concat(df_days, ignore_index=True)
    return concatenated

def compute_daily_metrics(df, cgm_col="Sensor Glucose (mg/dL)"):
    if df.empty:
        return [0]*18
    df["Day"] = df["Timestamp"].dt.date
    results = []
    for dval, group in df.groupby("Day"):
        total_expected = 288
        group = group.sort_values("Timestamp")
        group["Hour"] = group["Timestamp"].dt.hour
        overnight = group[group["Hour"] < 6][cgm_col]
        daytime = group[(group["Hour"] >= 6) & (group["Hour"] < 24)][cgm_col]
        whole_day = group[cgm_col]
        def compute_segment_metrics(segment):
            if len(segment) == 0:
                return [0, 0, 0, 0, 0, 0]
            hyper = np.round((segment > 180).sum() * 100 / total_expected, 4)
            hyper_crit = np.round((segment > 250).sum() * 100 / total_expected, 4)
            in_range = np.round(((segment >= 70) & (segment <= 180)).sum() * 100 / total_expected, 4)
            in_range_sec = np.round(((segment >= 70) & (segment <= 150)).sum() * 100 / total_expected, 4)
            hypo1 = np.round((segment < 70).sum() * 100 / total_expected, 4)
            hypo2 = np.round((segment < 54).sum() * 100 / total_expected, 4)
            return [hyper, hyper_crit, in_range, in_range_sec, hypo1, hypo2]
        
        seg_metrics = {
            "overnight": compute_segment_metrics(overnight),
            "daytime": compute_segment_metrics(daytime),
            "whole": compute_segment_metrics(whole_day)
        }
        results.append(seg_metrics)
    
    if not results:
        return [0]*18
    
    n = len(results)
    whole_acc = np.zeros(6)
    day_acc = np.zeros(6)
    night_acc = np.zeros(6)
    
    for r in results:
        whole_acc += r["whole"]
        day_acc += r["daytime"]
        night_acc += r["overnight"]
    
    whole_avg = np.round(whole_acc / n, 4).tolist()
    day_avg = np.round(day_acc / n, 4).tolist()
    night_avg = np.round(night_acc / n, 4).tolist()
    
    return night_avg + day_avg + whole_avg

def main():
    cgm = pd.read_csv("CGMData.csv", low_memory=False)
    insulin = pd.read_csv("InsulinData.csv", low_memory=False)
    cgm["Sensor Glucose (mg/dL)"] = pd.to_numeric(cgm["Sensor Glucose (mg/dL)"], errors="coerce")
    cgm = parse_timestamps(cgm, "Date", "Time")
    insulin = parse_timestamps(insulin, "Date", "Time")
    cgm.dropna(subset=["Timestamp"], inplace=True)
    insulin.dropna(subset=["Timestamp"], inplace=True)
    cgm.sort_values("Timestamp", ascending=True, inplace=True)
    insulin.sort_values("Timestamp", ascending=True, inplace=True)
    auto_start = find_auto_mode_start_time(insulin, alarm_col="Alarm", event="AUTO MODE ACTIVE PLGM OFF")
    if auto_start is None:
        cgm_manual = cgm.copy()
        cgm_auto = cgm.iloc[0:0].copy()
    else:
        boundary = find_nearest_cgm_time(cgm, auto_start)
        if boundary is None:
            cgm_manual = cgm.copy()
            cgm_auto = cgm.iloc[0:0].copy()
        else:
            cgm_manual = cgm[cgm["Timestamp"] < boundary].copy()
            cgm_auto = cgm[cgm["Timestamp"] >= boundary].copy()
    cgm_manual_daily = reindex_day_and_interpolate(cgm_manual, "Sensor Glucose (mg/dL)", min_valid=1)
    cgm_auto_daily = reindex_day_and_interpolate(cgm_auto, "Sensor Glucose (mg/dL)", min_valid=1)
    manual_metrics = compute_daily_metrics(cgm_manual_daily, "Sensor Glucose (mg/dL)")
    auto_metrics = compute_daily_metrics(cgm_auto_daily, "Sensor Glucose (mg/dL)")
    result_df = pd.DataFrame([manual_metrics, auto_metrics])
    result_df.to_csv("Result.csv", header=False, index=False)

if __name__ == "__main__":
    main()



