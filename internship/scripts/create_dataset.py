import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# to make file matching easier, basically normalizing names
def clean_name(name):
    name = name.lower()
    name = name.replace(" ", "")
    name = name.replace("_", "")
    name = name.replace("-", "")
    return name


# find a txt file using keywords
def find_file(folder, keywords, exclude=None):
    if exclude is None:
        exclude = []
    keywords = [clean_name(k) for k in keywords]
    exclude = [clean_name(x) for x in exclude]

    for f in Path(folder).glob("*.txt"):
        name = clean_name(f.name)
        for k in keywords:
            if k in name:
                if any(x in name for x in exclude):
                    continue
                return str(f)

    return None


# read signal file after "Data:"
def read_signal_file(path):
    times = []
    values = []
    data_started = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Data:"):
                data_started = True
                continue
            if not data_started:
                continue
            if ";" not in line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue

            time_str = parts[0].strip()
            value_str = parts[1].strip().split()[0]
            t = pd.to_datetime(time_str, format="%d.%m.%Y %H:%M:%S,%f", errors="coerce")
            if pd.isna(t):
                t = pd.to_datetime(time_str, errors="coerce", dayfirst=True)
            v = pd.to_numeric(value_str, errors="coerce")
            if pd.isna(t) or pd.isna(v):
                continue
            times.append(t)
            values.append(v)

    df = pd.DataFrame({
        "timestamp": times,
        "value": values
    })

    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df


# read event file
def read_event_file(path):
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or ";" not in line:
                continue
            if line.startswith("Signal") or line.startswith("Start") or line.startswith("Unit"):
                continue
            parts = [x.strip() for x in line.split(";")]
            if len(parts) < 4:
                continue
            time_range = parts[0]
            event_name = parts[2]
            if "-" not in time_range:
                continue
            try:
                start_str, end_str = time_range.split("-")
                start = pd.to_datetime(start_str, errors="coerce", dayfirst=True)
                if pd.isna(start):
                    continue

                same_day = start.strftime("%d.%m.%Y ")
                end = pd.to_datetime(same_day + end_str, errors="coerce", dayfirst=True)

                if pd.isna(end):
                    continue
                if end < start:
                    end += pd.Timedelta(days=1)
                rows.append({
                    "start": start,
                    "end": end,
                    "event": event_name
                })
            except:
                pass

    return pd.DataFrame(rows)


# bandpass filter for breathing range about 0.17 Hz to 0.4 Hz
def bandpass_filter(signal, fs, low=0.17, high=0.4, order=4):
    if len(signal) < 20:
        return signal

    nyq = 0.5 * fs
    low_cut = low / nyq
    high_cut = high / nyq
    b, a = butter(order, [low_cut, high_cut], btype="band")
    filtered = filtfilt(b, a, signal)
    return filtered


# align all signals to 1-second resolution
def align_signals(flow_df, thor_df, spo2_df=None):
    flow = flow_df.copy().set_index("timestamp")
    thor = thor_df.copy().set_index("timestamp")
    flow_1s = flow["value"].resample("1s").mean()
    thor_1s = thor["value"].resample("1s").mean()

    data = {
        "flow": flow_1s,
        "thor": thor_1s
    }
    if spo2_df is not None and len(spo2_df) > 0:
        spo2 = spo2_df.copy().set_index("timestamp")
        spo2_1s = spo2["value"].resample("1s").mean()
        data["spo2"] = spo2_1s

    df = pd.DataFrame(data)
    df = df.interpolate(method="time").ffill().bfill().reset_index()

    return df


# assign label for one window if overlap > 50%, assign event otherwise Normal
def get_window_label(start_time, end_time, events_df):
    if events_df is None or len(events_df) == 0:
        return "Normal"

    window_len = (end_time - start_time).total_seconds()
    best_label = "Normal"
    best_overlap = 0

    for _, row in events_df.iterrows():
        overlap_start = max(start_time, row["start"])
        overlap_end = min(end_time, row["end"])
        overlap = (overlap_end - overlap_start).total_seconds()
        if overlap > 0:
            overlap_ratio = overlap / window_len
            if overlap_ratio > 0.5 and overlap > best_overlap:
                best_overlap = overlap
                best_label = row["event"]

    return best_label


# process one patient folder
def process_patient(folder):
    patient_name = Path(folder).name
    flow_file = find_file(
        folder,
        ["flow nasal", "flow signal", "flow"],
        exclude=["events"]
    )
    thor_file = find_file(
        folder,
        ["thorac movement", "thorac signal", "thorac", "thoracic movement", "thoracic signal", "thoracic"],
        exclude=["events"]
    )
    spo2_file = find_file(
        folder,
        ["spo2 signal", "spo2"]
    )
    event_file = find_file(
        folder,
        ["flow events"]
    )

    print(f"\n{patient_name}")
    print("flow file :", flow_file)
    print("thor file :", thor_file)
    print("spo2 file :", spo2_file)
    print("event file:", event_file)

    if flow_file is None or thor_file is None:
        print("Skipping because core breathing files are missing")
        return []
    flow_df = read_signal_file(flow_file)
    thor_df = read_signal_file(thor_file)
    spo2_df = None
    if spo2_file is not None:
        spo2_df = read_signal_file(spo2_file)
    events_df = pd.DataFrame()
    if event_file is not None:
        events_df = read_event_file(event_file)
    print("flow rows:", len(flow_df))
    print("thor rows:", len(thor_df))
    if spo2_df is not None:
        print("spo2 rows:", len(spo2_df))
    # filtering the breathing signals
    if len(flow_df) > 0:
        flow_df["value"] = bandpass_filter(flow_df["value"].values, fs=32)
    if len(thor_df) > 0:
        thor_df["value"] = bandpass_filter(thor_df["value"].values, fs=32)
    try:
        aligned_df = align_signals(flow_df, thor_df, spo2_df)
    except Exception as e:
        print("Problem aligning signals:", e)
        return []

    rows = []
    window_seconds = 30
    step_seconds = 15   # 50% overlap
    start_time = aligned_df["timestamp"].min()
    end_time = aligned_df["timestamp"].max()
    current_start = start_time

    while current_start + pd.Timedelta(seconds=window_seconds) <= end_time:
        current_end = current_start + pd.Timedelta(seconds=window_seconds)
        window_df = aligned_df[
            (aligned_df["timestamp"] >= current_start) &
            (aligned_df["timestamp"] < current_end)
        ].copy()

        if len(window_df) < 25:
            current_start += pd.Timedelta(seconds=step_seconds)
            continue

        label = get_window_label(current_start, current_end, events_df)

        row = {
            "patient": patient_name,
            "window_start": current_start,
            "window_end": current_end,
            "label": label,
            # save the actual filtered window values
            "flow_window": window_df["flow"].tolist(),
            "thor_window": window_df["thor"].tolist()
        }

        if "spo2" in window_df.columns:
            row["spo2_window"] = window_df["spo2"].tolist()
        else:
            row["spo2_window"] = None

        rows.append(row)
        current_start += pd.Timedelta(seconds=step_seconds)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", "--in_dir", required=True, help='input directory, like "Data"')
    parser.add_argument("-out_dir", "--out_dir", required=True, help='output directory, like "Dataset"')
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for item in sorted(os.listdir(in_dir)):
        folder = in_dir / item
        if folder.is_dir():
            rows = process_patient(folder)
            all_rows.extend(rows)

    dataset_df = pd.DataFrame(all_rows)
    csv_path = out_dir / "breathing_windows_final.csv"
    pkl_path = out_dir / "breathing_windows_final.pkl"
    dataset_df.to_csv(csv_path, index=False)
    dataset_df.to_pickle(pkl_path)

    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved Pickle to {pkl_path}")
    print(f"Total windows: {len(dataset_df)}")

    if len(dataset_df) > 0:
        print("\nLabel counts:")
        print(dataset_df["label"].value_counts())


if __name__ == "__main__":
    main()