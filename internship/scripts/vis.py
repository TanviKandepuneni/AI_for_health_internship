import os
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages


def find_file(folder_path, keyword):
    for file in os.listdir(folder_path):
        if keyword.lower() in file.lower() and file.endswith(".txt"):
            return os.path.join(folder_path, file)
    return None


def read_signal_file(file_path):
    timestamps = []
    values = []
    data_started = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line == "":
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
            value_str = parts[1].strip()

            try:
                time_val = pd.to_datetime(time_str, format="%d.%m.%Y %H:%M:%S,%f")
                value_val = float(value_str)
                timestamps.append(time_val)
                values.append(value_val)
            except:
                pass

    return pd.DataFrame({
        "timestamp": timestamps,
        "value": values
    })


def read_event_file(file_path):
    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            if ";" not in line:
                continue
            # ignore header-like lines
            if line.startswith("Signal") or line.startswith("Start") or line.startswith("Data") or line.startswith("Unit"):
                continue
            parts = [x.strip() for x in line.split(";")]
            if len(parts) < 4:
                continue
            time_range = parts[0]
            duration = parts[1]
            event_name = parts[2]
            sleep_stage = parts[3]

            if "-" not in time_range:
                continue

            try:
                start_str, end_time_only = time_range.split("-")
                start_dt = pd.to_datetime(start_str, format="%d.%m.%Y %H:%M:%S,%f")
                same_day = start_dt.strftime("%d.%m.%Y ")
                end_dt = pd.to_datetime(same_day + end_time_only, format="%d.%m.%Y %H:%M:%S,%f")
                if end_dt < start_dt:
                    end_dt += pd.Timedelta(days=1)
                rows.append({
                    "start": start_dt,
                    "end": end_dt,
                    "duration": float(duration),
                    "event": event_name,
                    "stage": sleep_stage
                })
            except:
                pass

    return pd.DataFrame(rows)


def add_event_shading(ax, events_df, chunk_start, chunk_end):
    color_map = {
        "Hypopnea": "#fff59d",
        "Obstructive Apnea": "#ef9a9a",
        "Mixed Apnea": "#ce93d8",
        "Central Apnea": "#90caf9",
        "Body event": "#c8e6c9"
    }
    current_events = events_df[
        (events_df["end"] >= chunk_start) & (events_df["start"] <= chunk_end)
    ]
    ymin, ymax = ax.get_ylim()
    for _, row in current_events.iterrows():
        start_time = max(row["start"], chunk_start)
        end_time = min(row["end"], chunk_end)
        event_name = row["event"]

        color = color_map.get(event_name, "#dddddd")

        ax.axvspan(start_time, end_time, color=color, alpha=0.45)

        middle = start_time + (end_time - start_time) / 2
        ax.text(
            middle,
            ymax - 0.05 * (ymax - ymin),
            event_name,
            ha="center",
            va="top",
            fontsize=7
        )


# downsampling to show fewer points on the graph( not too cluttered)
def downsample_for_plot(df, step):
    if len(df) == 0:
        return df
    return df.iloc[::step].copy()

def set_nice_ylim(ax, values):
    if len(values) == 0:
        return
    vmin = values.min()
    vmax = values.max()
    if vmin == vmax:
        pad = 1
    else:
        pad = 0.08 * (vmax - vmin)

    ax.set_ylim(vmin - pad, vmax + pad)


def plot_one_chunk(pdf, nasal_chunk, thor_chunk, spo2_chunk, events_df, person_name, chunk_start, chunk_end):
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

    # downsampling just for visualization
    nasal_plot = downsample_for_plot(nasal_chunk, 4)
    thor_plot = downsample_for_plot(thor_chunk, 4)
    spo2_plot = downsample_for_plot(spo2_chunk, 1)

    # nasal airflow
    axes[0].plot(
        nasal_plot["timestamp"],
        nasal_plot["value"],
        linewidth=0.8,
        label="Nasal Flow"
    )
    axes[0].set_ylabel("Nasal Flow")
    axes[0].legend(loc="upper right", fontsize=8)
    set_nice_ylim(axes[0], nasal_chunk["value"])

    # thoracic
    axes[1].plot(
        thor_plot["timestamp"],
        thor_plot["value"],
        linewidth=0.8,
        label="Thoracic/Abdominal Resp."
    )
    axes[1].set_ylabel("Resp. Amplitude")
    axes[1].legend(loc="upper right", fontsize=8)
    set_nice_ylim(axes[1], thor_chunk["value"])

    # spo2
    axes[2].step(
        spo2_plot["timestamp"],
        spo2_plot["value"],
        where="post",
        linewidth=0.9,
        label="SpO2"
    )
    axes[2].set_ylabel("SpO2 (%)")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper right", fontsize=8)
    set_nice_ylim(axes[2], spo2_chunk["value"])

    title = f"{person_name} - {chunk_start.strftime('%Y-%m-%d %H:%M:%S')} to {chunk_end.strftime('%H:%M:%S')}"
    axes[0].set_title(title, fontsize=11)

    for ax in axes:
        ax.set_xlim(chunk_start, chunk_end)
        ax.grid(True, alpha=0.3)
        add_event_shading(ax, events_df, chunk_start, chunk_end)

    axes[0].tick_params(axis="x", labelbottom=False)
    axes[1].tick_params(axis="x", labelbottom=False)

    axes[2].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[2].tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-name",
        "--name",
        default="Data/AP05",
        help='path to one participant folder, for example "Data/AP05"'
    )
    args = parser.parse_args()

    participant_folder = args.name
    person_name = Path(participant_folder).name

    nasal_file = find_file(participant_folder, "flow nasal")
    thor_file = find_file(participant_folder, "thorac")
    spo2_file = find_file(participant_folder, "spo2")
    event_file = find_file(participant_folder, "flow events")

    if nasal_file is None:
        raise FileNotFoundError("Could not find Flow Nasal file")
    if thor_file is None:
        raise FileNotFoundError("Could not find Thorac Movement file")
    if spo2_file is None:
        raise FileNotFoundError("Could not find SPO2 file")
    if event_file is None:
        raise FileNotFoundError("Could not find Flow Events file")

    nasal_df = read_signal_file(nasal_file)
    thor_df = read_signal_file(thor_file)
    spo2_df = read_signal_file(spo2_file)
    events_df = read_event_file(event_file)

    print("Nasal rows:", len(nasal_df))
    print("Thoracic rows:", len(thor_df))
    print("SpO2 rows:", len(spo2_df))
    print("Event rows:", len(events_df))

    overall_start = min(
        nasal_df["timestamp"].min(),
        thor_df["timestamp"].min(),
        spo2_df["timestamp"].min()
    )

    overall_end = max(
        nasal_df["timestamp"].max(),
        thor_df["timestamp"].max(),
        spo2_df["timestamp"].max()
    )

    output_dir = Path("Visualizations")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{person_name}_visualization.pdf"

    chunk_size = pd.Timedelta(minutes=5)
    current_start = overall_start

    with PdfPages(output_file) as pdf:
        while current_start < overall_end:
            current_end = min(current_start + chunk_size, overall_end)

            nasal_chunk = nasal_df[
                (nasal_df["timestamp"] >= current_start) &
                (nasal_df["timestamp"] < current_end)
            ]

            thor_chunk = thor_df[
                (thor_df["timestamp"] >= current_start) &
                (thor_df["timestamp"] < current_end)
            ]

            spo2_chunk = spo2_df[
                (spo2_df["timestamp"] >= current_start) &
                (spo2_df["timestamp"] < current_end)
            ]

            if len(nasal_chunk) > 0 and len(thor_chunk) > 0 and len(spo2_chunk) > 0:
                plot_one_chunk(
                    pdf,
                    nasal_chunk,
                    thor_chunk,
                    spo2_chunk,
                    events_df,
                    person_name,
                    current_start,
                    current_end
                )

            current_start = current_end

    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()