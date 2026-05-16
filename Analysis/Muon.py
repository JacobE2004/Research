import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILE PATH
# ============================================================
# Put your detector CSV file path here.
#
# If the CSV is in the same folder as this Python code, just use:
# DATA_FILE = "events_20260429_135424.csv"
#
# If using Ubuntu/WSL and the file is in Windows Downloads, it may look like:
# DATA_FILE = "/mnt/c/Users/Jmell/Downloads/events_20260429_135424.csv"

DATA_FILE = r"C:\Users\Jmell\Downloads\events_20260501_143050.csv"


# ============================================================
# DETECTOR SETUP
# ============================================================
# Your lab setup:
#
# BOTTOM = bottom detector signal
# COINC  = hardware coincidence signal
#          meaning TOP and BOTTOM triggered close together
#
# TOP is not used because you cannot actively measure it separately.
#
# This code calculates:
#
# delta_t = t_COINC - t_BOTTOM
#
# Positive delta_t means BOTTOM was logged before COINC.
# Negative delta_t means COINC was logged before the nearest BOTTOM.
# ============================================================

BOTTOM_DETECTOR = "BOTTOM"
COINC_DETECTOR = "COINC"

# Search window for finding the nearest BOTTOM event around each COINC event.
# 10,000,000 ns = 10 ms
PAIR_SEARCH_WINDOW_NS = 10_000_000

# Coincidence timing window.
#
# 100 ns = 100
# 1 microsecond = 1_000
# 100 microseconds = 100_000
# 1 millisecond = 1_000_000
#
# Raspberry Pi Python GPIO timing is not reliable at true 100 ns,
# so 100_000 ns is a more realistic starting value.
COINCIDENCE_WINDOW_NS = 100_000

HISTOGRAM_BINS = 40


# ============================================================
# LOAD DATA
# ============================================================

def load_events(csv_file):
    df = pd.read_csv(csv_file)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # If your newer logger file has EVENT and PAIR rows,
    # only keep the raw detector EVENT rows.
    if "row_type" in df.columns:
        df = df[df["row_type"].astype(str).str.upper().str.strip() == "EVENT"].copy()

    required_columns = ["timestamp", "detector", "pin"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV file: {col}")

    df["detector"] = df["detector"].astype(str).str.upper().str.strip()

    timing_source = "timestamp"

    # Prefer monotonic_ns if it exists, because it is better for timing.
    if "monotonic_ns" in df.columns:
        df["monotonic_ns"] = pd.to_numeric(df["monotonic_ns"], errors="coerce")

        if df["monotonic_ns"].notna().any():
            df = df.dropna(subset=["monotonic_ns"]).copy()
            df["t_ns"] = df["monotonic_ns"].astype(np.int64)
            timing_source = "monotonic_ns"

    # Fallback: use the timestamp column.
    if timing_source == "timestamp":
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp_dt"]).copy()

        t0 = df["timestamp_dt"].min()

        df["t_ns"] = (
            (df["timestamp_dt"] - t0).dt.total_seconds() * 1e9
        ).round().astype(np.int64)

    df = df.sort_values("t_ns").reset_index(drop=True)

    df["time_since_start_s"] = (df["t_ns"] - df["t_ns"].min()) / 1e9

    return df, timing_source


# ============================================================
# PAIR EACH COINC EVENT WITH NEAREST BOTTOM EVENT
# ============================================================

def pair_coinc_with_nearest_bottom(df):
    bottom_df = df[df["detector"] == BOTTOM_DETECTOR].copy().reset_index(drop=True)
    coinc_df = df[df["detector"] == COINC_DETECTOR].copy().reset_index(drop=True)

    if len(bottom_df) == 0:
        print("No BOTTOM events found.")
        return pd.DataFrame()

    if len(coinc_df) == 0:
        print("No COINC events found.")
        return pd.DataFrame()

    bottom_times = bottom_df["t_ns"].to_numpy(dtype=np.int64)

    pairs = []

    for coinc_index, coinc_row in coinc_df.iterrows():
        coinc_time = int(coinc_row["t_ns"])

        # Find where the COINC time fits into the sorted BOTTOM times
        insert_position = np.searchsorted(bottom_times, coinc_time)

        candidates = []

        # Candidate: closest BOTTOM before COINC
        if insert_position - 1 >= 0:
            bottom_index = insert_position - 1
            bottom_time = int(bottom_times[bottom_index])
            delta_t_ns = coinc_time - bottom_time

            candidates.append({
                "bottom_index": bottom_index,
                "bottom_time": bottom_time,
                "delta_t_ns": delta_t_ns,
                "position": "before"
            })

        # Candidate: closest BOTTOM after COINC
        if insert_position < len(bottom_times):
            bottom_index = insert_position
            bottom_time = int(bottom_times[bottom_index])
            delta_t_ns = coinc_time - bottom_time

            candidates.append({
                "bottom_index": bottom_index,
                "bottom_time": bottom_time,
                "delta_t_ns": delta_t_ns,
                "position": "after"
            })

        if len(candidates) == 0:
            continue

        # Choose whichever BOTTOM event is closest in absolute time
        closest = min(candidates, key=lambda x: abs(x["delta_t_ns"]))

        bottom_row = bottom_df.loc[closest["bottom_index"]]
        delta_t_ns = closest["delta_t_ns"]

        within_search_window = abs(delta_t_ns) <= PAIR_SEARCH_WINDOW_NS
        accepted = abs(delta_t_ns) <= COINCIDENCE_WINDOW_NS

        pairs.append({
            "coinc_event_number": coinc_index + 1,
            "bottom_event_number": closest["bottom_index"] + 1,

            "coinc_timestamp": coinc_row["timestamp"],
            "bottom_timestamp": bottom_row["timestamp"],

            "coinc_pin": coinc_row["pin"],
            "bottom_pin": bottom_row["pin"],

            "coinc_t_ns": coinc_time,
            "bottom_t_ns": closest["bottom_time"],

            "delta_t_ns": delta_t_ns,
            "delta_t_us": delta_t_ns / 1_000,
            "delta_t_ms": delta_t_ns / 1_000_000,

            "nearest_bottom_position": closest["position"],

            "within_search_window": within_search_window,
            "accepted_within_coincidence_window": accepted
        })

    return pd.DataFrame(pairs)


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(df, pairs, timing_source):
    print()
    print("========== MUON DETECTION DATA SUMMARY ==========")
    print()
    print("Detector setup:")
    print("BOTTOM = bottom detector pulse")
    print("COINC  = hardware coincidence pulse from TOP + BOTTOM")
    print("TOP is ignored because it is not actively readable right now.")
    print()
    print(f"Timing source used: {timing_source}")

    if timing_source == "timestamp":
        print()
        print("WARNING:")
        print("This file does not have monotonic_ns timing.")
        print("The analysis is using regular timestamps, so timing precision is limited.")

    print()
    print("Detector counts:")
    counts = df["detector"].value_counts()
    print(counts.to_string())

    run_time_s = (df["t_ns"].max() - df["t_ns"].min()) / 1e9
    run_time_min = run_time_s / 60

    print()
    print(f"Total run time: {run_time_s:.3f} seconds")
    print(f"Total run time: {run_time_min:.3f} minutes")

    print()
    print("Detector rates:")
    for detector, count in counts.items():
        rate_per_s = count / run_time_s if run_time_s > 0 else np.nan
        rate_per_min = rate_per_s * 60
        print(f"{detector}: {rate_per_s:.3f} events/s = {rate_per_min:.3f} events/min")

    bottom_count = counts.get(BOTTOM_DETECTOR, 0)
    coinc_count = counts.get(COINC_DETECTOR, 0)

    print()
    print("Hardware coincidence / muon-candidate rate:")
    if run_time_s > 0:
        coinc_rate_per_min = coinc_count / run_time_s * 60
        print(f"{coinc_count} COINC events in {run_time_s:.3f} seconds")
        print(f"COINC rate = {coinc_rate_per_min:.3f} muon candidates/min")

    if bottom_count > 0:
        ratio = coinc_count / bottom_count
        print(f"COINC / BOTTOM ratio = {coinc_count} / {bottom_count} = {ratio:.6f}")

    print()
    print("Timing analysis:")
    print(f"delta_t = t_{COINC_DETECTOR} - t_{BOTTOM_DETECTOR}")
    print("Positive delta_t means BOTTOM was logged before COINC.")
    print("Negative delta_t means COINC was logged before the nearest BOTTOM.")
    print()
    print(f"Pair search window: {PAIR_SEARCH_WINDOW_NS} ns")
    print(f"Coincidence acceptance window: {COINCIDENCE_WINDOW_NS} ns")

    print()
    print(f"Total COINC events: {coinc_count}")
    print(f"Total COINC-BOTTOM pairs made: {len(pairs)}")

    if len(pairs) == 0:
        print("No COINC-BOTTOM pairs were made.")
        return

    within_search = pairs[pairs["within_search_window"] == True]
    accepted = pairs[pairs["accepted_within_coincidence_window"] == True]

    print(f"Pairs within search window: {len(within_search)}")
    print(f"Pairs accepted within coincidence window: {len(accepted)}")

    print()
    print("Delta-t statistics for nearest BOTTOM to each COINC:")
    print(f"Mean delta_t:   {pairs['delta_t_ns'].mean():.3f} ns")
    print(f"Median delta_t: {pairs['delta_t_ns'].median():.3f} ns")
    print(f"Std delta_t:    {pairs['delta_t_ns'].std():.3f} ns")
    print(f"Min delta_t:    {pairs['delta_t_ns'].min():.3f} ns")
    print(f"Max delta_t:    {pairs['delta_t_ns'].max():.3f} ns")

    print()
    print("Closest BOTTOM event to each COINC event:")
    for _, row in pairs.iterrows():
        print(
            f"COINC event {int(row['coinc_event_number'])}: "
            f"nearest BOTTOM was {row['nearest_bottom_position']} COINC, "
            f"delta_t = {row['delta_t_ns']} ns "
            f"= {row['delta_t_us']:.3f} us "
            f"= {row['delta_t_ms']:.6f} ms"
        )


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

def save_summary_txt(df, pairs, timing_source, output_file):
    counts = df["detector"].value_counts()
    run_time_s = (df["t_ns"].max() - df["t_ns"].min()) / 1e9

    with open(output_file, "w") as f:
        f.write("MUON DETECTION DATA SUMMARY\n")
        f.write("===========================\n\n")

        f.write("Detector setup:\n")
        f.write("BOTTOM = bottom detector pulse\n")
        f.write("COINC  = hardware coincidence pulse from TOP + BOTTOM\n")
        f.write("TOP is ignored because it is not actively readable right now.\n\n")

        f.write(f"Timing source used: {timing_source}\n\n")

        f.write("Detector counts:\n")
        f.write(counts.to_string())
        f.write("\n\n")

        f.write(f"Total run time: {run_time_s:.3f} seconds\n")
        f.write(f"Total run time: {run_time_s / 60:.3f} minutes\n\n")

        f.write("Detector rates:\n")
        for detector, count in counts.items():
            rate_per_s = count / run_time_s if run_time_s > 0 else np.nan
            rate_per_min = rate_per_s * 60
            f.write(f"{detector}: {rate_per_s:.3f} events/s = {rate_per_min:.3f} events/min\n")

        bottom_count = counts.get(BOTTOM_DETECTOR, 0)
        coinc_count = counts.get(COINC_DETECTOR, 0)

        f.write("\nHardware coincidence / muon-candidate rate:\n")
        if run_time_s > 0:
            coinc_rate_per_min = coinc_count / run_time_s * 60
            f.write(f"{coinc_count} COINC events in {run_time_s:.3f} seconds\n")
            f.write(f"COINC rate = {coinc_rate_per_min:.3f} muon candidates/min\n")

        if bottom_count > 0:
            ratio = coinc_count / bottom_count
            f.write(f"COINC / BOTTOM ratio = {coinc_count} / {bottom_count} = {ratio:.6f}\n")

        f.write("\nTiming analysis:\n")
        f.write(f"delta_t = t_{COINC_DETECTOR} - t_{BOTTOM_DETECTOR}\n")
        f.write("Positive delta_t means BOTTOM was logged before COINC.\n")
        f.write("Negative delta_t means COINC was logged before the nearest BOTTOM.\n")
        f.write(f"Pair search window: {PAIR_SEARCH_WINDOW_NS} ns\n")
        f.write(f"Coincidence acceptance window: {COINCIDENCE_WINDOW_NS} ns\n\n")

        f.write(f"Total COINC-BOTTOM pairs made: {len(pairs)}\n")

        if len(pairs) > 0:
            accepted = pairs[pairs["accepted_within_coincidence_window"] == True]
            f.write(f"Pairs accepted within coincidence window: {len(accepted)}\n\n")

            f.write("Delta-t statistics:\n")
            f.write(f"Mean delta_t:   {pairs['delta_t_ns'].mean():.3f} ns\n")
            f.write(f"Median delta_t: {pairs['delta_t_ns'].median():.3f} ns\n")
            f.write(f"Std delta_t:    {pairs['delta_t_ns'].std():.3f} ns\n")
            f.write(f"Min delta_t:    {pairs['delta_t_ns'].min():.3f} ns\n")
            f.write(f"Max delta_t:    {pairs['delta_t_ns'].max():.3f} ns\n")


# ============================================================
# MAKE PLOTS
# ============================================================

def make_plots(df, pairs, output_prefix):
    # Plot 1: cumulative detector counts
    plt.figure(figsize=(8, 5))

    for detector in [BOTTOM_DETECTOR, COINC_DETECTOR]:
        detector_df = df[df["detector"] == detector].copy()

        if len(detector_df) == 0:
            continue

        x = detector_df["time_since_start_s"].to_numpy()
        y = np.arange(1, len(detector_df) + 1)

        plt.step(x, y, where="post", label=detector)

    plt.xlabel("Time since start [s]")
    plt.ylabel("Cumulative counts")
    plt.title("Cumulative Detector Counts")
    plt.legend()
    plt.tight_layout()

    cumulative_file = f"{output_prefix}_cumulative_counts.png"
    plt.savefig(cumulative_file, dpi=300)
    print(f"Saved plot: {cumulative_file}")

    # Plot 2: event timeline
    timeline_df = df[df["detector"].isin([BOTTOM_DETECTOR, COINC_DETECTOR])].copy()

    if len(timeline_df) > 0:
        y_map = {
            BOTTOM_DETECTOR: 0,
            COINC_DETECTOR: 1
        }

        plt.figure(figsize=(8, 3))

        plt.scatter(
            timeline_df["time_since_start_s"],
            timeline_df["detector"].map(y_map),
            marker="|",
            s=100
        )

        plt.yticks([0, 1], [BOTTOM_DETECTOR, COINC_DETECTOR])
        plt.xlabel("Time since start [s]")
        plt.ylabel("Detector channel")
        plt.title("Event Timeline")
        plt.tight_layout()

        timeline_file = f"{output_prefix}_event_timeline.png"
        plt.savefig(timeline_file, dpi=300)
        print(f"Saved plot: {timeline_file}")

    # Plot 3: delta-t histogram
    if len(pairs) > 0:
        plt.figure(figsize=(8, 5))

        plt.hist(pairs["delta_t_ns"], bins=HISTOGRAM_BINS)

        plt.axvline(0, linestyle="-", label="zero delay")
        plt.axvline(COINCIDENCE_WINDOW_NS, linestyle="--", label="+ window")
        plt.axvline(-COINCIDENCE_WINDOW_NS, linestyle="--", label="- window")

        plt.xlabel(f"Delta t = t_{COINC_DETECTOR} - t_{BOTTOM_DETECTOR} [ns]")
        plt.ylabel("Number of COINC events")
        plt.title("Delta-t Histogram")
        plt.legend()
        plt.tight_layout()

        hist_file = f"{output_prefix}_delta_t_histogram.png"
        plt.savefig(hist_file, dpi=300)
        print(f"Saved plot: {hist_file}")

        # Plot 4: delta-t by event number
        plt.figure(figsize=(8, 5))

        plt.plot(
            np.arange(1, len(pairs) + 1),
            pairs["delta_t_ns"],
            marker="o",
            linestyle=""
        )

        plt.axhline(0, linestyle="-", label="zero delay")
        plt.axhline(COINCIDENCE_WINDOW_NS, linestyle="--", label="+ window")
        plt.axhline(-COINCIDENCE_WINDOW_NS, linestyle="--", label="- window")

        plt.xlabel("COINC event number")
        plt.ylabel(f"Delta t = t_{COINC_DETECTOR} - t_{BOTTOM_DETECTOR} [ns]")
        plt.title("Delta-t for Each COINC Event")
        plt.legend()
        plt.tight_layout()

        dt_file = f"{output_prefix}_delta_t_by_event.png"
        plt.savefig(dt_file, dpi=300)
        print(f"Saved plot: {dt_file}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze BOTTOM and COINC muon detector data."
    )

    parser.add_argument(
        "csv_file",
        nargs="?",
        default=DATA_FILE,
        help="CSV file from the Raspberry Pi detector logger"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_file)

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        print()
        print("Check the DATA_FILE variable near the top of this script,")
        print("or run the script with the CSV file path like this:")
        print("python3 analyze_muon_bottom_coinc.py events_20260429_135424.csv")
        return

    df, timing_source = load_events(csv_path)

    pairs = pair_coinc_with_nearest_bottom(df)

    output_prefix = csv_path.stem

    paired_csv = f"{output_prefix}_bottom_coinc_pairs.csv"
    summary_txt = f"{output_prefix}_summary.txt"

    pairs.to_csv(paired_csv, index=False)

    print_summary(df, pairs, timing_source)

    save_summary_txt(df, pairs, timing_source, summary_txt)

    print()
    print(f"Saved paired-event data: {paired_csv}")
    print(f"Saved summary file: {summary_txt}")

    make_plots(df, pairs, output_prefix)


if __name__ == "__main__":
    main()