# ...

import numpy as np
import pickle
import zipfile
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
def load_subject_pickle(DATA_SOURCE: Path, subject_id: str, verbose: bool = True):

    pkl_rel_path = f"{subject_id}/{subject_id}.pkl"
    if verbose:
        print(f"\nLoading {pkl_rel_path} ...")

    if DATA_SOURCE.is_file() and DATA_SOURCE.suffix == ".zip":
        with zipfile.ZipFile(DATA_SOURCE, "r") as zf:
            with zf.open(pkl_rel_path) as f:
                data = pickle.load(f, encoding="latin1")
    else:
        pkl_path = DATA_SOURCE / subject_id / f"{subject_id}.pkl"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

    if verbose:
        print(f"Loaded {subject_id} successfully.")
    return data

###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
def summarise_subject(subject_id: str, data: dict):

    FS_WRIST = {
        "ACC": 32,
        "BVP": 64,
        "EDA": 4,
        "TEMP": 4,
    }

    LABEL_MAP = {
        1: "Baseline",
        2: "Stress",
        3: "Amusement",
        4: "Meditation",
    }

    print(f"\n===== {subject_id} SUMMARY (WRIST ONLY) =====")

    wrist = data['signal']['wrist'] # wrist only
    labels = np.asarray(data['label'])

    # print sampling rates and shapes
    print('\nSampling rates (Hz) and signal shapes:')
    for sensor_name in ["ACC", "BVP", "EDA", "TEMP"]:
        arr = np.asarray(wrist[sensor_name])
        shape = arr.shape
        print(f"{sensor_name}: {FS_WRIST[sensor_name]} Hz, shape: {shape}")

    # compute durations per sensor
    print("\nDurations (seconds) per sensor:")
    durations = {}
    for sensor_name in ["ACC", "BVP", "EDA", "TEMP"]:
        arr = np.asarray(wrist[sensor_name])
        n_samples = arr.shape[0]
        durations[sensor_name] = n_samples / FS_WRIST[sensor_name]
        print(f"{sensor_name}: {durations[sensor_name]:.2f}s")
    
    # use max duration as "total" (all sensors should align closely)
    total_duration = max(durations.values())
    print(f"\nTotal recording duration (based on longest sensor): {total_duration:.2f}s")

    # label distribution (keep only Baseline/Stress/Amusement/Meditation)
    print("\nLabel distribution:")
    label_counts = {LABEL_MAP[k]: int(np.sum(labels == k)) for k in LABEL_MAP}
    total_labeled = sum(label_counts.values())
    for label_name, count in label_counts.items():
        percentage = (count / total_labeled * 100) if total_labeled > 0 else 0
        print(f"{label_name}: {count} samples ({percentage:.2f}%)")

###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
def compact_summary(subject_id, data):

    FS_WRIST = {
        "ACC": 32,
        "BVP": 64,
        "EDA": 4,
        "TEMP": 4,
    }
     
    LABEL_MAP = {
        1: "Baseline",
        2: "Stress",
        3: "Amusement",
        4: "Meditation",
    }

    # Wrist-only signals
    wrist = data["signal"]["wrist"]
    labels = np.asarray(data["label"]).astype(int)

    # Duration (all wrist sensors align; BVP is fine as reference)
    duration_s = len(wrist["BVP"]) / FS_WRIST["BVP"]

    # Label counts and percentages (Baseline/Stress/Amusement/Meditation only)
    counts = Counter(labels)
    total = sum(counts[k] for k in LABEL_MAP.keys())

    row = {
        "subject": subject_id,
        "duration_s": duration_s,
        "BVP_n": len(wrist["BVP"]),
        "EDA_n": len(wrist["EDA"]),
        "TEMP_n": len(wrist["TEMP"]),
        "ACC_n": len(wrist["ACC"]),
        "Baseline%": 100.0 * counts[1] / total if total else 0.0,
        "Stress%": 100.0 * counts[2] / total if total else 0.0,
        "Amusement%": 100.0 * counts[3] / total if total else 0.0,
        "Meditation%": 100.0 * counts[4] / total if total else 0.0,
    }
    return row

###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
def plot_raw_wrist_windows(
    data: dict,
    fs_label: int = 700,
    conditions: dict | None = None,
    win_s: int = 60,
    acc_mode: str = "axes",
):
    """
    Plot 60-second windows of raw wrist signals for each condition.
    Keeps the notebook cell short by handling indices and plotting here.
    """

    if conditions is None:
        conditions = {
            1: "Baseline",
            2: "Stress",
            3: "Amusement",
        }

    fs_wrist = {
        "BVP": 64,
        "EDA": 4,
        "TEMP": 4,
        "ACC": 32,
    }

    colors = {
        1: "steelblue",
        2: "firebrick",
        3: "darkorange",
        4: "mediumseagreen",
    }

    labels = np.asarray(data["label"]).astype(int)
    wrist = data["signal"]["wrist"]

    # Flatten 1D signals and compute ACC variants.
    bvp = np.asarray(wrist["BVP"]).flatten()
    eda = np.asarray(wrist["EDA"]).flatten()
    temp = np.asarray(wrist["TEMP"]).flatten()
    acc = np.asarray(wrist["ACC"])
    acc_mag = np.sqrt((acc ** 2).sum(axis=1))

    signal_map = {
        "BVP": (bvp, fs_wrist["BVP"], "BVP (a.u.)"),
        "EDA": (eda, fs_wrist["EDA"], "EDA (uS)"),
        "TEMP": (temp, fs_wrist["TEMP"], "TEMP (C)"),
    }

    # Find first index for each condition in the label timeline.
    starts = {}
    for label_val in conditions:
        idx = np.where(labels == label_val)[0]
        if len(idx) == 0:
            print(f"[warn] Condition {label_val} not found in labels.")
            continue
        starts[label_val] = int(idx[0])

    print("\nSection 5: Visualizing Raw Signals")
    print(f"Window length: {win_s}s per condition")
    print("Conditions:", conditions)

    # Plot one figure per signal with side-by-side panels for conditions.
    for sig_name, (sig_vals, fs_sig, y_label) in signal_map.items():
        win = win_s * fs_sig
        t = np.arange(win) / fs_sig
        fig, axes = plt.subplots(1, len(conditions), figsize=(16, 3), sharey=True)
        fig.suptitle(f"{sig_name} — {win_s}s per condition", fontsize=12)

        for ax, (label_val, label_name) in zip(axes, conditions.items()):
            if label_val not in starts:
                ax.set_title(f"{label_name} (missing)")
                ax.set_xlabel("Time (s)")
                continue
            s_label = starts[label_val]
            s_sig = int(s_label / fs_label * fs_sig)
            ax.plot(
                t,
                sig_vals[s_sig : s_sig + win],
                color=colors.get(label_val, "grey"),
                lw=0.8,
            )
            ax.set_title(label_name)
            ax.set_xlabel("Time (s)")

        axes[0].set_ylabel(y_label)
        plt.tight_layout()
        plt.show()

    # ACC plotting
    acc_fs = fs_wrist["ACC"]
    acc_win = win_s * acc_fs
    t_acc = np.arange(acc_win) / acc_fs

    if acc_mode == "magnitude":
        fig, axes = plt.subplots(1, len(conditions), figsize=(16, 3), sharey=True)
        fig.suptitle(f"ACC |g| — {win_s}s per condition", fontsize=12)
        for ax, (label_val, label_name) in zip(axes, conditions.items()):
            if label_val not in starts:
                ax.set_title(f"{label_name} (missing)")
                ax.set_xlabel("Time (s)")
                continue
            s_label = starts[label_val]
            s_sig = int(s_label / fs_label * acc_fs)
            ax.plot(
                t_acc,
                acc_mag[s_sig : s_sig + acc_win],
                color=colors.get(label_val, "grey"),
                lw=0.8,
            )
            ax.set_title(label_name)
            ax.set_xlabel("Time (s)")
        axes[0].set_ylabel("ACC |g|")
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(
            3,
            len(conditions),
            figsize=(16, 6),
            sharex=True,
            sharey="row",
        )
        fig.suptitle(f"ACC (X, Y, Z) — {win_s}s per condition", fontsize=12)
        if len(conditions) == 1:
            axes = np.array(axes).reshape(3, 1)
        axis_labels = ["ACC X", "ACC Y", "ACC Z"]

        for col, (label_val, label_name) in enumerate(conditions.items()):
            if label_val not in starts:
                for row in range(3):
                    axes[row, col].set_title(f"{label_name} (missing)")
                    axes[row, col].set_xlabel("Time (s)")
                continue
            s_label = starts[label_val]
            s_sig = int(s_label / fs_label * acc_fs)
            for row in range(3):
                axes[row, col].plot(
                    t_acc,
                    acc[s_sig : s_sig + acc_win, row],
                    color=colors.get(label_val, "grey"),
                    lw=0.8,
                )
                if row == 0:
                    axes[row, col].set_title(label_name)
                axes[row, col].set_xlabel("Time (s)")
                axes[row, col].set_ylabel(axis_labels[row])

        plt.tight_layout()
        plt.show()

###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
def plot_meditation_windows(
    data: dict,
    fs_label: int = 700,
    win_s: int = 60,
    acc_mode: str = "axes",
):
    """
    Plot the first 60 seconds of each meditation segment (label 4).
    Useful for comparing the start of each meditation block.
    """

    labels = np.asarray(data["label"]).astype(int)

    # Find contiguous meditation segments
    idx = np.where(labels == 4)[0]
    if len(idx) == 0:
        print("[warn] No meditation segments found (label 4).")
        return

    breaks = np.where(np.diff(idx) > 1)[0]
    starts = [idx[0]] + [idx[b + 1] for b in breaks]
    ends = [idx[b] for b in breaks] + [idx[-1]]
    segments = list(zip(starts, ends))

    print(f"Meditation segments found: {len(segments)}")

    fs_wrist = {
        "BVP": 64,
        "EDA": 4,
        "TEMP": 4,
        "ACC": 32,
    }

    colors = ["mediumseagreen", "seagreen", "darkgreen", "olive"]

    wrist = data["signal"]["wrist"]
    bvp = np.asarray(wrist["BVP"]).flatten()
    eda = np.asarray(wrist["EDA"]).flatten()
    temp = np.asarray(wrist["TEMP"]).flatten()
    acc = np.asarray(wrist["ACC"])
    acc_mag = np.sqrt((acc ** 2).sum(axis=1))

    signal_map = {
        "BVP": (bvp, fs_wrist["BVP"], "BVP (a.u.)"),
        "EDA": (eda, fs_wrist["EDA"], "EDA (uS)"),
        "TEMP": (temp, fs_wrist["TEMP"], "TEMP (C)"),
    }

    # Plot BVP/EDA/TEMP across meditation segments
    for sig_name, (sig_vals, fs_sig, y_label) in signal_map.items():
        win = win_s * fs_sig
        fig, axes = plt.subplots(1, len(segments), figsize=(16, 3), sharey=True)
        fig.suptitle(f"{sig_name} — Meditation segments ({win_s}s)", fontsize=12)
        if len(segments) == 1:
            axes = [axes]

        for i, (start, _end) in enumerate(segments):
            s_sig = int(start / fs_label * fs_sig)
            max_win = len(sig_vals) - s_sig
            win_use = min(win, max_win)
            t = np.arange(win_use) / fs_sig
            if win_use < win:
                print(
                    f"[warn] Meditation {i+1} shorter than {win_s}s for {sig_name} "
                    f"(using {win_use / fs_sig:.1f}s)"
                )
            axes[i].plot(
                t,
                sig_vals[s_sig : s_sig + win_use],
                color=colors[i % len(colors)],
                lw=0.8,
            )
            axes[i].set_title(f"Meditation {i+1}")
            axes[i].set_xlabel("Time (s)")

        axes[0].set_ylabel(y_label)
        plt.tight_layout()
        plt.show()

###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
def butter_lowpass_filter(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4):
    """
    Low-pass Butterworth filter with zero-phase filtering (filtfilt).
    """
    nyq = 0.5 * fs
    cutoff = min(cutoff_hz, 0.99 * nyq)
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, x)


def butter_bandpass_filter(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
):
    """
    Band-pass Butterworth filter with zero-phase filtering (filtfilt).
    """
    nyq = 0.5 * fs
    high = min(high_hz, 0.99 * nyq)
    low = max(low_hz, 0.01)
    if high <= low:
        raise ValueError("Band-pass cutoff invalid: high must be > low.")
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def moving_average(x: np.ndarray, window_samples: int):
    """
    Simple moving average for smoothing.
    """
    if window_samples <= 1:
        return x.copy()
    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    return np.convolve(x, kernel, mode="same")


def snr_proxy_db(raw: np.ndarray, filtered: np.ndarray):
    """
    Simple SNR proxy: 10*log10(var(raw) / var(raw - filtered)).
    """
    noise = raw - filtered
    var_raw = np.var(raw)
    var_noise = np.var(noise)
    if var_noise == 0:
        return float("inf")
    return 10.0 * np.log10(var_raw / var_noise)

    # Plot ACC across meditation segments
    acc_fs = fs_wrist["ACC"]
    acc_win = win_s * acc_fs

    if acc_mode == "magnitude":
        fig, axes = plt.subplots(1, len(segments), figsize=(16, 3), sharey=True)
        fig.suptitle(f"ACC |g| — Meditation segments ({win_s}s)", fontsize=12)
        if len(segments) == 1:
            axes = [axes]
        for i, (start, _end) in enumerate(segments):
            s_sig = int(start / fs_label * acc_fs)
            max_win = len(acc_mag) - s_sig
            win_use = min(acc_win, max_win)
            t = np.arange(win_use) / acc_fs
            axes[i].plot(
                t,
                acc_mag[s_sig : s_sig + win_use],
                color=colors[i % len(colors)],
                lw=0.8,
            )
            axes[i].set_title(f"Meditation {i+1}")
            axes[i].set_xlabel("Time (s)")
        axes[0].set_ylabel("ACC |g|")
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(
            3,
            len(segments),
            figsize=(16, 6),
            sharex=True,
            sharey="row",
        )
        fig.suptitle(f"ACC (X, Y, Z) — Meditation segments ({win_s}s)", fontsize=12)
        if len(segments) == 1:
            axes = np.array(axes).reshape(3, 1)
        axis_labels = ["ACC X", "ACC Y", "ACC Z"]

        for col, (start, _end) in enumerate(segments):
            s_sig = int(start / fs_label * acc_fs)
            max_win = len(acc) - s_sig
            win_use = min(acc_win, max_win)
            t = np.arange(win_use) / acc_fs
            for row in range(3):
                axes[row, col].plot(
                    t,
                    acc[s_sig : s_sig + win_use, row],
                    color=colors[col % len(colors)],
                    lw=0.8,
                )
                if row == 0:
                    axes[row, col].set_title(f"Meditation {col+1}")
                axes[row, col].set_xlabel("Time (s)")
                axes[row, col].set_ylabel(axis_labels[row])

        plt.tight_layout()
        plt.show()