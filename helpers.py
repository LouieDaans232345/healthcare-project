# ...

import numpy as np
import pickle
import zipfile
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks
import neurokit2 as nk

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


def _fallback_features(feature_names, prev_features=None, fallback="prev"):
    if fallback == "prev" and prev_features is not None:
        return {name: prev_features.get(name, np.nan) for name in feature_names}
    return {name: np.nan for name in feature_names}


def _slope(values: np.ndarray, fs: float):
    if values.size < 2:
        return np.nan
    t = np.arange(values.size, dtype=float) / float(fs)
    coeff = np.polyfit(t, values, 1)
    return float(coeff[0])


def _sanitize_signal(values: np.ndarray):
    signal = np.asarray(values, dtype=float).flatten()
    if signal.size == 0:
        return signal
    if not np.all(np.isfinite(signal)):
        median = np.nanmedian(signal)
        if not np.isfinite(median):
            median = 0.0
        signal = np.nan_to_num(signal, nan=median, posinf=median, neginf=median)
    return signal


def _normalize_signal(values: np.ndarray):
    signal = _sanitize_signal(values)
    if signal.size == 0:
        return signal
    std = np.std(signal)
    if std == 0 or not np.isfinite(std):
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def _ppg_peak_indices(ppg_clean: np.ndarray, fs_bvp: float):
    try:
        signals, info = nk.ppg_peaks(ppg_clean, sampling_rate=fs_bvp, method="elgendi")
    except Exception:
        signals, info = None, None

    peak_idx = None
    if isinstance(info, dict):
        peak_idx = info.get("PPG_Peaks")
    if peak_idx is None and isinstance(signals, pd.DataFrame) and "PPG_Peaks" in signals:
        peak_idx = np.where(signals["PPG_Peaks"].to_numpy() == 1)[0]

    peak_idx = np.asarray(peak_idx, dtype=int) if peak_idx is not None else np.array([], dtype=int)

    if peak_idx.size < 2:
        norm = _normalize_signal(ppg_clean)
        if norm.size:
            distance = max(1, int(0.3 * fs_bvp))
            prominence = 0.4 * np.std(norm) if np.std(norm) else 0.1
            peaks, _ = find_peaks(norm, distance=distance, prominence=prominence)
            if peaks.size < 2:
                peaks, _ = find_peaks(-norm, distance=distance, prominence=prominence)
            peak_idx = np.asarray(peaks, dtype=int) if peaks.size else peak_idx

    return peak_idx


def _bvp_spectral_fallback(bvp_signal: np.ndarray, fs_bvp: float):
    features = {
        "bvp_hr_mean": np.nan,
        "bvp_hr_std": np.nan,
        "bvp_hrv_mean": np.nan,
        "bvp_hrv_std": np.nan,
        "bvp_hrv_nn50": 0,
        "bvp_hrv_pnn50": 0.0,
        "bvp_hrv_rmssd": 0.0,
        "bvp_hrv_ulf": 0.0,
        "bvp_hrv_lf": 0.0,
        "bvp_hrv_hf": 0.0,
        "bvp_hrv_uhf": 0.0,
        "bvp_hrv_lf_hf": 0.0,
    }

    bvp = _normalize_signal(bvp_signal)
    if bvp.size < 4:
        return features

    freqs, power = _psd_welch(bvp, fs_bvp, 0.7, 3.0)
    if freqs is None or power is None or not np.any(power):
        return features

    total = np.sum(power)
    if total <= 0:
        return features

    freq_peak = freqs[np.argmax(power)]
    freq_mean = float(np.sum(freqs * power) / total)
    freq_var = float(np.sum(((freqs - freq_mean) ** 2) * power) / total)
    freq_std = np.sqrt(max(freq_var, 0.0))

    hr_mean = float(freq_peak * 60.0)
    hr_std = float(freq_std * 60.0)

    rr_mean = 60.0 / hr_mean if hr_mean > 0 else np.nan
    rr_std = abs((60.0 / (hr_mean ** 2)) * hr_std) if hr_mean > 0 else np.nan

    features["bvp_hr_mean"] = hr_mean
    features["bvp_hr_std"] = hr_std
    features["bvp_hrv_mean"] = rr_mean * 1000.0 if np.isfinite(rr_mean) else np.nan
    features["bvp_hrv_std"] = rr_std * 1000.0 if np.isfinite(rr_std) else np.nan
    features["bvp_hrv_rmssd"] = features["bvp_hrv_std"]

    freqs_hrv, power_hrv = _psd_welch(bvp, fs_bvp, 0.01, 1.0)
    if freqs_hrv is not None and power_hrv is not None and np.any(power_hrv):
        features["bvp_hrv_ulf"] = _bandpower(freqs_hrv, power_hrv, 0.01, 0.04)
        features["bvp_hrv_lf"] = _bandpower(freqs_hrv, power_hrv, 0.04, 0.15)
        features["bvp_hrv_hf"] = _bandpower(freqs_hrv, power_hrv, 0.15, 0.4)
        features["bvp_hrv_uhf"] = _bandpower(freqs_hrv, power_hrv, 0.4, 1.0)

    if features["bvp_hrv_hf"]:
        features["bvp_hrv_lf_hf"] = float(features["bvp_hrv_lf"] / features["bvp_hrv_hf"])

    return features


def _bandpower(freqs: np.ndarray, power: np.ndarray, fmin: float, fmax: float):
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return np.nan
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(power[mask], freqs[mask]))
    return float(np.trapz(power[mask], freqs[mask]))


def _integrate_trapz(values: np.ndarray, dx: float = 1.0):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, dx=dx))
    return float(np.trapz(values, dx=dx))


def _psd_welch(signal: np.ndarray, fs: float, fmin: float, fmax: float):
    if signal.size < 4:
        return None, None
    nperseg = min(256, signal.size)
    freqs, power = welch(signal, fs=fs, nperseg=nperseg, detrend="constant")
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], power[mask]


def _psd_neurokit(signal: np.ndarray, fs: float, fmin: float, fmax: float):
    psd = nk.signal_psd(
        signal,
        sampling_rate=fs,
        method="welch",
        min_frequency=fmin,
        max_frequency=fmax,
        normalize=False,
    )
    if isinstance(psd, tuple):
        freqs, power = psd
        return freqs, power
    if isinstance(psd, pd.DataFrame):
        if "Frequency" in psd.columns:
            freqs = psd["Frequency"].to_numpy()
            if "Power" in psd.columns:
                power = psd["Power"].to_numpy()
            else:
                power = psd[psd.columns[-1]].to_numpy()
            return freqs, power
    return None, None


def extract_bvp_features(bvp_window, fs_bvp=64, prev_features=None, fallback="prev"):
    feature_names = [
        "bvp_hr_mean",
        "bvp_hr_std",
        "bvp_hrv_mean",
        "bvp_hrv_std",
        "bvp_hrv_nn50",
        "bvp_hrv_pnn50",
        "bvp_hrv_rmssd",
        "bvp_hrv_ulf",
        "bvp_hrv_lf",
        "bvp_hrv_hf",
        "bvp_hrv_uhf",
        "bvp_hrv_lf_hf",
    ]

    if nk is None:
        raise ImportError("neurokit2 is required for BVP/HRV features")

    bvp = _sanitize_signal(bvp_window)
    if bvp.size < 4:
        return _fallback_features(feature_names, prev_features, fallback)

    try:
        try:
            bvp_clean = nk.ppg_clean(bvp, sampling_rate=fs_bvp)
        except Exception:
            bvp_clean = bvp

        peak_idx = _ppg_peak_indices(bvp_clean, fs_bvp)

        if peak_idx.size < 2:
            return _bvp_spectral_fallback(bvp_clean, fs_bvp)

        ibi = np.diff(peak_idx) / float(fs_bvp)
        ibi_ms = ibi * 1000.0
        hr = 60.0 / ibi

        diff_ibi = np.diff(ibi_ms)
        nn50 = int(np.sum(np.abs(diff_ibi) > 50.0)) if diff_ibi.size else 0
        pnn50 = float(nn50 / diff_ibi.size) if diff_ibi.size else np.nan
        rmssd = float(np.sqrt(np.mean(diff_ibi ** 2))) if diff_ibi.size else np.nan

        features = {
            "bvp_hr_mean": float(np.mean(hr)) if hr.size else np.nan,
            "bvp_hr_std": float(np.std(hr)) if hr.size else np.nan,
            "bvp_hrv_mean": float(np.mean(ibi_ms)) if ibi_ms.size else np.nan,
            "bvp_hrv_std": float(np.std(ibi_ms)) if ibi_ms.size else np.nan,
            "bvp_hrv_nn50": nn50,
            "bvp_hrv_pnn50": pnn50,
            "bvp_hrv_rmssd": rmssd,
        }

        # Frequency-domain HRV features
        ulf = lf = hf = uhf = np.nan
        if ibi.size >= 3:
            t_beats = np.cumsum(ibi)
            fs_hrv = 4.0
            t_uniform = np.arange(0.0, t_beats[-1], 1.0 / fs_hrv)
            ibi_interp = np.interp(t_uniform, t_beats, ibi)

            try:
                freqs, power = _psd_neurokit(ibi_interp, fs_hrv, 0.01, 1.0)
            except Exception:
                freqs, power = None, None

            if freqs is None or power is None:
                freqs, power = _psd_welch(ibi_interp, fs_hrv, 0.01, 1.0)

            if freqs is not None and power is not None and freqs.size:
                ulf = _bandpower(freqs, power, 0.01, 0.04)
                lf = _bandpower(freqs, power, 0.04, 0.15)
                hf = _bandpower(freqs, power, 0.15, 0.4)
                uhf = _bandpower(freqs, power, 0.4, 1.0)

        features["bvp_hrv_ulf"] = ulf
        features["bvp_hrv_lf"] = lf
        features["bvp_hrv_hf"] = hf
        features["bvp_hrv_uhf"] = uhf
        features["bvp_hrv_lf_hf"] = float(lf / hf) if hf and not np.isnan(hf) else np.nan

        return features
    except Exception:
        return _bvp_spectral_fallback(bvp, fs_bvp)


def extract_eda_features(eda_window, fs_eda=4, prev_features=None, fallback="prev"):
    feature_names = [
        "eda_mean",
        "eda_std",
        "eda_min",
        "eda_max",
        "eda_slope",
        "eda_range",
        "eda_scr_peaks",
        "eda_scr_mean_amp",
        "eda_scr_auc",
    ]

    if nk is None:
        raise ImportError("neurokit2 is required for EDA features")

    eda = _sanitize_signal(eda_window)
    if eda.size < 2:
        return _fallback_features(feature_names, prev_features, fallback)

    slope = _slope(eda, fs_eda)
    features = {
        "eda_mean": float(np.mean(eda)),
        "eda_std": float(np.std(eda)),
        "eda_min": float(np.min(eda)),
        "eda_max": float(np.max(eda)),
        "eda_slope": slope,
        "eda_range": float(np.max(eda) - np.min(eda)),
        "eda_scr_peaks": 0,
        "eda_scr_mean_amp": 0.0,
        "eda_scr_auc": 0.0,
    }

    try:
        phasic_df = nk.eda_phasic(eda, sampling_rate=fs_eda, method="smoothmedian")
        if isinstance(phasic_df, tuple):
            phasic_df = phasic_df[0]
        scr = phasic_df["EDA_Phasic"].to_numpy() if "EDA_Phasic" in phasic_df else eda

        peaks_signals, peaks_info = nk.eda_peaks(scr, sampling_rate=fs_eda)
        if isinstance(peaks_signals, tuple):
            peaks_signals = peaks_signals[0]
        peak_idx = peaks_info.get("SCR_Peaks") if isinstance(peaks_info, dict) else None
        if peak_idx is None and isinstance(peaks_signals, pd.DataFrame) and "SCR_Peaks" in peaks_signals:
            peak_idx = np.where(peaks_signals["SCR_Peaks"].to_numpy() == 1)[0]
        peak_idx = np.asarray(peak_idx, dtype=int) if peak_idx is not None else np.array([], dtype=int)

        if peak_idx.size < 1:
            norm = _normalize_signal(scr)
            if norm.size:
                distance = max(1, int(0.5 * fs_eda))
                prominence = 0.3 * np.std(norm) if np.std(norm) else 0.01
                peak_idx, _ = find_peaks(norm, distance=distance, prominence=prominence)
                peak_idx = np.asarray(peak_idx, dtype=int)

        scr_count = int(peak_idx.size)
        if scr_count:
            amps = peaks_info.get("SCR_Peaks_Amplitudes") if isinstance(peaks_info, dict) else None
            if amps is None:
                if isinstance(peaks_signals, pd.DataFrame) and "SCR_Amplitude" in peaks_signals:
                    amps = peaks_signals["SCR_Amplitude"].to_numpy()[peak_idx]
                else:
                    amps = scr[peak_idx]
            scr_mean_amp = float(np.mean(amps)) if np.size(amps) else 0.0
        else:
            scr_mean_amp = 0.0

        scr_auc = _integrate_trapz(np.maximum(scr, 0.0), dx=1.0 / fs_eda)

        features["eda_scr_peaks"] = scr_count
        features["eda_scr_mean_amp"] = scr_mean_amp
        features["eda_scr_auc"] = scr_auc
    except Exception:
        scr_auc = _integrate_trapz(np.maximum(eda, 0.0), dx=1.0 / fs_eda)
        features["eda_scr_auc"] = scr_auc

    return features


def extract_temp_features(temp_window, fs_temp=4, prev_features=None, fallback="prev"):
    feature_names = [
        "temp_mean",
        "temp_std",
        "temp_min",
        "temp_max",
        "temp_range",
        "temp_slope",
    ]

    temp = _sanitize_signal(temp_window)
    if temp.size < 2:
        return _fallback_features(feature_names, prev_features, fallback)

    try:
        return {
            "temp_mean": float(np.mean(temp)),
            "temp_std": float(np.std(temp)),
            "temp_min": float(np.min(temp)),
            "temp_max": float(np.max(temp)),
            "temp_range": float(np.max(temp) - np.min(temp)),
            "temp_slope": _slope(temp, fs_temp),
        }
    except Exception:
        return _fallback_features(feature_names, prev_features, fallback)


def _peak_frequency(signal: np.ndarray, fs: float):
    freqs, power = _psd_welch(signal, fs, 0.0, fs / 2.0)
    if freqs is None or power is None or freqs.size < 2:
        return np.nan
    # skip DC component
    idx = np.argmax(power[1:]) + 1
    return float(freqs[idx])


def extract_acc_features(
    acc_window,
    fs_acc=32,
    prev_features=None,
    fallback="prev",
    smooth_window_s=1.0,
):
    feature_names = [
        "acc_x_mean",
        "acc_x_std",
        "acc_y_mean",
        "acc_y_std",
        "acc_z_mean",
        "acc_z_std",
        "acc_mag_mean",
        "acc_mag_std",
        "acc_x_absint",
        "acc_y_absint",
        "acc_z_absint",
        "acc_x_peakfreq",
        "acc_y_peakfreq",
        "acc_z_peakfreq",
    ]

    acc = np.asarray(acc_window, dtype=float)
    if acc.size and not np.all(np.isfinite(acc)):
        col_medians = np.nanmedian(acc, axis=0)
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
        for col in range(min(acc.shape[1], 3)):
            acc[:, col] = np.where(np.isfinite(acc[:, col]), acc[:, col], col_medians[col])
    if acc.ndim != 2 or acc.shape[1] != 3:
        return _fallback_features(feature_names, prev_features, fallback)

    try:
        x = acc[:, 0]
        y = acc[:, 1]
        z = acc[:, 2]

        if smooth_window_s and smooth_window_s > 0:
            win = int(smooth_window_s * fs_acc)
            x = moving_average(x, win)
            y = moving_average(y, win)
            z = moving_average(z, win)

        mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        dt = 1.0 / float(fs_acc)

        return {
            "acc_x_mean": float(np.mean(x)),
            "acc_x_std": float(np.std(x)),
            "acc_y_mean": float(np.mean(y)),
            "acc_y_std": float(np.std(y)),
            "acc_z_mean": float(np.mean(z)),
            "acc_z_std": float(np.std(z)),
            "acc_mag_mean": float(np.mean(mag)),
            "acc_mag_std": float(np.std(mag)),
            "acc_x_absint": float(np.sum(np.abs(x)) * dt),
            "acc_y_absint": float(np.sum(np.abs(y)) * dt),
            "acc_z_absint": float(np.sum(np.abs(z)) * dt),
            "acc_x_peakfreq": _peak_frequency(x, fs_acc),
            "acc_y_peakfreq": _peak_frequency(y, fs_acc),
            "acc_z_peakfreq": _peak_frequency(z, fs_acc),
        }
    except Exception:
        return _fallback_features(feature_names, prev_features, fallback)


def build_feature_dataframe(
    segments_by_subject,
    fs_wrist=None,
    fallback="prev",
    acc_smooth_window_s=1.0,
    verbose=True,
):
    if fs_wrist is None:
        fs_wrist = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}

    rows = []
    for subject_id, segs in segments_by_subject.items():
        if verbose:
            print(f"Extracting features for Subject {subject_id}...")

        n_windows = len(segs.get("label", []))
        prev_bvp = None
        prev_eda = None
        prev_temp = None
        prev_acc = None

        for idx in range(n_windows):
            bvp_win = segs["BVP"][idx]
            eda_win = segs["EDA"][idx]
            temp_win = segs["TEMP"][idx]
            acc_win = segs["ACC"][idx]

            bvp_feat = extract_bvp_features(bvp_win, fs_wrist["BVP"], prev_bvp, fallback)
            eda_feat = extract_eda_features(eda_win, fs_wrist["EDA"], prev_eda, fallback)
            temp_feat = extract_temp_features(temp_win, fs_wrist["TEMP"], prev_temp, fallback)
            acc_feat = extract_acc_features(
                acc_win,
                fs_wrist["ACC"],
                prev_acc,
                fallback,
                smooth_window_s=acc_smooth_window_s,
            )

            row = {
                "subject": subject_id,
                "window_index": idx,
                "label": int(segs["label"][idx]),
            }
            row.update(bvp_feat)
            row.update(eda_feat)
            row.update(temp_feat)
            row.update(acc_feat)
            rows.append(row)

            prev_bvp = bvp_feat
            prev_eda = eda_feat
            prev_temp = temp_feat
            prev_acc = acc_feat

    return pd.DataFrame(rows)


def build_feature_dataframe_for_subject(
    segments_by_subject,
    subject_id,
    fs_wrist=None,
    fallback="prev",
    acc_smooth_window_s=1.0,
):
    if subject_id not in segments_by_subject:
        raise KeyError(f"Subject {subject_id} not found in segments_by_subject")

    return build_feature_dataframe(
        {subject_id: segments_by_subject[subject_id]},
        fs_wrist=fs_wrist,
        fallback=fallback,
        acc_smooth_window_s=acc_smooth_window_s,
        verbose=True,
    )
