"""Numerical routines extracted from the original notebook; no GUI side effects."""
import re
from pathlib import Path
import numpy as np
import pandas as pd


def load_gate_spectrum_file(file_path):
    """
    Assumed file format:

    Photon energy(eV)    gate1    gate2    gate3 ...
    energy_1             I        I        I
    energy_2             I        I        I
    ...

    First column: photon energy
    Other column headers: gate voltages
    Values: intensity
    """

    file_path = Path(file_path)

    # sep=None lets pandas infer delimiter: tab, comma, spaces, etc.
    df = pd.read_csv(
        file_path,
        sep=None,
        engine="python",
        comment="#"
    )

    # Remove completely empty rows/columns
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    energy = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()

    gate_labels = df.columns[1:]
    gates = pd.to_numeric(gate_labels, errors="coerce").to_numpy()

    intensity = df.iloc[:, 1:].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy()

    # Remove invalid energy rows
    valid_energy = np.isfinite(energy)
    energy = energy[valid_energy]
    intensity = intensity[valid_energy, :]

    # Remove invalid gate columns
    valid_gates = np.isfinite(gates)
    gates = gates[valid_gates]
    intensity = intensity[:, valid_gates]

    # Sort by energy ascending
    e_order = np.argsort(energy)
    energy = energy[e_order]
    intensity = intensity[e_order, :]

    # Sort by gate ascending
    g_order = np.argsort(gates)
    gates = gates[g_order]
    intensity = intensity[:, g_order]

    return energy, gates, intensity

def get_nearest_gate_spectrum(energy, gates, intensity, target_gate):
    idx = np.argmin(np.abs(gates - target_gate))
    nearest_gate = gates[idx]
    spectrum = intensity[:, idx]
    return nearest_gate, spectrum, idx

def calculate_peak_to_peak_in_window(energy, gates, intensity, energy_window):
    emin, emax = energy_window
    mask = (energy >= emin) & (energy <= emax)

    if not np.any(mask):
        raise ValueError(
            f"No data points found in energy window {energy_window} eV."
        )

    selected_intensity = intensity[mask, :]

    max_vals = np.nanmax(selected_intensity, axis=0)
    min_vals = np.nanmin(selected_intensity, axis=0)
    diff_vals = max_vals - min_vals

    result = pd.DataFrame({
        "gate_voltage_V": gates,
        "max_intensity": max_vals,
        "min_intensity": min_vals,
        "max_minus_min": diff_vals,
    })

    return result

def extract_tg_minus_bg_from_filename(file_name):
    """
    Extract the number after TG-BG=, TG - BG =, or similar patterns.

    Examples:
    P2_0.9TG-BG=4.5 _001_avg4_DR_R_Self.dat -> 4.5
    TG - BG = -2.dat -> -2
    """

    patterns = [
        r"TG\s*-\s*BG\s*=\s*([-+]?\d*\.?\d+)",
        r"TG\s*minus\s*BG\s*=\s*([-+]?\d*\.?\d+)",
        r"TG_BG\s*=\s*([-+]?\d*\.?\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, file_name, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    return np.nan

def extract_temperature_from_filename(file_name):
    """Extract numeric temperature: _1.67KREF -> 1.67, _10KREF -> 10."""
    match = re.search(
        r"_([-+]?(?:\d+(?:\.\d+)?|\.\d+))KREF(?=_|\.|$)",
        file_name,
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else np.nan

def extract_summary_max_in_gate_window(result_df, gate_window):
    """
    From max_minus_min vs gate voltage curve,
    extract the maximum value within a selected gate voltage window.
    """

    gmin, gmax = gate_window
    gate_low = min(gmin, gmax)
    gate_high = max(gmin, gmax)

    mask = (
        (result_df["gate_voltage_V"] >= gate_low) &
        (result_df["gate_voltage_V"] <= gate_high)
    )

    if not np.any(mask):
        return {
            "summary_max_minus_min": np.nan,
            "summary_gate_at_max_V": np.nan,
            "summary_gate_window_low_V": gate_low,
            "summary_gate_window_high_V": gate_high,
        }

    sub_df = result_df.loc[mask].copy()

    idx_max = sub_df["max_minus_min"].idxmax()

    return {
        "summary_max_minus_min": result_df.loc[idx_max, "max_minus_min"],
        "summary_gate_at_max_V": result_df.loc[idx_max, "gate_voltage_V"],
        "summary_gate_window_low_V": gate_low,
        "summary_gate_window_high_V": gate_high,
    }

def build_gate_temperature_matrix(combined_df):
    """Rows: gate voltage (V); columns: temperature (K); values: max-min."""
    valid = combined_df.loc[
        np.isfinite(combined_df["gate_voltage_V"]) &
        np.isfinite(combined_df["temperature_K_from_filename"])
    ].copy()
    if valid.empty:
        return pd.DataFrame()

    # Preserve measurements exactly; do not average repeated temperatures/gates.
    keys = ["gate_voltage_V", "temperature_K_from_filename"]
    if valid.duplicated(subset=keys).any():
        raise ValueError(
            "Multiple intensity values exist for the same gate and temperature. "
            "Select one spectrum file per temperature to export the matrix."
        )

    matrix = valid.pivot(
        index="gate_voltage_V",
        columns="temperature_K_from_filename",
        values="max_minus_min",
    ).sort_index(axis=0).sort_index(axis=1)
    matrix.index.name = "gate_voltage_V"
    matrix.columns.name = "temperature_K"
    return matrix
