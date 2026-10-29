"""RC analysis, reproducible exports, and figures shared by the GUI and tests."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import json
import os
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from notebook_core import (
    load_gate_spectrum_file, get_nearest_gate_spectrum,
    calculate_peak_to_peak_in_window, extract_tg_minus_bg_from_filename,
    extract_temperature_from_filename, extract_summary_max_in_gate_window,
    build_gate_temperature_matrix,
)

AXES = {
    "temperature": ("temperature_K_from_filename", "Temperature (K)", "temperature"),
    "TG_minus_BG": ("TG_minus_BG_from_filename", "TG - BG from file name (V)", "TG-BG"),
}


def filesystem_path(path):
    """Use extended Windows paths independently of system long-path policy."""
    path = Path(path).resolve()
    text = str(path)
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        text = "\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text
    return Path(text)


def display_path(path):
    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[8:]
    return text[4:] if text.startswith("\\\\?\\") else text


@dataclass(frozen=True)
class Settings:
    target_gate: float = 5.0
    energy_window: tuple = (1.84, 1.88)
    gate_window: tuple = (4.0, 8.0)
    cmap: str = "bwr"
    symmetric: bool = True
    save: bool = True
    dpi: int = 300
    summary_axis: str = "temperature"

    def validate(self):
        if not all(np.isfinite(v) for v in (self.target_gate, *self.energy_window, *self.gate_window)):
            raise ValueError("参数必须是有限数值。")
        if self.energy_window[0] >= self.energy_window[1]:
            raise ValueError("Photon energy 下限必须小于上限。")
        if self.gate_window[0] > self.gate_window[1]:
            raise ValueError("Gate window 下限不能大于上限。")
        if self.cmap not in colormaps:
            raise ValueError(f"未知 colormap: {self.cmap}")
        if not 50 <= self.dpi <= 1200:
            raise ValueError("DPI 必须在 50–1200 之间。")
        if self.summary_axis not in (*AXES, "none"):
            raise ValueError("无效的汇总横轴。")


@dataclass
class Measurement:
    path: Path
    energy: np.ndarray
    gates: np.ndarray
    intensity: np.ndarray
    nearest_gate: float
    spectrum: np.ndarray
    table: pd.DataFrame
    summary: dict
    export_stem: str


@dataclass
class Batch:
    settings: Settings
    measurements: list = field(default_factory=list)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: list = field(default_factory=list)
    output: Path | None = None
    cancelled: bool = False


def analyze_file(path, settings, export_stem=None):
    path = Path(path)
    energy, gates, intensity = load_gate_spectrum_file(filesystem_path(path))
    if not len(energy) or not len(gates) or not np.isfinite(intensity).any():
        raise ValueError("文件中没有有效的 photon energy、gate 或 intensity 数据。")
    nearest, spectrum, _ = get_nearest_gate_spectrum(energy, gates, intensity, settings.target_gate)
    table = calculate_peak_to_peak_in_window(energy, gates, intensity, settings.energy_window)
    if not np.isfinite(table["max_minus_min"]).any():
        raise ValueError("所选能量范围内没有有效的 intensity 数据。")
    table.insert(0, "file_name", path.name)
    table.insert(1, "file_stem", path.stem)
    table.insert(2, "TG_minus_BG_from_filename", extract_tg_minus_bg_from_filename(path.name))
    table.insert(3, "temperature_K_from_filename", extract_temperature_from_filename(path.name))
    # All-NaN gate columns cannot supply a maximum.
    finite = table.loc[np.isfinite(table["max_minus_min"])]
    summary = extract_summary_max_in_gate_window(finite, settings.gate_window)
    summary.update({key: table.iloc[0][key] for key in (
        "file_name", "file_stem", "TG_minus_BG_from_filename", "temperature_K_from_filename")})
    summary.update(energy_window_low_eV=settings.energy_window[0], energy_window_high_eV=settings.energy_window[1])
    return Measurement(path, energy, gates, intensity, nearest, spectrum, table, summary, export_stem or path.stem)


def plot_items(batch):
    items = []
    for i, m in enumerate(batch.measurements):
        items.extend([(f"{m.path.name} · Color map", "color", i),
                      (f"{m.path.name} · Spectrum", "spectrum", i),
                      (f"{m.path.name} · Max − min", "peak", i)])
    if batch.measurements:
        items.append(("汇总 · All max − min curves", "combined", None))
        if batch.settings.summary_axis != "none":
            items.append(("汇总 · Window maximum", "summary", None))
    return items


def make_figure(batch, kind, index=None):
    s = batch.settings
    fig = Figure(figsize=(7.4, 5.2), layout="constrained")
    ax = fig.add_subplot(111)
    low, high = s.gate_window
    if index is not None:
        m = batch.measurements[index]
        ax.set_title(m.path.stem, fontsize=9, wrap=True)
    if kind == "color":
        vmax = np.nanmax(np.abs(m.intensity))
        norm = TwoSlopeNorm(vmin=-(vmax or 1.0), vcenter=0, vmax=(vmax or 1.0)) if s.symmetric else None
        mesh = ax.pcolormesh(m.energy, m.gates, m.intensity.T, shading="auto", cmap=s.cmap, norm=norm)
        fig.colorbar(mesh, ax=ax, label="Intensity")
        ax.axhline(m.nearest_gate, color="k", ls="--", lw=1)
        ax.axhspan(low, high, alpha=0.15)
        ax.set(xlabel="Photon energy (eV)", ylabel="Gate voltage (V)")
    elif kind == "spectrum":
        ax.plot(m.energy, m.spectrum, lw=1.5)
        ax.axvspan(*s.energy_window, alpha=0.2)
        ax.set(xlabel="Photon energy (eV)", ylabel="Intensity")
        ax.set_title(f"{m.path.stem}\nTarget = {s.target_gate:g} V; nearest = {m.nearest_gate:g} V", fontsize=9)
    elif kind in ("peak", "combined"):
        for measurement in ([m] if kind == "peak" else batch.measurements):
            ax.plot(measurement.table["gate_voltage_V"], measurement.table["max_minus_min"],
                    marker="o", markersize=3, lw=1.5, label=measurement.path.stem)
        ax.axvspan(low, high, alpha=0.15)
        ax.set(xlabel="Gate voltage (V)", ylabel="Max - min intensity")
        if kind == "peak" and np.isfinite(m.summary["summary_max_minus_min"]):
            ax.plot(m.summary["summary_gate_at_max_V"], m.summary["summary_max_minus_min"],
                    "k*", markersize=14, label="Window maximum")
        if kind == "combined":
            ax.legend(fontsize=6)
        ax.set_title(f"Max - min intensity\nEnergy window: {s.energy_window[0]:.4f}–{s.energy_window[1]:.4f} eV", fontsize=10)
    elif kind == "summary":
        col, label, title = AXES[s.summary_axis]
        valid = batch.summary.loc[np.isfinite(batch.summary[col]) & np.isfinite(batch.summary["summary_max_minus_min"])]
        if valid.empty:
            ax.text(0.5, 0.5, "No valid filename values or gate-window maxima", ha="center", transform=ax.transAxes)
        else:
            ax.plot(valid[col], valid["summary_max_minus_min"], marker="o", lw=1.5)
            for _, row in valid.iterrows():
                ax.annotate(f'{row["summary_gate_at_max_V"]:.2f} V',
                            (row[col], row["summary_max_minus_min"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(xlabel=label, ylabel="Maximum max-min intensity")
        ax.set_title(f"Maximum max-min intensity vs {title}\nEnergy: {s.energy_window[0]:g}–{s.energy_window[1]:g} eV; gate: {low:g}–{high:g} V", fontsize=10)
    return fig


def save_plot(batch, kind, index, filename):
    fig = make_figure(batch, kind, index)
    try:
        FigureCanvasAgg(fig)
        fig.savefig(batch.output / filename, dpi=batch.settings.dpi, bbox_inches="tight")
    finally:
        fig.clear()


def run_batch(paths, settings, output=None, log: Callable = lambda msg: None,
              progress: Callable = lambda value: None, cancelled: Callable = lambda: False):
    from dataclasses import asdict
    messages = []
    external_log = log

    def log(message):
        messages.append(message)
        external_log(message)

    settings.validate()
    batch = Batch(settings, output=filesystem_path(output) if settings.save and output else None)
    if settings.save and batch.output is None:
        raise ValueError("请选择结果保存目录。")
    if batch.output:
        batch.output.mkdir(parents=True, exist_ok=True)
        manifest = {"settings": asdict(settings), "input_files": [str(Path(p).resolve()) for p in paths]}
        (batch.output / "analysis_settings.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    used_stems = set()
    for i, path in enumerate(paths):
        if cancelled():
            batch.cancelled = True
            break
        path = Path(path)
        log(f"[{i + 1}/{len(paths)}] {path.name}")
        stem = path.stem
        suffix = 2
        while stem.casefold() in used_stems:
            stem = f"{path.stem}__{suffix}"
            suffix += 1
        used_stems.add(stem.casefold())
        try:
            m = analyze_file(path, settings, stem)
        except Exception as exc:
            batch.errors.append(f"{path.name}: {exc}")
            log(f"失败：{batch.errors[-1]}")
            progress(int((i + 1) / len(paths) * 90))
            continue
        batch.measurements.append(m)
        if not np.isfinite(m.summary["summary_max_minus_min"]):
            log("提示：所选 gate window 内没有有效数据；该文件不参与最终最大值曲线。")
        if batch.output:
            try:
                m.table[["gate_voltage_V", "max_minus_min"]].to_csv(batch.output / f"{stem}_max_minus_min.csv", index=False)
                for kind, filename in (
                    ("color", f"{stem}_colorplot.png"),
                    ("spectrum", f"{stem}_spectrum_near_{settings.target_gate:g}V.png"),
                    ("peak", f"{stem}_max_minus_min_vs_gate.png"),
                ):
                    if cancelled():
                        batch.cancelled = True
                        break
                    save_plot(batch, kind, len(batch.measurements) - 1, filename)
            except Exception as exc:
                batch.errors.append(f"{path.name} 保存失败: {exc}")
                log(batch.errors[-1])
        progress(int((i + 1) / len(paths) * 90))
    if batch.measurements:
        batch.summary = pd.DataFrame([m.summary for m in batch.measurements])
        if settings.summary_axis in AXES:
            col = AXES[settings.summary_axis][0]
            batch.summary = batch.summary.sort_values(col, kind="stable", na_position="last")
            invalid = ~(np.isfinite(batch.summary[col]) & np.isfinite(batch.summary["summary_max_minus_min"]))
            if invalid.any():
                log(f"最终汇总跳过 {invalid.sum()} 个文件：文件名横轴数值缺失，或 gate window 内没有有效最大值。")
        if batch.output:
            combined = pd.concat([m.table for m in batch.measurements], ignore_index=True)
            try:
                matrix = build_gate_temperature_matrix(combined)
                if not matrix.empty:
                    matrix.to_csv(batch.output / "combined_max_minus_min_all_files_matrix.csv", index_label="gate_voltage_V")
                else:
                    log("温度矩阵跳过：文件名中未找到有效的 _xxKREF 温度。")
            except ValueError as exc:
                log(f"温度矩阵跳过：{exc}")
            if settings.summary_axis != "none":
                batch.summary.to_csv(batch.output / f"summary_max_minus_min_vs_{settings.summary_axis}.csv", index=False)
            if not cancelled():
                save_plot(batch, "combined", None, "combined_max_minus_min_vs_gate.png")
                if settings.summary_axis != "none" and not cancelled():
                    save_plot(batch, "summary", None, f"summary_max_minus_min_vs_{settings.summary_axis}.png")
    batch.cancelled = batch.cancelled or cancelled()
    if batch.output:
        (batch.output / "analysis_log.txt").write_text(
            f"Analyzed: {len(batch.measurements)} / {len(paths)}\nCancelled: {batch.cancelled}\n" + "\n".join(messages), encoding="utf-8")
    progress(100)
    return batch
