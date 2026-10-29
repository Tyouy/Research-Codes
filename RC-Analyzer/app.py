"""RC analyzer desktop application. Run app.py or the packaged executable."""
import os
import sys
from pathlib import Path
from datetime import datetime
import traceback
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QIcon, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
    QPushButton, QFileDialog, QListWidget, QAbstractItemView, QSplitter,
    QTabWidget, QPlainTextEdit, QProgressBar, QMessageBox, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
)
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from analysis import Settings, run_batch, make_figure, plot_items, display_path


class Worker(QThread):
    message = pyqtSignal(str)
    progress = pyqtSignal(int)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, paths, settings, output):
        super().__init__()
        self.paths, self.settings, self.output = paths, settings, output

    def run(self):
        try:
            batch = run_batch(self.paths, self.settings, self.output,
                              self.message.emit, self.progress.emit, self.isInterruptionRequested)
            self.completed.emit(batch)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RC analyzer")
        self.resize(1380, 850)
        self.setMinimumSize(1120, 680)
        self.batch = None
        self.worker = None
        self.canvas = self.toolbar = None
        self._pending_close = False
        self._smoke_run = False
        self._smoke_failure = None
        self.last_output = None
        icon = Path(__file__).parent / "assets" / "rc_analyzer.ico"
        self.setWindowIcon(QIcon(str(icon)))
        self.statusBar().showMessage("就绪")
        split = QSplitter()
        self.setCentralWidget(split)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(450)
        panel = QWidget()
        self.controls = panel
        left = QVBoxLayout(panel)
        title = QLabel("RC analyzer")
        title.setObjectName("title")
        left.addWidget(title)
        left.addWidget(QLabel("Reflectance contrast · 批量光谱分析"))

        file_group = QGroupBox("1   输入文件")
        fl = QVBoxLayout(file_group)
        buttons = QHBoxLayout()
        for label, callback in (("添加文件…", self.add_files), ("移除选中", self.remove_files), ("清空", self.clear_files)):
            b = QPushButton(label)
            b.clicked.connect(callback)
            buttons.addWidget(b)
        fl.addLayout(buttons)
        self.files = QListWidget()
        self.files.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.files.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.files.setMinimumHeight(100)
        self.files.setMaximumHeight(100)
        fl.addWidget(self.files)
        self.file_count = QLabel("0 个文件 · 支持 .dat / .txt / .csv")
        fl.addWidget(self.file_count)
        left.addWidget(file_group)

        parameter_group = QGroupBox("2   分析参数（对应第一个 cell）")
        form = QFormLayout(parameter_group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.target = self.number(5)
        self.emin, self.emax = self.number(1.84), self.number(1.88)
        self.gmin, self.gmax = self.number(4), self.number(8)
        form.addRow("Target gate (V)", self.target)
        form.addRow("Energy window (eV)", self.pair(self.emin, self.emax))
        form.addRow("Summary gate window (V)", self.pair(self.gmin, self.gmax))
        self.cmap = QComboBox()
        self.cmap.setEditable(True)
        self.cmap.addItems(sorted(colormaps))
        self.cmap.setCurrentText("bwr")
        self.cmap.setMinimumWidth(100)
        self.cmap.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        form.addRow("Colormap", self.cmap)
        self.symmetric = QCheckBox("色标关于 0 对称")
        self.symmetric.setChecked(True)
        form.addRow("Symmetric color", self.symmetric)
        self.save = QCheckBox("保存 PNG 图片和 CSV 数据")
        self.save.setChecked(True)
        form.addRow("SAVE_RESULTS", self.save)
        self.dpi = QSpinBox()
        self.dpi.setRange(50, 1200)
        self.dpi.setValue(300)
        form.addRow("DPI", self.dpi)
        self.axis = QComboBox()
        self.axis.addItem("Temperature（_xxKREF）", "temperature")
        self.axis.addItem("TG − BG（TG-BG=xx）", "TG_minus_BG")
        self.axis.addItem("跳过最终汇总", "none")
        self.axis.setMinimumWidth(100)
        self.axis.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        form.addRow("最终汇总横轴", self.axis)
        self.target.setToolTip("TARGET_GATE：选择最接近该 gate voltage 的光谱。")
        self.emin.setToolTip("ENERGY_WINDOW：在此能量范围内计算 max − min。")
        self.emax.setToolTip(self.emin.toolTip())
        self.gmin.setToolTip("GATE_WINDOW_FOR_SUMMARY_MAX：在此 gate 范围内寻找最大 max − min。")
        self.gmax.setToolTip(self.gmin.toolTip())
        left.addWidget(parameter_group)

        output_group = QGroupBox("3   保存位置")
        ol = QVBoxLayout(output_group)
        row = QHBoxLayout()
        self.output = QLineEdit()
        self.output.setPlaceholderText("默认：首个输入文件所在目录")
        row.addWidget(self.output)
        browse = QPushButton("浏览…")
        browse.clicked.connect(self.choose_output)
        row.addWidget(browse)
        ol.addLayout(row)
        note = QLabel("每次自动创建 RC analyzer results / 时间戳 子文件夹。")
        note.setWordWrap(True)
        ol.addWidget(note)
        self.save.toggled.connect(output_group.setEnabled)
        left.addWidget(output_group)
        self.start = QPushButton("开始分析")
        self.start.setObjectName("primary")
        self.start.clicked.connect(self.start_analysis)
        left.addWidget(self.start)
        left.addStretch()
        scroll.setWidget(panel)
        split.addWidget(scroll)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.tabs = QTabWidget()
        rl.addWidget(self.tabs)
        preview = QWidget()
        pv = QVBoxLayout(preview)
        self.plot_choice = QComboBox()
        self.plot_choice.setMinimumContentsLength(25)
        self.plot_choice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.plot_choice.currentIndexChanged.connect(self.show_plot)
        pv.addWidget(self.plot_choice)
        self.figure_area = QWidget()
        self.figure_layout = QVBoxLayout(self.figure_area)
        self.figure_layout.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QLabel("选择数据文件，调整参数，然后点击「开始分析」。\n\n图形将在这里显示，可缩放、平移或单独保存。")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.figure_layout.addWidget(self.placeholder)
        pv.addWidget(self.figure_area, 1)
        self.tabs.addTab(preview, "图形预览")
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tabs.addTab(self.table, "汇总数据")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.log, "运行日志")
        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        bottom.addWidget(self.progress, 1)
        self.cancel = QPushButton("停止")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancel_analysis)
        bottom.addWidget(self.cancel)
        self.open_output = QPushButton("打开结果文件夹")
        self.open_output.setEnabled(False)
        self.open_output.clicked.connect(self.open_results)
        bottom.addWidget(self.open_output)
        rl.addLayout(bottom)
        split.addWidget(right)
        split.setSizes([470, 910])

    @staticmethod
    def number(value):
        widget = QDoubleSpinBox()
        widget.setRange(-1e8, 1e8)
        widget.setDecimals(6)
        widget.setValue(value)
        widget.setSingleStep(0.01)
        widget.setKeyboardTracking(False)
        widget.setMinimumWidth(100)
        widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        return widget

    @staticmethod
    def pair(low, high):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(low)
        layout.addWidget(QLabel("至"))
        layout.addWidget(high)
        return widget

    def update_count(self):
        self.file_count.setText(f"{self.files.count()} 个文件 · 支持 .dat / .txt / .csv")

    def add_paths(self, paths):
        existing = {self.files.item(i).text().casefold() for i in range(self.files.count())}
        for path in paths:
            path = str(Path(path).resolve())
            if path.casefold() not in existing:
                self.files.addItem(path)
                self.files.item(self.files.count() - 1).setToolTip(path)
                existing.add(path.casefold())
        self.update_count()

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择光谱文件", str(Path.cwd()), "Data files (*.dat *.txt *.csv);;All files (*)")
        self.add_paths(paths)

    def remove_files(self):
        for item in self.files.selectedItems():
            self.files.takeItem(self.files.row(item))
        self.update_count()

    def clear_files(self):
        self.files.clear()
        self.update_count()

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "选择结果保存目录", self.output.text() or str(Path.cwd()))
        if folder:
            self.output.setText(folder)

    def read_settings(self):
        return Settings(self.target.value(), (self.emin.value(), self.emax.value()),
                        (self.gmin.value(), self.gmax.value()), self.cmap.currentText().strip(),
                        self.symmetric.isChecked(), self.save.isChecked(), self.dpi.value(), self.axis.currentData())

    def start_analysis(self):
        if self.worker and self.worker.isRunning():
            return
        try:
            paths = [self.files.item(i).text() for i in range(self.files.count())]
            if not paths:
                raise ValueError("请先添加光谱文件。")
            settings = self.read_settings()
            settings.validate()
            output = None
            if settings.save:
                base = Path(self.output.text().strip() or Path(paths[0]).parent).expanduser()
                output = base / "RC analyzer results" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        except Exception as exc:
            QMessageBox.warning(self, "检查输入", str(exc))
            return
        self.log.clear()
        self.progress.setValue(0)
        self.controls.setEnabled(False)
        self.cancel.setEnabled(True)
        self.open_output.setEnabled(False)
        self.plot_choice.setEnabled(False)
        self.tabs.setCurrentIndex(2)
        self.statusBar().showMessage("正在分析…")
        self.log.appendPlainText(f"TARGET_GATE={settings.target_gate:g}; ENERGY_WINDOW={settings.energy_window}; GATE_WINDOW={settings.gate_window}")
        if output:
            self.log.appendPlainText(f"保存位置：{output}")
        self.worker = Worker(paths, settings, output)
        self.worker.message.connect(self.log.appendPlainText)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.completed.connect(self.analysis_done)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def analysis_done(self, batch):
        self.batch = batch
        self.last_output = batch.output
        self.open_output.setEnabled(bool(batch.output))
        self.plot_choice.blockSignals(True)
        self.plot_choice.clear()
        for label, kind, index in plot_items(batch):
            self.plot_choice.addItem(label, (kind, index))
        self.plot_choice.blockSignals(False)
        self.update_table(batch.summary)
        self.show_plot()
        prefix = "已停止" if batch.cancelled else "完成"
        status = f"{prefix}：成功分析 {len(batch.measurements)} / {self.files.count()} 个文件；错误 {len(batch.errors)} 项。"
        self.statusBar().showMessage(status)
        self.log.appendPlainText(status)
        if not batch.measurements:
            self.tabs.setCurrentIndex(2)
        elif not self._pending_close:
            self.tabs.setCurrentIndex(0)
        if self._smoke_run:
            self._smoke_failure = None if batch.measurements and not batch.errors else "Analysis returned no data or errors"

    def analysis_failed(self, error):
        self.log.appendPlainText(error)
        self.statusBar().showMessage("分析失败，请查看运行日志。")
        self.tabs.setCurrentIndex(2)
        self._smoke_failure = error
        if not self._smoke_run:
            QMessageBox.critical(self, "分析失败", error.splitlines()[-1])

    def worker_finished(self):
        self.controls.setEnabled(True)
        self.cancel.setEnabled(False)
        self.plot_choice.setEnabled(True)
        if self._smoke_run:
            report = Path(os.environ["RC_ANALYZER_SMOKE_REPORT"])
            try:
                # Exercise every plot type, including the numeric summary, in the packaged GUI.
                for i in range(self.plot_choice.count()):
                    self.plot_choice.setCurrentIndex(i)
                    QApplication.processEvents()
                if self._smoke_failure:
                    raise RuntimeError(self._smoke_failure)
                QApplication.processEvents()
                self.grab().save(str(report.with_suffix(".png")))
                report.write_text(f"OK: {len(self.batch.measurements)} measurements; {self.plot_choice.count()} plots; {self.table.rowCount()} summary rows", encoding="utf-8")
                QTimer.singleShot(0, self.close)
            except Exception:
                report.write_text(traceback.format_exc(), encoding="utf-8")
                QTimer.singleShot(0, self.close)
        elif self._pending_close:
            self.close()

    def update_table(self, table):
        self.table.clear()
        self.table.setRowCount(len(table))
        self.table.setColumnCount(len(table.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in table.columns])
        for i, row in enumerate(table.itertuples(index=False, name=None)):
            for j, value in enumerate(row):
                text = f"{value:.8g}" if isinstance(value, (int, float, np.number)) else str(value)
                self.table.setItem(i, j, QTableWidgetItem(text))

    def show_plot(self, _index=None):
        if self.toolbar:
            self.figure_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None
        if self.canvas:
            self.figure_layout.removeWidget(self.canvas)
            self.canvas.figure.clear()
            self.canvas.deleteLater()
            self.canvas = None
        choice = self.plot_choice.currentData()
        self.placeholder.setVisible(choice is None)
        if choice is None or self.batch is None:
            return
        try:
            fig = make_figure(self.batch, *choice)
            self.canvas = FigureCanvasQTAgg(fig)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.figure_layout.addWidget(self.toolbar)
            self.figure_layout.addWidget(self.canvas, 1)
            self.canvas.draw()
        except Exception:
            self.log.appendPlainText(traceback.format_exc())
            self.tabs.setCurrentIndex(2)
            if self._smoke_run:
                self._smoke_failure = "Preview rendering failed"

    def cancel_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.cancel.setEnabled(False)
            self.statusBar().showMessage("正在停止，等待当前文件的操作完成…")

    def open_results(self):
        if self.last_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(display_path(self.last_output)))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self._pending_close = True
            self.cancel_analysis()
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RC analyzer")
    app.setOrganizationName("RC analyzer")
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget { font-family: 'Segoe UI', 'Microsoft YaHei UI'; font-size: 10pt; }
        QMainWindow { background: #f3f6fa; }
        QGroupBox { font-weight: 600; border: 1px solid #d4dce6; border-radius: 7px; margin-top: 12px; padding-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        QPushButton { padding: 7px 10px; border-radius: 5px; border: 1px solid #c6d1df; background: white; }
        QPushButton:hover { background: #eaf1fa; }
        QPushButton:disabled { color: #94a3b8; background: #f1f3f5; }
        QPushButton#primary { color: white; background: #2165ac; font-weight: 600; padding: 12px; }
        QLabel#title { font-size: 23pt; font-weight: 700; color: #173a60; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { padding: 3px; min-height: 19px; }
        QProgressBar { border: 1px solid #d4dce6; border-radius: 5px; text-align: center; }
        QProgressBar::chunk { background: #3988c7; }
    """)
    window = MainWindow()
    window.show()
    smoke_input = os.environ.get("RC_ANALYZER_SMOKE_INPUT")
    if smoke_input and os.environ.get("RC_ANALYZER_SMOKE_REPORT"):
        window._smoke_run = True
        window.add_paths(smoke_input.split(os.pathsep))
        smoke_output = os.environ.get("RC_ANALYZER_SMOKE_OUTPUT")
        window.save.setChecked(bool(smoke_output))
        if smoke_output:
            window.output.setText(smoke_output)
        QTimer.singleShot(100, window.start_analysis)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
