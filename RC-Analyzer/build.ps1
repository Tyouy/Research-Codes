$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:PYINSTALLER_CONFIG_DIR = Join-Path $PSScriptRoot 'build\pyinstaller-cache'
    python -m PyInstaller --noconfirm --clean --onefile --windowed --name 'RC analyzer' --icon 'assets\rc_analyzer.ico' --add-data 'assets\rc_analyzer.ico;assets' --exclude-module PyQt5 --exclude-module PySide6 --exclude-module tkinter --exclude-module IPython --exclude-module jupyter --exclude-module scipy --exclude-module pytest app.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }
    Write-Output "Built: $PSScriptRoot\dist\RC analyzer.exe"
}
finally { Pop-Location }
