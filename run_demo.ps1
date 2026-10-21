# ===========================================================================
# Aircraft Engine Digital Twin - one-command demo launcher
# ---------------------------------------------------------------------------
#   .\run_demo.ps1                 -> simulation mode (no FlightGear needed)
#   .\run_demo.ps1 -FlightGear     -> expect FlightGear telnet on :5500
#
# Starts backend.py + a local web server, then opens the dashboard.
# Stop with Ctrl+C or by closing the window (child processes are killed).
# ===========================================================================
param(
    [switch]$FlightGear,
    [int]$WsPort = 8765,
    [int]$HttpPort = 8000
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

# TensorFlow is installed to C:\pylibs via --target; must be on PYTHONPATH
$env:PYTHONPATH = "C:\pylibs"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

if (-not (Test-Path "$here\models\lstm_rul.h5")) {
    Write-Host "No trained models found - running train_models.py first..." -ForegroundColor Yellow
    python train_models.py
    if ($LASTEXITCODE -ne 0) { Write-Error "Training failed"; exit 1 }
}

$backendArgs = @("backend.py", "--ws-port", $WsPort)
if (-not $FlightGear) { $backendArgs += "--simulation" }

$backend = Start-Process python -ArgumentList $backendArgs -PassThru -NoNewWindow
$http    = Start-Process python -ArgumentList @("-m", "http.server", "$HttpPort",
           "--directory", $here) -PassThru -NoNewWindow
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "  Dashboard : http://localhost:$HttpPort/index.html" -ForegroundColor Cyan
Write-Host "  Telemetry : ws://localhost:$WsPort" -ForegroundColor Cyan
if (-not $FlightGear) {
    Write-Host "  Mode      : SIMULATION - type 'fault egt' / 'fault oil' /" -ForegroundColor Yellow
    Write-Host "              'fault mp' here to inject faults, 'clear' to reset." -ForegroundColor Yellow
}
Write-Host "  Press Ctrl+C to stop both services." -ForegroundColor Gray
Write-Host ""

try {
    Start-Process "http://localhost:$HttpPort/index.html"
} catch { }

try {
    while ($true) { Start-Sleep -Seconds 1 }
}
finally {
    foreach ($p in @($backend, $http)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}
