$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "==========================="
Write-Host "Starting Classroom Monitor"
Write-Host "==========================="

Write-Host "Booting Backend API in new window..."
Start-Process powershell -ArgumentList "-NoExit -File .\start_backend.ps1"

Start-Sleep -Seconds 2

Write-Host "Booting Flutter Frontend in new window..."
Start-Process powershell -ArgumentList "-NoExit -File .\start_frontend.ps1"

Write-Host "Both processes invoked! Keep the windows open to view logs."
