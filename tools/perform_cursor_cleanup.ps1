<#
.SYNOPSIS
    Automates the process of resolving performance issues in Cursor caused by basedpyright analyzing too many .pt files.
.DESCRIPTION
    This script is a comprehensive solution for the system slowdown caused by basedpyright. It performs the following actions:
    1.  Requires Administrator privileges to run.
    2.  Warns the user and waits for confirmation before proceeding.
    3.  Forcefully terminates Cursor and any running basedpyright (node.exe) processes to release file locks.
    4.  Backs up the existing Cursor settings.json file.
    5.  Modifies settings.json to add "**/*.pt" to the "python.analysis.exclude" list, preventing basedpyright from scanning these files.
    6.  Uses the high-performance 'robocopy' utility to move all .pt files from the project root to a specified backup directory, preserving the folder structure.
    7.  Restarts the Cursor application once all operations are complete.
.PARAMETER ProjectRoot
    The root directory of the trading project. Defaults to 'E:\trading'.
.PARAMETER BackupRoot
    The destination directory where .pt files will be moved. Defaults to 'E:\trading_pt_backup'.
.EXAMPLE
    PS C:\> ./tools/perform_cursor_cleanup.ps1
    
    Runs the script with default paths, requires administrative rights.
#>

[CmdletBinding()]
param (
    [string]$ProjectRoot = "E:\trading",
    [string]$BackupRoot = "E:\trading_pt_backup"
)

#Requires -RunAsAdministrator

# --- 초기 설정 및 경고 ---
Clear-Host
Write-Host "=================================================================" -ForegroundColor Yellow
Write-Host "🚀 Cursor 성능 최적화 및 .pt 파일 정리 스크립트" -ForegroundColor Yellow
Write-Host "================================================================="
Write-Host
Write-Host "이 스크립트는 다음 작업을 자동으로 수행합니다:"
Write-Host "1. Cursor 및 basedpyright 관련 프로세스를 강제 종료합니다."
Write-Host "2. Cursor 설정 파일(settings.json)을 백업하고, .pt 파일을 분석에서 제외하도록 수정합니다."
Write-Host "3. '$ProjectRoot' 내의 모든 .pt 파일을 '$BackupRoot' 폴더로 이동합니다. (수백만 개일 경우 매우 오래 걸릴 수 있습니다!)"
Write-Host "4. 모든 작업 완료 후 Cursor를 다시 시작합니다."
Write-Host

Read-Host -Prompt "✅ 작업을 계속하려면 Enter 키를, 중단하려면 Ctrl+C를 누르세요"

# --- 1단계: 프로세스 종료 ---
Write-Host "`n--- 1단계: Cursor 및 관련 프로세스 종료 중... ---" -ForegroundColor Cyan
$ErrorActionPreference = 'SilentlyContinue'
Stop-Process -Name "Cursor" -Force
Get-CimInstance Win32_Process -Filter "Name = 'node.exe' AND CommandLine LIKE '%basedpyright-langserver%'" | ForEach-Object {
    Write-Host "    - 종료 중: PID $($_.ProcessId)"
    Stop-Process -Id $_.ProcessId -Force
}
$ErrorActionPreference = 'Continue'
Write-Host "✅ 프로세스 종료 완료." -ForegroundColor Green
Start-Sleep -Seconds 3

# --- 2단계: Cursor 설정(settings.json) 수정 ---
Write-Host "`n--- 2단계: .pt 파일 분석 제외 설정 추가 중... ---" -ForegroundColor Cyan
$settingsPath = Join-Path $env:APPDATA "Cursor\User\settings.json"
$settingsBackupPath = Join-Path $env:APPDATA "Cursor\User\settings.json.bak"

if (-not (Test-Path $settingsPath)) {
    Write-Warning "Cursor 설정 파일을 찾을 수 없습니다: $settingsPath"
    # Create an empty settings file if it doesn't exist
    New-Item -Path $settingsPath -ItemType File -Value "{}" -Force
}

# 설정 파일 백업
Copy-Item -Path $settingsPath -Destination $settingsBackupPath -Force
Write-Host "    - 기존 설정을 '$settingsBackupPath'에 백업했습니다."

# JSON 파싱 및 수정
$settingsJson = Get-Content $settingsPath -Raw | ConvertFrom-Json
$excludePaths = [System.Collections.Generic.List[string]]($settingsJson.'python.analysis.exclude')

if (-not ($excludePaths -contains "**/*.pt")) {
    $excludePaths.Add("**/*.pt")
    $settingsJson.'python.analysis.exclude' = $excludePaths
    $settingsJson | ConvertTo-Json -Depth 100 | Set-Content -Path $settingsPath -Encoding UTF8
    Write-Host "    - 'python.analysis.exclude'에 '**/*.pt' 패턴을 추가했습니다."
} else {
    Write-Host "    - 분석 제외 설정에 이미 '**/*.pt' 패턴이 존재합니다."
}
Write-Host "✅ 설정 파일 수정 완료." -ForegroundColor Green

# --- 3단계: .pt 파일 이동 ---
Write-Host "`n--- 3단계: robocopy를 사용하여 .pt 파일 이동 시작... ---" -ForegroundColor Cyan
Write-Host "이 작업은 파일 개수에 따라 매우 긴 시간이 소요될 수 있습니다. robocopy 진행 상황을 확인하세요."

if (-not (Test-Path -Path $BackupRoot)) {
    New-Item -Path $BackupRoot -ItemType Directory | Out-Null
    Write-Host "    - 백업 폴더 생성: $BackupRoot"
}

# robocopy 실행
robocopy $ProjectRoot $BackupRoot *.pt /S /MOVE /R:1 /W:1 /MT:16
Write-Host "✅ .pt 파일 이동 완료." -ForegroundColor Green


# --- 4단계: Cursor 재시작 ---
Write-Host "`n--- 4단계: Cursor 재시작 중... ---" -ForegroundColor Cyan
$cursorExePath = Join-Path $env:LOCALAPPDATA "Programs\Cursor\Cursor.exe"
if (Test-Path $cursorExePath) {
    Start-Process -FilePath $cursorExePath
    Write-Host "✅ Cursor가 다시 시작되었습니다." -ForegroundColor Green
} else {
    Write-Warning "Cursor 실행 파일을 찾을 수 없습니다: $cursorExePath"
}

Write-Host "`n🎉 모든 작업이 성공적으로 완료되었습니다!" -ForegroundColor Magenta 