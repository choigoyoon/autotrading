<#
.SYNOPSIS
    Monitors the memory usage of node.exe processes, which commonly host language servers like basedpyright.
.DESCRIPTION
    This script provides a real-time view of running node.exe processes, showing their process ID, 
    memory consumption in megabytes, and the command-line arguments used to launch them.
    This is useful for identifying resource-heavy language servers before running the cleanup script.
    The view refreshes every 5 seconds. Press Ctrl+C to stop.
.EXAMPLE
    PS C:\> ./tools/monitor_pyright_memory.ps1
    
    This command starts the monitoring loop.
#>

while ($true) {
    Clear-Host
    Write-Host "---" -ForegroundColor Green
    Write-Host "👀 실시간 basedpyright 프로세스 메모리 사용량 모니터링 (node.exe)" -ForegroundColor Green
    Write-Host "---" -ForegroundColor Green
    Write-Host "(5초마다 갱신됩니다. 중지하려면 Ctrl+C를 누르세요.)" -ForegroundColor Cyan
    Write-Host ""

    try {
        $processes = Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" -ErrorAction Stop
        if ($processes) {
            $processes | Select-Object ProcessId, @{Name="메모리 (MB)"; Expression={[math]::Round($_.WorkingSetSize / 1MB, 2)}}, CommandLine `
                      | Format-Table -AutoSize
        } else {
            Write-Host "✅ 현재 실행 중인 'node.exe' 프로세스가 없습니다." -ForegroundColor Green
        }
    } catch {
        Write-Warning "프로세스 정보를 가져오는 중 오류 발생: $($_.Exception.Message)"
    }

    Write-Host "-----------------------------------------------------------------"
    Start-Sleep -Seconds 5
} 