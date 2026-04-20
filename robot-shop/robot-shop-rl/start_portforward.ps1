# start_portforward.ps1
# ----------------------
# Starts port-forwarding for Prometheus, Grafana, and Robot Shop.
# Each service runs in its own background job and automatically
# reconnects after 3 seconds if the connection drops.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File start_portforward.ps1
#
# Press Ctrl+C to stop all port-forwards.

$services = @(
    @{ Name = "Prometheus";  Namespace = "istio-system"; Service = "svc/prometheus"; Local = 9090; Remote = 9090 },
    @{ Name = "Grafana";     Namespace = "istio-system"; Service = "svc/grafana";    Local = 3000; Remote = 3000 },
    @{ Name = "Robot Shop";  Namespace = "robot-shop";   Service = "svc/web";        Local = 8080; Remote = 8080 }
)

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Robot Shop RL -- Port Forward Manager" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

$jobs = @()

foreach ($svc in $services) {
    $name      = $svc.Name
    $namespace = $svc.Namespace
    $service   = $svc.Service
    $local     = $svc.Local
    $remote    = $svc.Remote

    Write-Host "  Starting port-forward: $name  ($local -> $remote)" -ForegroundColor Green

    $job = Start-Job -ScriptBlock {
        param($name, $namespace, $service, $local, $remote)
        while ($true) {
            kubectl port-forward -n $namespace $service "${local}:${remote}" 2>&1 | Out-Null
            Start-Sleep -Seconds 3
        }
    } -ArgumentList $name, $namespace, $service, $local, $remote

    $jobs += $job
}

Write-Host ""
Write-Host "  All port-forwards are running." -ForegroundColor Cyan
Write-Host ""
Write-Host "  Prometheus  -->  http://localhost:9090" -ForegroundColor White
Write-Host "  Grafana     -->  http://localhost:3000" -ForegroundColor White
Write-Host "  Robot Shop  -->  http://localhost:8080" -ForegroundColor White
Write-Host ""
Write-Host "  Press Ctrl+C to stop all port-forwards." -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

try {
    while ($true) {
        Start-Sleep -Seconds 10

        foreach ($job in $jobs) {
            if ($job.State -eq "Failed") {
                Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] Job $($job.Id) failed. Restarting..." -ForegroundColor Red
                # Jobs restart themselves internally via the while loop,
                # this is just an extra safety log.
            }
        }
    }
} finally {
    Write-Host ""
    Write-Host "  Stopping all port-forwards..." -ForegroundColor Yellow
    $jobs | Stop-Job
    $jobs | Remove-Job
    Write-Host "  Stopped." -ForegroundColor Green
    Write-Host ""
}
