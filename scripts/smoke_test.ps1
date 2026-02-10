# Start port-forward in background
Start-Process -NoNewWindow -FilePath "kubectl" -ArgumentList "port-forward service/cats-dogs-service 8080:80"

Start-Sleep -Seconds 5

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing

    if ($response.Content -match "ok") {
        Write-Host "Smoke test passed"
        exit 0
    } else {
        Write-Error "Health endpoint failed"
        exit 1
    }
}
catch {
    Write-Error "Service not reachable"
    exit 1
}
