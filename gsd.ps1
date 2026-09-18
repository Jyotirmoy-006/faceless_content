param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)

$cliPath = Join-Path $PSScriptRoot "scripts\gsd_cli.py"
if (-not (Test-Path $cliPath)) {
    $cliPath = Join-Path (Get-Location) "scripts\gsd_cli.py"
}

if (Test-Path $cliPath) {
    python $cliPath @ArgsList
} else {
    Write-Host "[!] GSD CLI script not found at $cliPath" -ForegroundColor Red
}
