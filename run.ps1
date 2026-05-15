param(
    [switch]$Build
)

$ErrorActionPreference = "Stop"

# Go to deployment folder
Set-Location -Path "$PSScriptRoot\deployment"

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed or not available in PATH."
}

# Ensure .env exists
if (-not (Test-Path "$PSScriptRoot\.env")) {
    if (-not (Test-Path "$PSScriptRoot\.env.example")) {
        throw ".env.example file is missing. Cannot create .env"
    }

    Copy-Item "$PSScriptRoot\.env.example" "$PSScriptRoot\.env"
    Write-Host "Created .env from .env.example. Please add your API keys before first run." -ForegroundColor Yellow
}

# Run Docker Compose
if ($Build) {
    docker compose up --build
} else {
    docker compose up
}