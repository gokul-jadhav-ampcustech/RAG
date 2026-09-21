<#
.SYNOPSIS
    Runs PostgreSQL with pgvector already built-in, via Docker — no manual
    compiling required.

.PREREQUISITE
    Docker Desktop must be installed and running.
    Get it here: https://www.docker.com/products/docker-desktop/

.USAGE
    powershell -ExecutionPolicy Bypass -File run_pgvector_docker.ps1
#>

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Write-Fail($msg) { Write-Host "    FAILED: $msg" -ForegroundColor Red }

$containerName = "rag-postgres"
$dbUser        = "postgres"
$dbPassword    = 'admin@123'
$dbName        = "RAG"
$hostPort      = 5432

# ---- 1. Check Docker is available and running -----------------------------

Write-Step "Checking Docker"
$docker = Get-Command docker.exe -ErrorAction SilentlyContinue
if (-not $docker) {
    Write-Fail "Docker is not installed or not on PATH."
    Write-Host "    Install Docker Desktop: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    exit 1
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Docker is installed but not running."
    Write-Host "    Start Docker Desktop, wait for it to say 'Engine running', then re-run this script." -ForegroundColor Yellow
    exit 1
}
Write-Ok "Docker is installed and running."

# ---- 2. Check whether the container already exists -------------------------

Write-Step "Checking for existing container named '$containerName'"
$existing = docker ps -a --filter "name=^/$containerName$" --format "{{.Names}}"

if ($existing -eq $containerName) {
    $status = docker inspect -f "{{.State.Status}}" $containerName
    if ($status -eq "running") {
        Write-Ok "Container '$containerName' is already running. Nothing to do."
        docker ps --filter "name=$containerName"
        exit 0
    } else {
        Write-Host "    Container exists but is stopped. Starting it..." -ForegroundColor Yellow
        docker start $containerName | Out-Null
        Write-Ok "Started existing container '$containerName'."
        exit 0
    }
}

# ---- 3. Check if port 5432 is already taken (e.g. a native Postgres service) --

Write-Step "Checking if port $hostPort is free"
$portInUse = Test-NetConnection -ComputerName "localhost" -Port $hostPort -WarningAction SilentlyContinue

if ($portInUse.TcpTestSucceeded) {
    Write-Host "    Port $hostPort is already in use (likely a native PostgreSQL service)." -ForegroundColor Yellow
    Write-Host "    Either stop that service (services.msc -> postgresql-x64-<version> -> Stop)," -ForegroundColor Yellow
    Write-Host "    or this script will map the container to port 5433 instead." -ForegroundColor Yellow
    $hostPort = 5433
}

# ---- 4. Run the pgvector-enabled Postgres container -------------------------

Write-Step "Starting PostgreSQL + pgvector container on port $hostPort"

docker run --name $containerName `
    -e POSTGRES_USER=$dbUser `
    -e POSTGRES_PASSWORD=$dbPassword `
    -e POSTGRES_DB=$dbName `
    -p "${hostPort}:5432" `
    -d pgvector/pgvector:pg16

if ($LASTEXITCODE -ne 0) {
    Write-Fail "Failed to start the container. See the error above."
    exit 1
}
Write-Ok "Container '$containerName' started."

# ---- 5. Wait for Postgres to accept connections ------------------------------

Write-Step "Waiting for PostgreSQL to be ready"
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    docker exec $containerName pg_isready -U $dbUser *> $null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
}

if ($ready) {
    Write-Ok "PostgreSQL is ready to accept connections."
} else {
    Write-Host "    Still starting up — give it a few more seconds before running the app." -ForegroundColor Yellow
}

# ---- 6. Print the .env line to use ------------------------------------------

$encodedPassword = $dbPassword -replace '@', '%40'
$databaseUrl = "postgresql://${dbUser}:${encodedPassword}@localhost:${hostPort}/${dbName}"

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " Postgres + pgvector is running in Docker." -ForegroundColor Green
Write-Host " Make sure this line is in your .env file:" -ForegroundColor Green
Write-Host ""
Write-Host " DATABASE_URL=$databaseUrl" -ForegroundColor White
Write-Host ""
Write-Host " Then run:  python run.py" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green
