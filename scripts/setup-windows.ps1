# setup.ps1
# Windows PowerShell version of setup.sh

Write-Host "Step 1/5: Checking if virtualenv is installed"

# Check if virtualenv is available
$virtualenvInstalled = Get-Command virtualenv -ErrorAction SilentlyContinue

if (-not $virtualenvInstalled) {
    Write-Host "Not installed, installing it now"

    if (pip install virtualenv) {
        Write-Host "virtualenv has been successfully installed."
    }
    else {
        Write-Host "Failed to install virtualenv."
        exit 1
    }
}
else {
    Write-Host "virtualenv is already installed."
}

Write-Host "Step 2/5: Creating a virtual environment named 'csci-566-project-venv'"

if (Test-Path "csci-566-project-venv") {
    Write-Host "Virtual environment already exists. Skipping creation."
}
else {
    if (python -m virtualenv csci-566-project-venv) {
        Write-Host "✅ Virtual environment has been successfully created."
    }
    else {
        Write-Host "❌ Failed to create virtual environment."
        exit 1
    }
}

Write-Host "Step 3/5: Ensuring FFmpeg shared build is installed for TorchCodec"

$ffmpegInstalled = Get-Command ffmpeg -ErrorAction SilentlyContinue

if (-not $ffmpegInstalled) {
    $wingetInstalled = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $wingetInstalled) {
        Write-Host "❌ winget is not available and ffmpeg was not found on PATH."
        Write-Host "Please install a shared FFmpeg build manually (recommended: Gyan FFmpeg Shared), then rerun this script."
        exit 1
    }

    Write-Host "ffmpeg not found on PATH, installing Gyan.FFmpeg.Shared via winget..."
    winget install --id Gyan.FFmpeg.Shared --exact --accept-package-agreements --accept-source-agreements --silent

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install FFmpeg Shared via winget."
        exit 1
    }
}
else {
    Write-Host "ffmpeg command is already available."
}

# Keep current shell aware of WinGet command aliases and FFmpeg DLL path.
$wingetLinks = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"
if ((Test-Path $wingetLinks) -and (-not (($env:Path -split ';') -contains $wingetLinks))) {
    $env:Path = "$wingetLinks;$env:Path"
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath) {
    $userPath = ""
}

if ((Test-Path $wingetLinks) -and ($userPath -notlike "*$wingetLinks*")) {
    $userPath = if ([string]::IsNullOrWhiteSpace($userPath)) { $wingetLinks } else { "$wingetLinks;$userPath" }
    [Environment]::SetEnvironmentVariable("Path", $userPath, "User")
}

$pkgRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
$ffmpegDllDir = Get-ChildItem -Path $pkgRoot -Recurse -Filter "avcodec*.dll" -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty DirectoryName

if ($ffmpegDllDir -and (-not (($env:Path -split ';') -contains $ffmpegDllDir))) {
    $env:Path = "$ffmpegDllDir;$env:Path"
}

if ($ffmpegDllDir -and ($userPath -notlike "*$ffmpegDllDir*")) {
    $userPath = if ([string]::IsNullOrWhiteSpace($userPath)) { $ffmpegDllDir } else { "$ffmpegDllDir;$userPath" }
    [Environment]::SetEnvironmentVariable("Path", $userPath, "User")
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "❌ ffmpeg still not available in this shell. Restart PowerShell and rerun the setup script."
    exit 1
}

Write-Host "✅ FFmpeg is available for this setup session."

Write-Host "Step 4/5: Installing dependencies in the virtual environment"

# Path to the virtual environment's python executable
$venvPython = Join-Path -Path "csci-566-project-venv" -ChildPath "Scripts\python.exe"

if (-Not (Test-Path $venvPython)) {
    Write-Host "❌ Could not find python in virtual environment."
    exit 1
}


if (& $venvPython -m pip install --upgrade pip -r requirements.txt) {
    Write-Host "✅ All dependencies have been successfully installed."
}
else {
    Write-Host "❌ Failed to install dependencies."
    exit 1
}

Write-Host "Step 5/5: Authenticate with HuggingFace to access the private dataset. You may have to press enter twice"

$hfTokenFromEnv = $false
$hfToken = $env:HF_TOKEN

function Invoke-HFLogin([string]$pythonExe, [string]$token) {
    $env:HF_TOKEN = $token
    & $pythonExe -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
    return ($LASTEXITCODE -eq 0)
}

if (-not [string]::IsNullOrWhiteSpace($hfToken)) {
    $hfTokenFromEnv = $true
    Write-Host "Found HF_TOKEN in environment. Trying it first."
}
else {
    $hfToken = Read-Host "Enter your Hugging Face token (starts with hf_)"
}

if ([string]::IsNullOrWhiteSpace($hfToken)) {
    Write-Host "❌ No Hugging Face token was provided."
    exit 1
}

$loginSucceeded = Invoke-HFLogin -pythonExe $venvPython -token $hfToken

if (-not $loginSucceeded -and $hfTokenFromEnv) {
    Write-Host "HF_TOKEN from environment appears invalid or expired."
    $hfToken = Read-Host "Paste a fresh Hugging Face token"

    if ([string]::IsNullOrWhiteSpace($hfToken)) {
        Write-Host "❌ No Hugging Face token was provided."
        exit 1
    }

    $loginSucceeded = Invoke-HFLogin -pythonExe $venvPython -token $hfToken
}

if ($loginSucceeded) {
    Write-Host "✅ Successfully logged in to HuggingFace."
}
else {
    Write-Host "❌ Failed to log in to HuggingFace. Check that your token is valid and has dataset read permission."
    exit 1
}

Write-Host "🎉 Yay! We're all set up and you didn't have to do anything! 🎉"