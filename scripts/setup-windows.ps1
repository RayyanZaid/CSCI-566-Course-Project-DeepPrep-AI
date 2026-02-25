# setup.ps1
# Windows PowerShell version of setup.sh

Write-Host "Step 1/4: Checking if virtualenv is installed"

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

Write-Host "Step 2/4: Creating a virtual environment named 'csci-566-project-venv'"

if (Test-Path "csci-566-project-venv") {
    Write-Host "Virtual environment already exists. Skipping creation."
}
else {
    if (virtualenv csci-566-project-venv) {
        Write-Host "✅ Virtual environment has been successfully created."
    }
    else {
        Write-Host "❌ Failed to create virtual environment."
        exit 1
    }
}


Write-Host "Step 3/4: Installing dependencies in the virtual environment"

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

Write-Host "Step 4/4: Authenticate with HuggingFace to access the private dataset. You may have to press enter twice"

if (python -c "from huggingface_hub import login; login()") {
    Write-Host "✅ Successfully logged in to HuggingFace."
}
else {
    Write-Host "❌ Failed to log in to HuggingFace."
    exit 1
}

Write-Host "🎉 Yay! We're all set up and you didn't have to do anything! 🎉"