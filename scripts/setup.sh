#!/bin/bash

echo "Step 1/4: Checking if virtualenv is installed"

if ! command -v virtualenv &> /dev/null
then
    echo "Not installed, installing it now"

    if pip install virtualenv; then
        echo "✅ virtualenv has been successfully installed."
    else
        echo "❌ Failed to install virtualenv."
        exit 1
    fi
else
    echo "✅ virtualenv is already installed."
fi  

echo "Step 2/4: Creating a virtual environment named 'csci-566-project-venv'"

if [ -d "csci-566-project-venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping creation."
else
    if virtualenv csci-566-project-venv; then
        echo "✅ Virtual environment has been successfully created."
    else
        echo "❌ Failed to create virtual environment."
        exit 1
    fi
fi

echo "Step 3/4: Activating the virtual environment and installing dependencies"

source csci-566-project-venv/bin/activate
if pip install -r requirements.txt; then
    echo "✅ All dependencies have been successfully installed."
else
    echo "❌ Failed to install dependencies."
    exit 1
fi

echo "Step 4/4: Authenticate with HuggingFace To Access the Private Dataset. Put in the auth token i sent in the chat"

if python -c "from huggingface_hub import login; login()" ; then
    echo "✅ Successfully logged in to HuggingFace."
else
    echo "❌ Failed to log in to HuggingFace."
    exit 1
fi

echo "🎉 Yay we all set up and u didn't have to do anything! 🎉"