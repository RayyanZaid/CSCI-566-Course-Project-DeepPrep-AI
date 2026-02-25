# CSCI 566 – Course Project: DeepPrep AI

DeepPrep AI is our group project for **CSCI 566 (Spring 2026)**.

This repository contains:

- **Backend** – Model building and training code
- **Frontend** – Interview Preparation Tool interface

---

## Project Structure

├── backend/ # Model building and training code  
├── frontend/ # Interview Prep Tool  
├── scripts/ # Setup and utility scripts  
├── playground/ # Small sample Python files that show data fetching from RecruitView dataset

---

## Getting Started

### 1. Prerequisites

Make sure you have the following installed:

- Python (3.x recommended)
- pip

You can verify installation with:

```bash
python --version
pip --version
```

---

### 2. Setup Instructions

Run the following commands from the root directory of the project (note: when prompted to enter API Token, enter the one I sent in the chat):

For Mac:

```bash
chmod +x ./scripts/setup.sh   # Give execution permission to the setup script
./scripts/setup.sh            # Run the installation script
```

For Windows:

```bash
.\scripts\setup-windows.ps1
```

(note: you may need to Open PowerShell as Administrator (or regular user) and allow scripts to run if it doesn't let you run for windows)

### 3. Running Python files

Now, with the virtualenv activated, you should be able to run python files. Try by running the `playground/viewTranscript.py` by running this in the terminal

```
python playground/viewTranscript.py
```

Notes:

1. If any steps go wrong in the setup.sh, then try to do them manually
2. When running pip install, remember to run `pip freeze > requirements.txt` so that others don't get errors when running the code
