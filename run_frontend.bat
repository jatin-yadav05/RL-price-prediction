@echo off
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call .\venv\Scripts\activate

echo Installing core dependencies...
pip install --upgrade pip

echo Installing PyTorch...
pip install torch torchvision torchaudio

echo Installing ML dependencies...
pip install tensorboard
pip install stable-baselines3[extra]
pip install scikit-learn gymnasium

echo Installing web dependencies...
pip install streamlit pandas numpy plotly

echo Installing package in development mode...
pip install -e .

echo Verifying installations...
pip list | findstr streamlit
pip list | findstr torch
pip list | findstr tensorboard

echo Creating required directories...
mkdir models 2>nul
mkdir logs 2>nul
mkdir experiments 2>nul

echo Starting frontend...
python -m streamlit run frontend/app.py

pause 