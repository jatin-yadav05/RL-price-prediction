@echo off
echo Activating virtual environment...
call .\venv\Scripts\activate

echo Starting frontend...
streamlit run frontend/app.py

pause 