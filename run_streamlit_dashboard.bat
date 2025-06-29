@echo off
echo Starting Sentiment Analysis Streamlit Dashboard...

REM Check for Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.6+
    pause
    exit /b 1
)

REM Install required dependencies if not present
echo Checking and installing required dependencies...
python -m pip install -q streamlit validators "tzlocal<6" "importlib-metadata<7" "packaging<24" "protobuf<5" "rich<14" "tenacity<9" numpy==1.22.3 vaderSentiment

REM Run the Streamlit app
echo Launching Streamlit dashboard...
streamlit run sentiment_streamlit_app.py

pause 