@echo off
echo Running Sentiment Analysis...

REM Check for Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.6+
    pause
    exit /b 1
)

REM Install required dependencies if not present
python -m pip install -q nltk vaderSentiment numpy==1.22.3

REM Create a Python script to handle the analysis
echo import sys > analyze_temp.py
echo import nltk >> analyze_temp.py
echo from nltk.sentiment.vader import SentimentIntensityAnalyzer >> analyze_temp.py
echo. >> analyze_temp.py
echo try: >> analyze_temp.py
echo     nltk.data.find('vader_lexicon') >> analyze_temp.py
echo except: >> analyze_temp.py
echo     nltk.download('vader_lexicon') >> analyze_temp.py
echo. >> analyze_temp.py
echo analyzer = SentimentIntensityAnalyzer() >> analyze_temp.py
echo. >> analyze_temp.py
echo if len(sys.argv) > 1: >> analyze_temp.py
echo     text = ' '.join(sys.argv[1:]) >> analyze_temp.py
echo else: >> analyze_temp.py
echo     text = "This is a sample text to analyze. Please provide your own text as a command-line argument." >> analyze_temp.py
echo. >> analyze_temp.py
echo print('Analyzing: "%s"' %% text) >> analyze_temp.py
echo scores = analyzer.polarity_scores(text) >> analyze_temp.py
echo. >> analyze_temp.py
echo print('Sentiment Scores:') >> analyze_temp.py
echo print('  Positive: %.4f' %% scores['pos']) >> analyze_temp.py
echo print('  Neutral:  %.4f' %% scores['neu']) >> analyze_temp.py
echo print('  Negative: %.4f' %% scores['neg']) >> analyze_temp.py
echo print('  Compound: %.4f' %% scores['compound']) >> analyze_temp.py
echo print() >> analyze_temp.py
echo if scores['compound'] >= 0.05: >> analyze_temp.py
echo     print('Overall Sentiment: POSITIVE') >> analyze_temp.py
echo elif scores['compound'] <= -0.05: >> analyze_temp.py
echo     print('Overall Sentiment: NEGATIVE') >> analyze_temp.py
echo else: >> analyze_temp.py
echo     print('Overall Sentiment: NEUTRAL') >> analyze_temp.py

REM Run the analysis
python analyze_temp.py %*

REM Clean up
del analyze_temp.py

echo.
pause 