@echo off
cd /d D:\MoodMonitor
call venv\Scripts\activate
start http://127.0.0.1:5000
python app.py
