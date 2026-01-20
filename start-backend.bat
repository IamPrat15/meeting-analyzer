@echo off
cd C:\Users\ACC USER\Projects\meeting-analyzer
call venv\Scripts\activate
cd backend
python -m uvicorn main:app --reload
pause