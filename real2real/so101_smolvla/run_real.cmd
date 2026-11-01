@echo off
rem Run after conda activate lerobot (or your configured environment).
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
python "%~dp0run_real.py" --execute %*
exit /b %errorlevel%
