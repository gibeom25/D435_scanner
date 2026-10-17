@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem 최근 두 스캔(예: 정면 + 위에서)을 이어 붙입니다. 특정 스캔: 스캔 합치기.bat scans\폴더1 scans\폴더2
if "%~1"=="" (
  ".venv\Scripts\python.exe" scanner.py --merge latest2
) else (
  ".venv\Scripts\python.exe" scanner.py --merge %*
)
pause
