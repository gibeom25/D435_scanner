@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem 그냥 실행: 파일 선택 창 / 파일이나 스캔 폴더를 이 아이콘에 끌어다 놓기: 바로 열기
if "%~1"=="" (
  ".venv\Scripts\python.exe" scanner.py --load
) else (
  ".venv\Scripts\python.exe" scanner.py --open "%~1"
)
pause
