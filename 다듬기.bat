@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem 메시 파일이나 스캔 폴더를 이 아이콘에 끌어다 놓으세요. 그냥 실행하면 편집기에서 [다듬기] 버튼을 쓰면 됩니다.
if "%~1"=="" (
  ".venv\Scripts\python.exe" scanner.py --load
) else (
  ".venv\Scripts\python.exe" scanner.py --repair "%~1"
)
pause
