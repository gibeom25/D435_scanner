@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem 그냥 실행: scans 안의 모든 스캔을 최신 보정으로 다시 계산
rem 스캔 폴더를 이 아이콘에 끌어다 놓기: 그 스캔만 다시 보정
rem 예전 결과는 각 폴더의 scene_prev.ply 로 보관됩니다
if "%~1"=="" (
  ".venv\Scripts\python.exe" scanner.py --recorrect
) else (
  ".venv\Scripts\python.exe" scanner.py --recorrect %*
)
pause
