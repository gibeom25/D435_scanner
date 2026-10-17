@echo off
chcp 65001 > nul
rem 실행 파일(dist\D435 Scanner)과 설치 파일(installer\D435_Scanner_Setup_x.y.z.exe)을 만든다.
rem 필요: .venv (requirements.txt + pyinstaller), Inno Setup 6
cd /d "%~dp0.."

".venv\Scripts\python.exe" -m PyInstaller packaging\d435_scanner.spec --noconfirm --workpath build --distpath dist
if errorlevel 1 goto :fail

set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
  echo Inno Setup 6이 없습니다: winget install JRSoftware.InnoSetup
  goto :fail
)
"%ISCC%" packaging\installer.iss
if errorlevel 1 goto :fail
echo.
echo 완료: installer 폴더를 확인하세요.
pause
exit /b 0

:fail
echo 빌드 실패
pause
exit /b 1
