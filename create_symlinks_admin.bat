@echo off
echo 🔗 관리자 권한으로 심볼릭 링크 생성 중...

REM 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ 관리자 권한 확인됨
) else (
    echo ❌ 관리자 권한이 필요합니다. 이 파일을 관리자로 실행해주세요.
    pause
    exit /b 1
)

REM 현재 디렉토리
set CURRENT_DIR=%~dp0
set DEV_DIR=%CURRENT_DIR%dev_workspace

echo 📂 현재 디렉토리: %CURRENT_DIR%
echo 📂 개발 디렉토리: %DEV_DIR%

REM 기존 폴더 삭제
if exist "%DEV_DIR%\results" rmdir /s /q "%DEV_DIR%\results"
if exist "%DEV_DIR%\models" rmdir /s /q "%DEV_DIR%\models"
if exist "%DEV_DIR%\logs" rmdir /s /q "%DEV_DIR%\logs"
if exist "%DEV_DIR%\data" rmdir /s /q "%DEV_DIR%\data"

REM 심볼릭 링크 생성
echo 🔗 results 심볼릭 링크 생성...
mklink /D "%DEV_DIR%\results" "%CURRENT_DIR%results"

echo 🔗 models 심볼릭 링크 생성...
mklink /D "%DEV_DIR%\models" "%CURRENT_DIR%models"

echo 🔗 logs 심볼릭 링크 생성...
mklink /D "%DEV_DIR%\logs" "%CURRENT_DIR%logs"

echo 🔗 data 심볼릭 링크 생성...
mklink /D "%DEV_DIR%\data" "%CURRENT_DIR%data"

echo.
echo ✅ 심볼릭 링크 생성 완료!
echo 💡 이제 dev_workspace 폴더를 Cursor에서 열어주세요
pause 