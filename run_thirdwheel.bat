@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title ThirdWheel AI - Object Relationship Analyzer

echo ============================================================
echo        THIRDWHEEL AI - Object Relationship Analyzer
echo             Always watching. Figuring out what objects do.
echo ============================================================
echo.

:: Determine Python executable
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    :: Try system python
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        echo [ERROR] Python not found!
        echo Please install Python 3.12+ and create a virtual environment:
        echo   python -m venv .venv
        echo   .venv\Scripts\pip install -e ".[hub,detector]"
        pause
        exit /b 1
    )
)

echo [INFO] Using Python: %PYTHON_EXE%
echo.

:: If user dragged and dropped a file onto the batch file:
if not "%~1"=="" (
    echo [INFO] Detected dropped file: %~1
    set "EXT=%~x1"
    if /i "!EXT!"==".mp4" goto DRAGGED_VIDEO
    if /i "!EXT!"==".avi" goto DRAGGED_VIDEO
    if /i "!EXT!"==".mov" goto DRAGGED_VIDEO
    if /i "!EXT!"==".mkv" goto DRAGGED_VIDEO
    if /i "!EXT!"==".jpg" goto DRAGGED_IMAGE
    if /i "!EXT!"==".jpeg" goto DRAGGED_IMAGE
    if /i "!EXT!"==".png" goto DRAGGED_IMAGE
    if /i "!EXT!"==".bmp" goto DRAGGED_IMAGE
    if /i "!EXT!"==".tiff" goto DRAGGED_IMAGE
    goto DRAGGED_IMAGE

    :DRAGGED_VIDEO
    set "MODE_PARAM=--mode video --video "%~1""
    goto DEVICE_SELECT

    :DRAGGED_IMAGE
    set "MODE_PARAM=--mode image --image "%~1""
    goto DEVICE_SELECT
)

:: Step 1: Input source selection
echo ------------------------------------------------------------
echo  SELECT INPUT SOURCE:
echo ------------------------------------------------------------
echo   [1] Live Camera / Webcam (Real-Time)
echo   [2] Image File (Photo Analysis)
echo   [3] Video File (Recorded Video)
echo.
set "SRC_CHOICE=1"
set /p "SRC_CHOICE=Select option [1, 2, or 3 - Press ENTER for Camera]: "

if "%SRC_CHOICE%"=="2" goto IMAGE_MODE
if "%SRC_CHOICE%"=="3" goto VIDEO_MODE

:CAMERA_MODE
echo.
echo ------------------------------------------------------------
echo  CAMERA NUMBER SELECTION:
echo ------------------------------------------------------------
echo   0 = Default Laptop Webcam
echo   1 = External USB Camera 1
echo   2 = External USB Camera 2
echo   3 = External USB Camera 3
echo.
set "CAM_NUM=0"
set /p "CAM_NUM=Enter Camera Number [0, 1, 2, 3... - Press ENTER for 0]: "
set "MODE_PARAM=--mode camera --camera !CAM_NUM!"
goto DEVICE_SELECT

:IMAGE_MODE
echo.
echo ------------------------------------------------------------
echo  IMAGE SELECTION:
echo ------------------------------------------------------------
echo Enter path to your image file (.jpg / .png / .bmp / .tiff):
echo (Press ENTER to use sample image assets/reel/images/horse.jpg)
set "IMG_PATH="
set /p "IMG_PATH=Image path: "
if "!IMG_PATH!"=="" set "IMG_PATH=assets/reel/images/horse.jpg"
set "MODE_PARAM=--mode image --image "!IMG_PATH!""
goto DEVICE_SELECT

:VIDEO_MODE
echo.
echo ------------------------------------------------------------
echo  VIDEO SELECTION:
echo ------------------------------------------------------------
echo Enter path to your video file (.mp4 / .avi / .mov / .mkv):
set "VID_PATH="
set /p "VID_PATH=Video path: "
if "!VID_PATH!"=="" (
    echo [ERROR] No video path entered.
    pause
    exit /b 1
)
set "MODE_PARAM=--mode video --video "!VID_PATH!""
goto DEVICE_SELECT

:DEVICE_SELECT
echo.
echo ------------------------------------------------------------
echo  COMPUTE HARDWARE ACCELERATION:
echo ------------------------------------------------------------
echo   [1] GPU (NVIDIA CUDA) - High-Speed Real-Time [Default]
echo   [2] CPU - Standard Processor Mode
echo.
set "DEV_CHOICE=1"
set /p "DEV_CHOICE=Select device [1 or 2 - Press ENTER for GPU]: "

if "%DEV_CHOICE%"=="2" (
    set "DEV=cpu"
    echo [INFO] Selected device: CPU
) else (
    set "DEV=cuda"
    echo [INFO] Selected device: GPU (CUDA)
)

echo.
echo ============================================================
echo  Starting ThirdWheel AI...
echo ============================================================
echo.

"%PYTHON_EXE%" thirdwheel.py !MODE_PARAM! --device !DEV!

echo.
echo ============================================================
echo  ThirdWheel session ended.
echo ============================================================
echo.
pause