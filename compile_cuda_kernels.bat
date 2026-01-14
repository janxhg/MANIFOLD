@echo off
REM CUDA Kernel Compilation Setup for Windows
REM ==========================================
REM This script sets up MSVC environment and compiles CUDA kernels

echo ========================================
echo   CUDA Kernel Compilation Setup
echo ========================================

REM Find and setup MSVC environment
echo.
echo [1/4] Locating MSVC Build Tools...

REM Try to find vcvarsall.bat (Visual Studio 2022)
SET "VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

IF NOT EXIST "%VCVARSALL%" (
    echo ERROR: Visual Studio 2022 not found at expected location
    echo Please install Visual Studio 2022 with C++ Desktop Development workload
    echo OR manually run vcvarsall.bat before running this script
    pause
    exit /b 1
)

echo Found: %VCVARSALL%

REM Setup MSVC environment for x64
echo.
echo [2/4] Setting up MSVC environment...
call "%VCVARSALL%" x64
IF ERRORLEVEL 1 (
    echo ERROR: Failed to setup MSVC environment
    pause
    exit /b 1
)

echo MSVC environment configured

REM Verify compiler is available
echo.
echo [3/4] Verifying compiler...
where cl >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: cl.exe not found in PATH after MSVC setup
    pause
    exit /b 1
)

echo Compiler available: 
cl /? | findstr /C:"Version"

REM Run Python test to compile kernels
echo.
echo [4/4] Compiling CUDA kernels via PyTorch JIT...
python src\cuda\test_kernel_load.py

IF ERRORLEVEL 1 (
    echo.
    echo WARNING: Kernel compilation encountered issues
    echo PyTorch fallback will be used automatically
) ELSE (
    echo.
    echo ========================================
    echo   SUCCESS: CUDA Kernels Compiled!
    echo ========================================
)

pause
