@echo off
REM One-time CUDA Kernel Precompilation
REM ====================================
REM Run this ONCE to compile and cache CUDA kernels

echo ========================================
echo   CUDA Kernel Precompilation
echo ========================================

REM Setup MSVC
SET "VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

IF NOT EXIST "%VCVARSALL%" (
    echo ERROR: Visual Studio 2022 not found
    pause
    exit /b 1
)

echo Setting up MSVC environment...
call "%VCVARSALL%" x64 >nul 2>&1

echo Precompiling kernels...
python src\cuda\precompile_kernels.py

IF ERRORLEVEL 1 (
    echo.
    echo ERROR: Kernel precompilation failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo   DONE! Kernels are now cached.
echo   You can use them normally without
echo   needing MSVC in PATH.
echo ========================================
pause
