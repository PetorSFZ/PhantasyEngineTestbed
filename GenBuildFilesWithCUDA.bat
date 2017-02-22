@echo off

: Admin escalation script from: https://sites.google.com/site/eneerge/scripts/batchgotadmin
:: BatchGotAdmin
:-------------------------------------
REM  --> Check for permissions
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"

REM --> If error flag set, we do not have admin.
if '%errorlevel%' NEQ '0' (
    echo Requesting administrative privileges...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"

    "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    pushd "%CD%"
    CD /D "%~dp0"
:--------------------------------------

: Create build_cuda directory (and delete old one if it exists)
rmdir /S /Q build_cuda
mkdir build_cuda
cd build_cuda

: Generate Visual Studio solution
cmake .. -G "Visual Studio 14 2015 Win64" -DCUDA_TRACER=TRUE -DPHANTASY_ENGINE_BUILD_TESTS=TRUE

: Create resources symlinks
cd Debug
mklink /D resources\ ..\..\resources

cd ..\RelWithDebInfo
mklink /D resources\ ..\..\resources

cd ..\Release
mklink /D resources\ ..\..\resources

PAUSE