@echo off

@setlocal
set "ROOT_DIR=%~dp0.."
set "SDK_DIR=C:\Intel\computer_vision_sdk"

set "SOLUTION_DIR64=%ROOT_DIR%\solution_2017"
if exist "%SDK_DIR%\bin\setupvars.bat" call "%SDK_DIR%\bin\setupvars.bat"

start %SOLUTION_DIR64%\intel64\Debug\cam_stream.exe -d CPU -async -i "cam" ^
-m "%ROOT_DIR%\cam_stream\models\face-detection-adas-0001\FP32\face-detection-adas-0001.xml"

pause