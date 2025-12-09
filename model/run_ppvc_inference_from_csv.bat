@echo off
chcp 65001 >nul 2>&1
setlocal

set "LOGFILE=%TEMP%\ppvc_inference_log.txt"

echo === PPVC Inference === > "%LOGFILE%"
echo Arguments: %~f1 %~f2 %~f3 %~f4 >> "%LOGFILE%"
echo. >> "%LOGFILE%"

REM --- Use conda run with full paths ---
REM Arguments: %1=nodes.csv %2=edges.csv %3=output.csv %4=annotation.csv (optional)
if "%~4"=="" (
    "C:\Users\charm\anaconda3\condabin\conda.bat" run -n base --no-capture-output python "C:\Users\charm\Spatial GNN\model\automated_inference_workflow.py" --nodes_csv "%~f1" --edges_csv "%~f2" --out_csv "%~f3" --model "C:\Users\charm\Spatial GNN\model\greattrained_model.pth" --device cpu >> "%LOGFILE%" 2>&1
) else (
    "C:\Users\charm\anaconda3\condabin\conda.bat" run -n base --no-capture-output python "C:\Users\charm\Spatial GNN\model\automated_inference_workflow.py" --nodes_csv "%~f1" --edges_csv "%~f2" --annotation_csv "%~f4" --out_csv "%~f3" --model "C:\Users\charm\Spatial GNN\model\greattrained_model.pth" --device cpu --save_cleaned >> "%LOGFILE%" 2>&1
)

set EXITCODE=%ERRORLEVEL%
echo Exit code: %EXITCODE% >> "%LOGFILE%"

if %EXITCODE% NEQ 0 (
    echo ERROR: Check log at %LOGFILE%
    type "%LOGFILE%"
)

exit /b %EXITCODE%
