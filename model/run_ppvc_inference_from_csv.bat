@echo off
setlocal

REM --- Activate Anaconda base environment ---
call "C:\Users\charm\anaconda3\Scripts\activate.bat" base

REM --- Go to the model folder (where run_inference_from_csv.py lives) ---
cd /d "C:\Users\charm\Spatial GNN\model"

REM --- Call the inference script ---
REM %1 = nodes.csv, %2 = edges.csv, %3 = predictions.csv (output)
python "run_inference_from_csv.py" ^
  --nodes_csv "%~1" ^
  --edges_csv "%~2" ^
  --out_csv   "%~3" ^
  --device cpu


exit /b %ERRORLEVEL%
