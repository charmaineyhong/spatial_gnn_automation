@echo off
setlocal

REM --- Activate Anaconda base environment ---
call "C:\Users\charm\anaconda3\Scripts\activate.bat" base

REM --- Go to the model folder (where run_inference_from_csv_4class.py lives) ---
cd /d "C:\Users\charm\Spatial GNN\model"

REM --- Call the 4-class inference script ---
REM %1 = nodes.csv, %2 = edges.csv, %3 = predictions.csv (output)
python "run_inference_from_csv_4class.py" ^
  --nodes_csv "%~1" ^
  --edges_csv "%~2" ^
  --out_csv   "%~3" ^
  --model "4trained_model.pth" ^
  --device cpu


exit /b %ERRORLEVEL%
