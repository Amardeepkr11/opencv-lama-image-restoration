@echo off
TITLE AI-Based Image Inpainting System (CapStone Project)
echo ============================================================
echo      AI-Based Image Inpainting System (OpenCV & LaMa)
echo ============================================================
echo.
echo Activating Virtual Environment...
call .\venv\Scripts\activate

echo.
echo Starting Streamlit Web Application...
.\venv\Scripts\python.exe -m streamlit run app/main.py

pause
