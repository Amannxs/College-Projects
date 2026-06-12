@echo off
call C:\Users\a216a\anaconda3\Scripts\activate.bat drowsy_env
conda activate drowsy_env
C:\Users\a216a\anaconda3\envs\drowsy_env\python.exe -m streamlit run frontend.py
pause