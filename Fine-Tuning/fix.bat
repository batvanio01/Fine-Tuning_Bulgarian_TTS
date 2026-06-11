@echo off
cd /d "%~dp0"
call venv\Scripts\activate
cd monotonic_align
python setup.py build_ext --inplace
cd ..
call venv\Scripts\deactivate
pause