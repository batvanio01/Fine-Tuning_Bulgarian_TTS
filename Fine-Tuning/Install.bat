@echo off
cd /d "%~dp0"
python -m venv venv
call venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install --upgrade gradio jinja2 fastapi starlette uvicorn
cd monotonic_align
python setup.py build_ext --inplace
cd ..
call venv\Scripts\deactivate
pause