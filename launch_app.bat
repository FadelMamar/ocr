call cd "D:\workspace\repos\ocr"

call .venv\Scripts\activate

@REM load env variables from .env file

call load_env.bat

start streamlit run src/ui.py --server.port 8500
call python src/app.py
