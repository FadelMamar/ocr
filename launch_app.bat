call cd "D:\workspace\repos\ocr"

call .venv\Scripts\activate

@REM load env variables from .env file

if exist .env (
    echo Loading .env file...
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        REM Skip lines starting with # (comments)
        echo %%a | findstr /r "^#" >nul
        if errorlevel 1 (
            REM Skip empty lines
            if not "%%a"=="" (
                set "%%a=%%b"
                @REM echo Set %%a=%%b
            )
        )
    )
) else (
    echo .env file not found
)

start streamlit run src/ui.py --server.port 8500
call python src\app.py
