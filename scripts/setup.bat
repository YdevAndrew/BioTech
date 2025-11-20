@echo off
echo ===============================
echo   BIO TECH - AMBIENTE LOCAL
echo ===============================

REM Verifica se o Python está acessível via `py`
py --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Erro: Python não encontrado no sistema.
    pause
    exit /b
)

echo Criando ambiente virtual...
py -m venv venv

echo Ativando ambiente virtual...
call venv\Scripts\activate

echo Instalando dependencias...
pip install --upgrade pip
pip install -r requirements.txt

echo Iniciando servidor FastAPI...
cd src
uvicorn discoveryai.api.server:app --reload --port 8000

pause
