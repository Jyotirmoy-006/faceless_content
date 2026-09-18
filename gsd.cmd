@echo off
set SCRIPT_DIR=%~dp0
if exist "%SCRIPT_DIR%scripts\gsd_cli.py" (
    python "%SCRIPT_DIR%scripts\gsd_cli.py" %*
) else (
    python "%CD%\scripts\gsd_cli.py" %*
)
