# Movie Recommender

Project structure for a movie recommendation system.

## Environment

This project uses a local Windows virtual environment at `.venv`.

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Pip is configured for this virtual environment with `timeout = 120` in `.venv\pip.ini`.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python tests\verify_environment.py
```
