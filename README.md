# Multi-agent disruption negotiation

Airbus Challenge #2, Autonomous SG 2026. The full design is in [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md).

## Set up

Use a virtual environment. A bare `pip` can belong to a different Python from `python` (Miniconda, say), which installs packages where `python` can't import them. That's why every install is `python -m pip`.

```bash
python -m venv .venv
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Then install:

```bash
python -m pip install -r requirements.txt
```
