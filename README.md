# digital-orrery


You need Python 3.10 or newer and Git.
 
**Windows (PowerShell)**

general setup:
```powershell
git clone https://github.com/swatis6/digital-orrery.git
cd digital-orrery
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python examples/leo_decay.py
```
for bsk_sims specifically:
```powershell
python -m pip install bsk matplotlib
python bsk_sims/teralink_detumble.py
```
 
If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.
 
**macOS / Linux**

general setup:
```bash
git clone https://github.com/swatis6/digital-orrery.git
cd digital-orrery
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python examples/leo_decay.py
```