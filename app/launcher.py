"""QuantumPneumoniaLab.exe: start the desktop app with the project's own Python environment.

PyInstaller turns this file into a small .exe (see build_exe.py). The .exe holds no PyTorch or
PennyLane itself: it lives in the project folder and runs app/main.py with .venv's pythonw.exe,
so the app always uses exactly the code and libraries of this project.
"""
import ctypes
import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    root = here.parent if here.name == "app" else here   # the .exe sits in the project root; this file in app/
    pythonw = root / ".venv" / "Scripts" / "pythonw.exe"
    app = root / "app" / "main.py"
    if not (pythonw.exists() and app.exists()):
        ctypes.windll.user32.MessageBoxW(
            None, f"Could not find the project's Python environment:\n{pythonw}\n\nKeep QuantumPneumoniaLab.exe in "
                  "the project folder and create the environment first (see README.md, 'Install').",
            "Quantum Pneumonia Lab", 0x10)
        return
    subprocess.Popen([str(pythonw), str(app)], cwd=root, creationflags=subprocess.DETACHED_PROCESS)


if __name__ == "__main__":
    main()
