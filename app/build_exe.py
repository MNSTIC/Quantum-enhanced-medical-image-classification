"""Build QuantumPneumoniaLab.exe (needs PyInstaller: `.venv\\Scripts\\python.exe -m pip install pyinstaller`).

    .venv\\Scripts\\python.exe app\\build_exe.py

Draws the app icon, then packs app/launcher.py into a one-file, windowless .exe in the project root.
"""
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def draw_icon() -> None:
    """A Bloch-sphere-style icon: a circle with a state arrow, on the app's blue."""
    s = 256
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, s - 8, s - 8), radius=48, fill="#2a78d6")
    d.ellipse((48, 48, s - 48, s - 48), outline="white", width=10)
    d.ellipse((48, 108, s - 48, s - 108), outline=(255, 255, 255, 140), width=5)   # the equator
    d.line((128, 128, 182, 70), fill="#eda100", width=12)                         # the state vector
    d.ellipse((168, 56, 196, 84), fill="#eda100")
    d.ellipse((116, 116, 140, 140), fill="white")
    img.save(HERE / "icon.png")
    img.save(HERE / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def build() -> None:
    build_dir = ROOT / "build"
    subprocess.run([sys.executable, "-m", "PyInstaller", "--onefile", "--noconsole", "--clean",
                    "--name", "QuantumPneumoniaLab", "--icon", str(HERE / "icon.ico"),
                    "--distpath", str(ROOT), "--workpath", str(build_dir), "--specpath", str(build_dir),
                    str(HERE / "launcher.py")], check=True)
    shutil.rmtree(build_dir, ignore_errors=True)
    print(f"built {ROOT / 'QuantumPneumoniaLab.exe'}")


if __name__ == "__main__":
    draw_icon()
    build()
