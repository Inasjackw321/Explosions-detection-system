"""Double-click (or `python run_app.py`) to open GulfSeis in your web browser."""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Installing requirements (first run only)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r",
                               os.path.join(HERE, "requirements.txt")])
    os.chdir(HERE)  # so .streamlit/config.toml and static/ are found
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", os.path.join(HERE, "app.py"),
                              "--server.enableStaticServing", "true", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
