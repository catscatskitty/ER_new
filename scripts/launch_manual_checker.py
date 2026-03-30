#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent.parent / "src" / "manual_check" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path)])

if __name__ == "__main__":
    main()