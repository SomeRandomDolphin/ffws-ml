"""Aplikasi Dash dashboard DAS Dhompo."""

import sys
from pathlib import Path

# Izinkan `python dashboard\app.py` dipanggil langsung tanpa -m
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dashboard.monitor import app

if __name__ == "__main__":
    app.run(debug=False, port=8050)
