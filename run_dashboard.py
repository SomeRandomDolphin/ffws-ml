"""Jalankan dashboard DAS Dhompo.

    python run_dashboard.py

Buka http://localhost:8050 setelah server siap.
"""

import argparse

from dashboard.app import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Welang Water Monitor — frontend demo")
    parser.add_argument(
        "--port", type=int, default=8050, help="Port HTTP lokal (default: 8050)"
    )
    args = parser.parse_args()
    app.run(debug=False, host="127.0.0.1", port=args.port)
