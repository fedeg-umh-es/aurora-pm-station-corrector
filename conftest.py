"""
Root conftest.py — adds the project root to sys.path so that
`from scripts.<module> import ...` and `from src.<module> import ...`
resolve correctly in a fresh pytest environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
