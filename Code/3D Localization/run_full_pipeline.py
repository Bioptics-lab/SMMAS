# -*- coding: utf-8 -*-
"""Run the TPLFM 4-view localization pipeline (implementation lives in code/)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CODE))
runpy.run_path(str(CODE / "run_full_pipeline.py"), run_name="__main__")
