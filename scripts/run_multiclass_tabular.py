#!/usr/bin/env python3
"""Run Glass, Wine Quality (red), and Digits experiments."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tabular_multiclass as exp

exp.CONFIG["base_path"] = str(ROOT / "data")
exp.CONFIG["output_dir"] = str(ROOT / "results")
os.makedirs(exp.CONFIG["output_dir"], exist_ok=True)

if __name__ == "__main__":
    exp.main()
