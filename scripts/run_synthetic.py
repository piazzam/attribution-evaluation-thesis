#!/usr/bin/env python3
"""Run the 15 controlled synthetic binary scenarios used by the public replication code."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import synthetic_binary as exp

exp.CONFIG["output_dir"] = str(ROOT / "results")
os.makedirs(exp.CONFIG["output_dir"], exist_ok=True)

if __name__ == "__main__":
    exp.main()
