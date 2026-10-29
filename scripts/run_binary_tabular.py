#!/usr/bin/env python3
"""Run the four binary real-tabular datasets used in Chapter 10."""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tabular_binary as exp

exp.CONFIG["base_path"] = str(ROOT / "data")
exp.CONFIG["output_dir"] = str(ROOT / "results")
os.makedirs(exp.CONFIG["output_dir"], exist_ok=True)

DATASETS = [
    ("Breast Cancer", exp.load_breast_cancer_data, "breast_cancer"),
    ("German Credit", exp.load_german_credit, "german_credit"),
    ("COMPAS", exp.load_compas, "compas"),
    ("Adult Income", exp.load_adult, "adult_income"),
]

if __name__ == "__main__":
    for name, loader, slug in DATASETS:
        exp.run_all_variants_for_dataset(name, loader, dataset_slug=slug)
