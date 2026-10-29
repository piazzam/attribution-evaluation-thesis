#!/usr/bin/env python3
"""Download and normalize the external real-tabular datasets used by the code.

No dataset is redistributed by this repository. This helper downloads public
source files from UCI / ProPublica and converts them to the compact formats
expected by the experiment loaders.
"""
from pathlib import Path
from urllib.request import urlopen
import io
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

URLS = {
    "adult": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "german": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
    "glass": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
    "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "compas": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
}

def fetch(url: str) -> bytes:
    print(f"Downloading {url}")
    with urlopen(url) as r:
        return r.read()

def main():
    (DATA / "adult.csv").write_bytes(fetch(URLS["adult"]))

    german = pd.read_csv(io.BytesIO(fetch(URLS["german"])), sep=r"\s+", header=None)
    german.to_csv(DATA / "german.csv", header=False, index=False)

    glass = pd.read_csv(io.BytesIO(fetch(URLS["glass"])), header=None)
    glass.iloc[:, 1:].to_csv(DATA / "glass.csv", header=False, index=False)

    wine = pd.read_csv(io.BytesIO(fetch(URLS["wine"])), sep=";")
    wine.to_csv(DATA / "winequality-red.csv", header=False, index=False)

    (DATA / "compas.csv").write_bytes(fetch(URLS["compas"]))
    print(f"Datasets prepared in {DATA}")

if __name__ == "__main__":
    main()
