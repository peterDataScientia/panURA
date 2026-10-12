"""Read PanURA batch-screening datasets in CSV, Excel or plain-text SMILES format."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


def read_screening_file(filename: str, content: bytes) -> pd.DataFrame:
    """Parse the supported upload formats without evaluating molecular activity.

    TXT format: one SMILES string per line, optionally preceded by 'SMILES'.
    Blank lines and comment lines beginning with # are ignored.
    CSV/XLSX format: columns with a header row; users select the SMILES column.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        lines = content.decode("utf-8-sig").splitlines()
        values = [line.strip() for line in lines
                  if line.strip() and not line.lstrip().startswith("#")]
        if values and values[0].casefold() == "smiles":
            values.pop(0)
        return pd.DataFrame({"SMILES": values})
    if suffix == ".csv":
        return pd.read_csv(BytesIO(content))
    if suffix == ".xlsx":
        return pd.read_excel(BytesIO(content))
    raise ValueError("Supported file formats are CSV, TXT and XLSX.")
