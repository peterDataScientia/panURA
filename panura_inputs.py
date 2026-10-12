"""Import and preserve molecular IDs through PanURA batch screening."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


ID_HEADERS = {"id", "identifier", "compound_id", "molecule_id", "sample_id", "compoundid"}


def ensure_molecule_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve supplied molecular identifiers, otherwise generate stable row IDs.

    IDs are always the first column and never used as QSAR descriptors.
    Existing identifier values (including leading zeroes) are retained.
    """
    df = frame.copy()
    matching = [column for column in df.columns
                if str(column).strip().casefold() in ID_HEADERS]
    if matching:
        id_col = matching[0]
        if not df.empty:
            values = df[id_col].astype("string").str.strip()
            blanks = values.isna() | values.eq("")
            if blanks.any():
                used = set(values.loc[~blanks].tolist())
                for index in df.index[blanks.fillna(False)]:
                    counter = df.index.get_loc(index) + 1
                    generated = f"Compound_{counter:04d}"
                    while generated in used:
                        counter += 1
                        generated = f"Compound_{counter:04d}"
                    values.loc[index] = generated
                    used.add(generated)
                df[id_col] = values
    else:
        id_col = "ID"
        # Avoid overwriting arbitrary columns. An exact ID/identifier is already
        # recognized above, so this is safe for ordinary SMILES-only uploads.
        df.insert(0, id_col, [f"Compound_{i:04d}" for i in range(1, len(df) + 1)])
    return df.loc[:, [id_col, *[col for col in df.columns if col != id_col]]]


def read_screening_file(filename: str, content: bytes) -> pd.DataFrame:
    """Import CSV, XLSX, or TXT with IDs retained.

    TXT accepts either a two-column tab-separated `ID<TAB>SMILES` table,
    or a one-SMILES-per-line list, with optional SMILES header. For a list,
    stable IDs are generated automatically. Comment lines beginning with #
    and empty lines are ignored.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        lines = [
            line.strip()
            for line in content.decode("utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if lines and "\t" in lines[0]:
            records = [line.split("\t", 1) for line in lines]
            if any(len(row) != 2 or not row[0].strip() or not row[1].strip()
                   for row in records):
                raise ValueError("TXT rows must contain ID and SMILES separated by a tab.")
            if [part.strip().casefold() for part in records[0]] == ["id", "smiles"]:
                records.pop(0)
            df = pd.DataFrame(records, columns=["ID", "SMILES"])
            df["ID"] = df["ID"].str.strip()
            df["SMILES"] = df["SMILES"].str.strip()
        else:
            if lines and lines[0].casefold() == "smiles":
                lines.pop(0)
            df = pd.DataFrame({"SMILES": lines})
        return ensure_molecule_ids(df)
    if suffix == ".csv":
        # dtype=str protects identifiers such as 00123 from numeric coercion.
        df = pd.read_csv(BytesIO(content), dtype=str)
        return ensure_molecule_ids(df)
    if suffix == ".xlsx":
        df = pd.read_excel(BytesIO(content), dtype=str)
        return ensure_molecule_ids(df)
    raise ValueError("Supported file formats are CSV, TXT and XLSX.")
