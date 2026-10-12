"""Sample-file and uploaded-SMILES parsing tests."""
from pathlib import Path

import pandas as pd
from rdkit import Chem

from panura_inputs import read_screening_file

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_example_csv_matches_example_txt():
    csv = read_screening_file("PanURA_example.csv", (EXAMPLES / "PanURA_example.csv").read_bytes())
    txt = read_screening_file("PanURA_example.txt", (EXAMPLES / "PanURA_example.txt").read_bytes())
    assert list(csv.columns) == ["ID", "SMILES"]
    assert list(txt.columns) == ["ID", "SMILES"]
    assert len(csv) == len(txt) == 6
    assert csv.ID.tolist() == txt.ID.tolist() == [f"URA_{i:03d}" for i in range(1, 7)]
    assert csv.SMILES.tolist() == txt.SMILES.tolist()
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in csv.SMILES)


def test_txt_accepts_optional_header_comments_and_blank_lines():
    content = b"# Molecular list\nSMILES\n\nCCO\n# comment\nOc1ccccc1\n"
    parsed = read_screening_file("screening.TXT", content)
    assert parsed.to_dict("list") == {"ID": ["Compound_0001", "Compound_0002"], "SMILES": ["CCO", "Oc1ccccc1"]}


def test_txt_without_header_accepts_single_smiles():
    assert read_screening_file("screen.txt", b"CCO\n").to_dict("list") == {"ID": ["Compound_0001"], "SMILES": ["CCO"]}


def test_empty_txt_returns_empty_table_with_smiles_column():
    df = read_screening_file("empty.txt", b"# no compounds\n\n")
    assert df.empty and list(df.columns) == ["ID", "SMILES"]


def test_unsupported_extension_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="Supported"):
        read_screening_file("input.pdf", b"CCO")


def test_csv_preserves_leading_zeros():
    df = read_screening_file("screen.csv", b"ID,SMILES\\n00035,CCO\\n00036,CCC\\n")
    assert df.ID.tolist() == ["00035", "00036"]


def test_single_column_csv_gets_stable_ids():
    df = read_screening_file("screen.csv", b"SMILES\\nCCO\\nCCC\\n")
    assert df.ID.tolist() == ["Compound_0001", "Compound_0002"]


def test_identifier_field_preserved():
    df = read_screening_file("screen.csv", b"Identifier,SMILES\\nLAB1,CCO\\nLAB2,CCC\\n")
    assert df.Identifier.tolist() == ["LAB1", "LAB2"]
