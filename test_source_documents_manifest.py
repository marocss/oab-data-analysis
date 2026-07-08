from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES_DIR = ROOT / "sources"
MANIFEST = SOURCES_DIR / "manifests" / "source_documents.csv"

PATH_FIELDS = ("original_path", "raw_text_path", "clean_text_path")
REQUIRED_SOURCE_IDS = {
    "cf_1988",
    "cc_2002",
    "estatuto_oab_1994",
    "regulamento_geral_oab",
    "codigo_etica_oab_2015",
    "provimento_oab_205_2021",
    "provimento_oab_144_2011",
    "provimento_oab_156_2013",
    "provimento_oab_212_2022",
    "cne_rces005_2018",
    "cne_rces002_2021",
    "exame_ordem_numeros_iv",
}


def read_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def path_stays_under_sources(relative_path: str) -> bool:
    if not relative_path:
        return True
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        return False
    resolved = (SOURCES_DIR / path).resolve()
    try:
        resolved.relative_to(SOURCES_DIR.resolve())
    except ValueError:
        return False
    return True


class SourceDocumentsManifestTest(unittest.TestCase):
    def test_source_ids_are_unique(self) -> None:
        rows = read_rows()
        source_ids = [row["source_id"] for row in rows]

        self.assertEqual(len(source_ids), len(set(source_ids)))

    def test_path_fields_stay_under_sources(self) -> None:
        for row in read_rows():
            for field in PATH_FIELDS:
                with self.subTest(source_id=row["source_id"], field=field):
                    self.assertTrue(path_stays_under_sources(row[field]))

    def test_signal_corpus_rows_have_required_classification_fields(self) -> None:
        for row in read_rows():
            if not is_truthy(row["include_in_signal_corpus"]):
                continue
            with self.subTest(source_id=row["source_id"]):
                self.assertTrue(row["discipline_id"])
                self.assertTrue(row["clean_text_path"])

    def test_manifest_does_not_reference_structured_residuals(self) -> None:
        for row in read_rows():
            for field in PATH_FIELDS:
                with self.subTest(source_id=row["source_id"], field=field):
                    self.assertNotIn("structured", Path(row[field]).parts)

    def test_required_current_sources_are_present(self) -> None:
        source_ids = {row["source_id"] for row in read_rows()}

        self.assertTrue(REQUIRED_SOURCE_IDS.issubset(source_ids))


if __name__ == "__main__":
    unittest.main()
