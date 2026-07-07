from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from extract_signals_candidates_disciplines import (
    FIELDNAMES,
    SourceMeta,
    candidate_rows,
    extract_candidates,
    extract_source_candidates,
    heading_topic_phrases,
    make_candidate,
    normalize_candidate,
    signal_id_for,
)


def source_meta() -> SourceMeta:
    return SourceMeta(
        source_id="test_source",
        title="Lei de Teste",
        source_type="law",
        discipline_id="direito_teste",
        local_path_text="text/test_source.txt",
    )


def alternate_source_meta() -> SourceMeta:
    return SourceMeta(
        source_id="alternate_source",
        title="Lei Alternativa",
        source_type="law",
        discipline_id="direito_teste",
        local_path_text="text/custom_alt_name.txt",
    )


def extract_from_text(text: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_source.txt"
        path.write_text(text, encoding="utf-8")
        candidates, heading_occurrences = extract_source_candidates(source_meta(), path, set())
        return candidates, heading_occurrences


class DisciplineSignalExtractionTest(unittest.TestCase):
    def test_lcp_heading_reconstruction_from_adjacent_blocks(self) -> None:
        candidates, _heading_occurrences = extract_from_text(
            "TÍTULO I\n\n"
            "DOS DIREITOS DO CONSUMIDOR\n\n"
            "CAPÍTULO I\n\n"
            "Da Personalidade e da Capacidade\n\n"
            "Art. 1º Consumidor é toda pessoa física ou jurídica.\n"
        )

        heading_forms = {
            candidate.candidate_normalized
            for candidate in candidates
            if candidate.signal_type == "heading_title"
        }

        self.assertIn("direitos do consumidor", heading_forms)
        self.assertIn("personalidade e da capacidade", heading_forms)

    def test_article_rubric_before_article_is_extracted(self) -> None:
        candidates, _heading_occurrences = extract_from_text(
            "PARTE GERAL\n\n"
            "Tempo do crime\n\n"
            "Art. 4º Considera-se praticado o crime no momento da ação.\n"
        )

        rubrics = {
            candidate.candidate_normalized
            for candidate in candidates
            if candidate.signal_type == "article_rubric"
        }

        self.assertEqual(rubrics, {"tempo do crime"})

    def test_heading_marker_with_suffix_uses_following_title(self) -> None:
        candidates, _heading_occurrences = extract_from_text(
            "Art. 178. O Decreto-Lei passa a vigorar acrescido do seguinte Capítulo II-B:\n\n"
            "“CAPÍTULO II-B\n\n"
            "DOS CRIMES EM LICITAÇÕES E CONTRATOS ADMINISTRATIVOS\n\n"
            "Art. 337-E. Texto legal.\n"
        )

        heading_forms = {
            candidate.candidate_normalized
            for candidate in candidates
            if candidate.signal_type == "heading_title"
        }

        self.assertIn("crimes em licitacoes e contratos administrativos", heading_forms)
        self.assertNotIn("capitulo ii-b", heading_forms)

    def test_dotted_article_number_is_preserved(self) -> None:
        candidates, _heading_occurrences = extract_from_text(
            "Seção I Dos Interditos\n\n"
            "Art. 1.767. Curatela é medida de proteção aos interditos.\n\n"
            "Art. 1.768. O processo que define a curatela deve observar a lei.\n"
        )

        article_numbers = {
            candidate.article_number
            for candidate in candidates
            if candidate.provenance_ref == "Art. 1.767"
        }
        heading_forms = {
            candidate.candidate_normalized
            for candidate in candidates
            if candidate.signal_type == "heading_title"
        }

        self.assertEqual(article_numbers, {"1.767"})
        self.assertIn("interditos", heading_forms)

    def test_definitions_from_article_paragraph_inciso_and_alinea(self) -> None:
        candidates, _heading_occurrences = extract_from_text(
            "Art. 1º Consumidor é toda pessoa física ou jurídica.\n\n"
            "Parágrafo único. Produto é qualquer bem móvel ou imóvel.\n\n"
            "I - meio ambiente, o conjunto de condições, leis e interações.\n\n"
            "a) poluidor, a pessoa física ou jurídica responsável pela degradação.\n"
        )

        by_type = {
            candidate.candidate_normalized: candidate.signal_type
            for candidate in candidates
            if candidate.signal_type in {"defined_term", "enumerated_term"}
        }

        self.assertEqual(by_type["consumidor"], "defined_term")
        self.assertEqual(by_type["produto"], "defined_term")
        self.assertEqual(by_type["meio ambiente"], "enumerated_term")
        self.assertEqual(by_type["poluidor"], "enumerated_term")

    def test_cleanup_and_rejection_rules(self) -> None:
        source = source_meta()
        cleaned = make_candidate(
            "heading_title",
            "ELEIÇÕES109",
            source,
            provenance_kind="heading",
            provenance_ref="line 1",
            hierarchy_path="",
            article_number="",
            stopwords=set(),
        )
        broken = make_candidate(
            "heading_title",
            "ORDEM DOS ADVOGADOS DO BRASIL (OAB",
            source,
            provenance_kind="heading",
            provenance_ref="line 2",
            hierarchy_path="",
            article_number="",
            stopwords=set(),
        )
        boilerplate = make_candidate(
            "heading_title",
            "Disposições Gerais",
            source,
            provenance_kind="heading",
            provenance_ref="line 3",
            hierarchy_path="",
            article_number="",
            stopwords=set(),
        )
        amendment = make_candidate(
            "heading_title",
            "Direitos do Advogado (NR)",
            source,
            provenance_kind="heading",
            provenance_ref="line 4",
            hierarchy_path="",
            article_number="",
            stopwords=set(),
        )

        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned.candidate, "ELEIÇÕES")
        self.assertIsNone(broken)
        self.assertIsNone(boilerplate)
        self.assertIsNotNone(amendment)
        self.assertEqual(amendment.candidate_normalized, "direitos do advogado")

    def test_deterministic_signal_id_and_schema_fields(self) -> None:
        candidate = make_candidate(
            "defined_term",
            "Tributo",
            source_meta(),
            provenance_kind="article",
            provenance_ref="Art. 3",
            hierarchy_path="",
            article_number="3",
            stopwords=set(),
            strong_definition_context=True,
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(signal_id_for(candidate), signal_id_for(candidate))
        self.assertIn("signal_id", FIELDNAMES)
        self.assertIn("specificity_score", FIELDNAMES)
        self.assertEqual(normalize_candidate("Direitos do Advogado (NR)"), "direitos do advogado")

    def test_heading_topic_phrases_keep_per_source_provenance_metadata(self) -> None:
        first_source = source_meta()
        second_source = SourceMeta(
            source_id="other_source",
            title="Outra Lei",
            source_type="law",
            discipline_id="direito_outro",
            local_path_text="text/other_source.txt",
        )
        phrase_candidates = heading_topic_phrases(
            [
                ("Tutela Coletiva Ambiental", first_source, "line 1", "Tutela Coletiva Ambiental"),
                ("Tutela Coletiva Ambiental", first_source, "line 1", "Tutela Coletiva Ambiental"),
                ("Tutela Coletiva Trabalhista", second_source, "line 2", "Tutela Coletiva Trabalhista"),
            ],
            set(),
            min_frequency=2,
        )
        rows = [
            row
            for row in candidate_rows(phrase_candidates)
            if row["candidate_normalized"] == "tutela coletiva"
        ]

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source_id"] for row in rows}, {"test_source", "other_source"})
        self.assertEqual({int(row["source_count"]) for row in rows}, {2})
        self.assertEqual(
            {
                row["source_id"]: row["frequency"]
                for row in rows
            },
            {"test_source": 2, "other_source": 1},
        )
        for row in rows:
            phrase_sources = [
                int(note.split("=", 1)[1])
                for note in str(row["quality_notes"]).split(";")
                if note.startswith("phrase_sources=")
            ]
            self.assertTrue(all(count <= int(row["source_count"]) for count in phrase_sources))

    def test_extract_candidates_resolves_local_path_text_under_alternate_text_dir(self) -> None:
        source = alternate_source_meta()
        with tempfile.TemporaryDirectory() as tmpdir:
            text_dir = Path(tmpdir)
            text_path = text_dir / "custom_alt_name.txt"
            text_path.write_text(
                "Art. 1º Conceito alternativo é toda categoria usada no teste.\n",
                encoding="utf-8",
            )

            candidates, skipped = extract_candidates(
                [source],
                text_dir,
                set(),
                min_phrase_frequency=2,
            )

        defined_terms = {
            candidate.candidate_normalized
            for candidate in candidates
            if candidate.signal_type == "defined_term"
        }
        self.assertEqual(skipped, [])
        self.assertIn("conceito alternativo", defined_terms)


if __name__ == "__main__":
    unittest.main()
