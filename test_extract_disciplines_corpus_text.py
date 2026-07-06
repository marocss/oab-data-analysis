from __future__ import annotations

import unittest

from extract_disciplines_corpus_text import ParagraphExtractor


def parse_paragraph_texts(html: str) -> list[str]:
    parser = ParagraphExtractor()
    parser.feed(html)
    parser.close()
    return [paragraph.text for paragraph in parser.paragraphs]


class ParagraphExtractorTest(unittest.TestCase):
    def test_inline_strike_text_is_removed(self) -> None:
        texts = parse_paragraph_texts(
            "<p>I - a postulacao a <strike>qualquer</strike> orgao judicial.</p>"
        )

        self.assertEqual(texts, ["I - a postulacao a orgao judicial."])

    def test_fully_struck_paragraph_is_omitted(self) -> None:
        texts = parse_paragraph_texts(
            "<p><strike>Art. 1o Texto revogado.</strike></p>"
            "<p>Art. 2o Texto vigente.</p>"
        )

        self.assertEqual(texts, ["Art. 2o Texto vigente."])

    def test_nested_markup_inside_strike_does_not_leak(self) -> None:
        texts = parse_paragraph_texts(
            "<p><strike>§ 2o O advogado tem imunidade profissional, "
            "nao constituindo injuria, difamacao "
            "<u><a href='http://example.test'>ou desacato</a></u> puniveis."
            "</strike></p>"
            "<p>§ 2o (Revogado).</p>"
        )

        self.assertEqual(texts, ["§ 2o (Revogado)."])
        self.assertNotIn("ou desacato", "\n".join(texts))

    def test_sup_ordinal_handling_is_preserved(self) -> None:
        texts = parse_paragraph_texts("<p>Art. 7<sup>o</sup>-A. Texto vigente.</p>")

        self.assertEqual(texts, ["Art. 7º -A. Texto vigente."])

    def test_s_tag_is_not_treated_as_deleted_text(self) -> None:
        texts = parse_paragraph_texts("<p>§ 1<s>º</s> Texto vigente.</p>")

        self.assertEqual(texts, ["§ 1º Texto vigente."])

    def test_centered_heading_split_words_are_repaired_within_same_paragraph(self) -> None:
        cases = [
            ("<p align='center'>S<span>eção IV</span></p>", "Seção IV"),
            (
                "<p align='center'>Da Confi<span>dencialidade e suas Exceções</span></p>",
                "Da Confidencialidade e suas Exceções",
            ),
            ("<p align='center'>Disposições C<span>omuns</span></p>", "Disposições Comuns"),
            ("<p align='center'>Da Mediaçã<span>o Judicial</span></p>", "Da Mediação Judicial"),
            ("<p align='center'>Disposiç<span>ões Comuns</span></p>", "Disposições Comuns"),
        ]

        for html, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(parse_paragraph_texts(html), [expected])

    def test_centered_heading_keeps_space_before_uppercase_chunk(self) -> None:
        texts = parse_paragraph_texts(
            "<p align='center'>Dos Mediadores<span>Judiciais</span></p>"
        )

        self.assertEqual(texts, ["Dos Mediadores Judiciais"])

    def test_non_centered_split_words_keep_existing_whitespace_behavior(self) -> None:
        texts = parse_paragraph_texts(
            "<p>Da Confi\n<span>dencialidade e suas Exceções</span></p>"
        )

        self.assertEqual(texts, ["Da Confi dencialidade e suas Exceções"])


if __name__ == "__main__":
    unittest.main()
