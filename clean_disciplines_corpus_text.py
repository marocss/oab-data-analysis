from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from extract_disciplines_corpus_text import (
    DEFAULT_LCP95_HTML,
    DEFAULT_MANIFEST,
    HIERARCHY_MARKER_RE,
    SOURCES_DIR,
    fold_for_match,
    summarize_output,
    validate_lcp95_reference,
)


ARTICLE_RE = re.compile(r"^Art\.\s+")
SIGNATURE_START_RE = re.compile(r"^(BRAS[IÍ]LIA|RIO DE JANEIRO)\s*,", re.I)

CHROME_EXACT = {
    "ATOS DECORRENTES DO DISPOSTO NO § 3o DO ART. 5o",
    "EMENDAS CONSTITUCIONAIS",
    "EMENDAS CONSTITUCIONAIS DE REVISAO",
    "INDICE",
    "INDICE TEMATICO",
    "MENSAGEM DE VETO",
    "PROMULGACAO PARTES VETADAS",
    "PRODUCAO DE EFEITOS",
    "REGULAMENTO",
    "VIGENCIA",
}

CHROME_LINE_RE = re.compile(
    r"^(?:"
    r"Vigência|"
    r"Mensagem de veto|"
    r"Regulamento|"
    r"Promulgação partes vetadas|"
    r"Produção de efeitos"
    r")(?:\s+(?:Vigência|Mensagem de veto|Regulamento|Promulgação partes vetadas|Produção de efeitos))*$",
    re.I,
)


def split_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]


def join_blocks(blocks: list[str]) -> str:
    return "\n\n".join(blocks).strip() + "\n"


def is_hierarchy_block(block: str) -> bool:
    return HIERARCHY_MARKER_RE.fullmatch(block) is not None


def is_article_block(block: str) -> bool:
    return ARTICLE_RE.match(block) is not None


def is_legal_restart(block: str) -> bool:
    folded = fold_for_match(block)
    return (
        folded.startswith("ATO DAS DISPOSICOES CONSTITUCIONAIS TRANSITORIAS")
        or is_article_block(block)
        or is_hierarchy_block(block)
    )


def is_signature_start(block: str) -> bool:
    return SIGNATURE_START_RE.match(block.strip()) is not None


def is_chrome_block(block: str) -> bool:
    folded = fold_for_match(block)
    if folded in CHROME_EXACT:
        return True
    if folded.startswith("ATOS DECORRENTES DO DISPOSTO"):
        return True
    if CHROME_LINE_RE.fullmatch(block.strip()):
        return True
    if folded.startswith("PRESIDENCIA DA REPUBLICA"):
        return True
    if folded.startswith("VIGENCIA ") or folded.startswith("MENSAGEM DE VETO "):
        return True
    if folded.startswith("VIDE ") or folded.startswith("(VIDE "):
        return True
    if folded.startswith("ESTE TEXTO NAO SUBSTITUI"):
        return True
    if folded.startswith("DOWNLOAD PARA ANEXO"):
        return True
    if folded in {"*"}:
        return True
    if folded.startswith("PARTICIPANTES:") or folded.startswith("IN MEMORIAM:"):
        return True
    return False


def remove_noise_blocks(blocks: list[str]) -> list[str]:
    output: list[str] = []
    previous_folded = ""
    skipping_signature = False
    seen_article = False

    for block in blocks:
        if skipping_signature:
            if is_legal_restart(block):
                skipping_signature = False
            else:
                continue

        if is_signature_start(block):
            skipping_signature = True
            continue

        folded = fold_for_match(block)
        if folded.startswith("ATO DAS DISPOSICOES CONSTITUCIONAIS TRANSITORIAS") and not seen_article:
            continue

        if is_chrome_block(block):
            continue

        if folded == previous_folded:
            continue

        output.append(block)
        previous_folded = folded
        if is_article_block(block):
            seen_article = True

    return output


def find_block_index(blocks: list[str], expected: str) -> int:
    expected_folded = fold_for_match(expected)
    for index, block in enumerate(blocks):
        if fold_for_match(block).startswith(expected_folded):
            return index
    raise ValueError(f"Could not find block starting with {expected!r}")


def slice_for_source(source_id: str, blocks: list[str]) -> list[str]:
    if source_id == "clt_1943_processo":
        start = find_block_index(blocks, "TÍTULO X")
        end = find_block_index(blocks, "TÍTULO XI")
        return blocks[start:end]

    if source_id == "clt_1943":
        end = find_block_index(blocks, "TÍTULO X")
        return blocks[:end]

    return blocks


def clean_text_for_classification(source_id: str, raw_text: str) -> str:
    blocks = split_blocks(raw_text)
    blocks = slice_for_source(source_id, blocks)
    blocks = remove_noise_blocks(blocks)
    if not blocks:
        raise ValueError(f"No cleaned blocks produced for {source_id}")
    return join_blocks(blocks)


def clean_manifest(args: argparse.Namespace) -> None:
    written = 0
    skipped_existing = 0
    skipped_missing_raw = 0

    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        source_id = row.get("source_id", "").strip()
        text_raw_rel = row.get("local_path_text_raw", "").strip()
        text_rel = row.get("local_path_text", "").strip()
        if not source_id or not text_rel:
            raise ValueError(f"Manifest row is missing source_id or local_path_text: {row}")
        if not text_raw_rel:
            text_raw_rel = f"text_raw/{source_id}.txt"

        input_path = SOURCES_DIR / text_raw_rel
        output_path = SOURCES_DIR / text_rel

        if not input_path.exists():
            skipped_missing_raw += 1
            print(f"Skipped missing raw text: {source_id}: {input_path}")
            continue
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            print(f"Skipped existing cleaned text: {source_id}: {output_path}")
            continue

        raw_text = input_path.read_text(encoding="utf-8")
        cleaned_text = clean_text_for_classification(source_id, raw_text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned_text, encoding="utf-8")
        stats = summarize_output(cleaned_text)
        written += 1
        print(
            f"Wrote {output_path} "
            f"(articles={stats['article_lines']}, paragraphs={stats['paragraph_lines']})"
        )

    print(
        "Done. "
        f"wrote={written}, skipped_existing={skipped_existing}, "
        f"skipped_missing_raw={skipped_missing_raw}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw Planalto text files for classification-quality corpus use."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lcp95-html", type=Path, default=DEFAULT_LCP95_HTML)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_lcp95_reference(args.lcp95_html)
    clean_manifest(args)


if __name__ == "__main__":
    main()
