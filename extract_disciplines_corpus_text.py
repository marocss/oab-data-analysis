from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES_DIR = ROOT / "sources"
DEFAULT_MANIFEST = SOURCES_DIR / "discipline_corpus_manifest.csv"
DEFAULT_LCP95_HTML = ROOT / "sources" / "raw" / "Lcp95.html"

SKIP_TAGS = {"head", "script", "style", "noscript"}
TEXT_BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6"}
ORDINAL_SUP_TAG = "__SUP_ORDINAL__"
HIERARCHY_MARKER_RE = re.compile(
    r"^(?P<marker>"
    r"P\s*A\s*R\s*T\s*E\s+G\s*E\s*R\s*A\s*L|"
    r"PARTE\s+(?:GERAL|ESPECIAL|[A-ZÁÉÍÓÚÂÊÔÃÕÇ]+)|"
    r"LIVRO\s+(?:[IVXLCDM]+|ÚNICO|UNICO)|"
    r"T[ÍI]TULO\s+(?:[IVXLCDM]+|ÚNICO|UNICO)|"
    r"CAP[ÍI]TULO\s+(?:[IVXLCDM]+|ÚNICO|UNICO)|"
    r"SUBSE[ÇC][ÃA]O\s+(?:[IVXLCDM]+|ÚNICA|UNICA)|"
    r"SE[ÇC][ÃA]O\s+(?:[IVXLCDM]+|ÚNICA|UNICA)"
    r")\b(?P<title>.*)$",
    re.IGNORECASE,
)
REFERENCE_TERMS = (
    "Art. 10",
    "artigo",
    "parágrafos",
    "incisos",
    "alíneas",
    "Subseções",
    "Seção",
    "Capítulo",
    "Título",
)


@dataclass
class Paragraph:
    text: str
    align: str


class ParagraphExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.sup_depth = 0
        self.current_attrs: dict[str, str] | None = None
        self.current_parts: list[str] = []
        self.paragraphs: list[Paragraph] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in TEXT_BLOCK_TAGS:
            self.flush_current()
            self.current_attrs = {key.lower(): value or "" for key, value in attrs}
            self.current_parts = []
        elif tag == "sup" and self.current_attrs is not None:
            self.sup_depth += 1
        elif tag == "br" and self.current_attrs is not None:
            self.current_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in TEXT_BLOCK_TAGS:
            self.flush_current()
        elif tag == "sup" and self.sup_depth:
            self.sup_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth or self.current_attrs is None:
            return
        if self.sup_depth and data.strip() in {"o", "º", "°"}:
            self.current_parts.append(ORDINAL_SUP_TAG)
            return
        self.current_parts.append(data)

    def close(self) -> None:
        super().close()
        self.flush_current()

    def flush_current(self) -> None:
        if self.current_attrs is None:
            return
        text = clean_block_text("".join(self.current_parts))
        if text:
            self.paragraphs.append(
                Paragraph(
                    text=text,
                    align=self.current_attrs.get("align", "").lower(),
                )
            )
        self.current_attrs = None
        self.current_parts = []


def decode_html(path: Path) -> str:
    data = path.read_bytes()
    charset_match = re.search(br"charset=[\"']?([\w.-]+)", data[:4096], re.IGNORECASE)
    encodings: list[str] = []
    if charset_match:
        encodings.append(charset_match.group(1).decode("ascii", errors="ignore"))
    encodings.extend(["utf-8-sig", "cp1252", "latin-1"])

    tried: set[str] = set()
    for encoding in encodings:
        normalized = encoding.lower()
        if normalized in tried:
            continue
        tried.add(normalized)
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("latin-1", errors="replace")


def clean_block_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v\n]+", " ", text)
    text = re.sub(rf"\s*{ORDINAL_SUP_TAG}\s*", "º ", text)
    text = re.sub(r"[ \t\f\v\n]+", " ", text)
    return text.strip()


def fold_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return clean_block_text(text).upper()


def extract_paragraphs(path: Path) -> list[Paragraph]:
    parser = ParagraphExtractor()
    parser.feed(decode_html(path))
    parser.close()
    return parser.paragraphs


def validate_lcp95_reference(path: Path) -> None:
    text = "\n".join(paragraph.text for paragraph in extract_paragraphs(path))
    missing = [term for term in REFERENCE_TERMS if term not in text]
    if missing:
        missing_terms = ", ".join(missing)
        raise ValueError(f"{path} is missing expected LCP 95 reference terms: {missing_terms}")


def block_matches_prefix(block_text: str, expected: str) -> bool:
    return fold_for_match(block_text).startswith(fold_for_match(expected))


def canonical_text_blocks(text: str) -> list[str]:
    match = HIERARCHY_MARKER_RE.fullmatch(text)
    if not match:
        return [text]
    marker = clean_block_text(match.group("marker"))
    title = clean_block_text(match.group("title"))
    if title:
        return [marker, title]
    return [marker]


def extract_text_blocks(
    input_html: Path,
    *,
    start_at: str | None,
    stop_before: str | None,
) -> str:
    paragraphs = extract_paragraphs(input_html)
    output: list[str] = []
    started = start_at is None

    for paragraph in paragraphs:
        if not started:
            if start_at and block_matches_prefix(paragraph.text, start_at):
                started = True
            else:
                continue

        if stop_before and block_matches_prefix(paragraph.text, stop_before):
            break

        output.extend(canonical_text_blocks(paragraph.text))

    if not output:
        start_description = f" starting at {start_at!r}" if start_at else ""
        raise ValueError(f"No text blocks extracted from {input_html}{start_description}")
    return "\n\n".join(output).strip() + "\n"


def count_pattern(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def summarize_output(text: str) -> dict[str, int]:
    return {
        "parts": count_pattern(r"^P\s*A\s*R\s*T\s*E\b", text),
        "books": count_pattern(r"^LIVRO\s+", text),
        "titles": count_pattern(r"^TÍTULO\s+", text),
        "chapters": count_pattern(r"^CAPÍTULO\s+", text),
        "sections": count_pattern(r"^Seção\s+", text),
        "subsections": count_pattern(r"^Subseção\s+", text),
        "article_lines": count_pattern(r"^Art\.\s+", text),
        "paragraph_lines": count_pattern(r"^(?:§|Parágrafo único\.)", text),
        "inciso_lines": count_pattern(r"^[IVXLCDM]+\s+[-–]", text),
        "alinea_lines": count_pattern(r"^[a-z]\)\s+", text),
    }


def write_text_output(output: Path, text: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def extract_single_file(args: argparse.Namespace) -> None:
    if args.input_html is None or args.output is None:
        raise SystemExit("--input-html and --output are required unless --manifest is used")

    text = extract_text_blocks(
        args.input_html,
        start_at=args.start_at,
        stop_before=args.stop_before,
    )
    stats = summarize_output(text)
    write_text_output(args.output, text)

    print(f"Wrote {args.output}")
    for key, value in stats.items():
        print(f"{key}: {value}")


def extract_manifest(args: argparse.Namespace) -> None:
    manifest_path = args.manifest
    assert manifest_path is not None

    written = 0
    skipped_existing = 0
    skipped_non_html = 0
    skipped_missing_raw = 0

    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        source_id = row.get("source_id", "").strip()
        raw_rel = row.get("local_path_raw", "").strip()
        text_raw_rel = row.get("local_path_text_raw", "").strip()

        if not raw_rel.lower().endswith((".html", ".htm")):
            skipped_non_html += 1
            print(f"Skipped non-HTML source: {source_id or raw_rel}")
            continue
        if not text_raw_rel:
            raise ValueError(f"{source_id or raw_rel} is missing local_path_text_raw")

        input_html = SOURCES_DIR / raw_rel
        output = SOURCES_DIR / text_raw_rel

        if not input_html.exists():
            skipped_missing_raw += 1
            print(f"Skipped missing raw HTML: {source_id}: {input_html}")
            continue
        if output.exists() and not args.overwrite:
            skipped_existing += 1
            print(f"Skipped existing raw text: {source_id}: {output}")
            continue

        text = extract_text_blocks(input_html, start_at=None, stop_before=None)
        write_text_output(output, text)
        stats = summarize_output(text)
        written += 1
        print(
            f"Wrote {output} "
            f"(articles={stats['article_lines']}, paragraphs={stats['paragraph_lines']})"
        )

    print(
        "Done. "
        f"wrote={written}, skipped_existing={skipped_existing}, "
        f"skipped_non_html={skipped_non_html}, skipped_missing_raw={skipped_missing_raw}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract raw plain text blocks from Planalto HTML sources."
    )
    parser.add_argument("--input-html", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Extract all HTML rows from a corpus manifest into local_path_text_raw.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-at", help="Optional block-text prefix where extraction should begin.")
    parser.add_argument("--stop-before", help="Optional block-text prefix where extraction should stop.")
    parser.add_argument("--lcp95-html", type=Path, default=DEFAULT_LCP95_HTML)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_lcp95_reference(args.lcp95_html)
    if args.manifest is not None:
        extract_manifest(args)
    else:
        extract_single_file(args)


if __name__ == "__main__":
    main()
