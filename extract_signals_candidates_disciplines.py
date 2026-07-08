from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from extract_disciplines_corpus_text import (
    DEFAULT_LCP95_HTML,
    HIERARCHY_MARKER_RE,
    validate_lcp95_reference,
)


ROOT = Path(__file__).resolve().parent
SOURCES_DIR = ROOT / "sources"
DEFAULT_MANIFEST = SOURCES_DIR / "manifests" / "source_documents.csv"
DEFAULT_TEXT_DIR = SOURCES_DIR / "extractions" / "clean_text"
DEFAULT_STOPWORDS = ROOT / "data" / "config" / "legal_stopwords.txt"
DEFAULT_OUTPUT = ROOT / "data" / "signals" / "signal_candidates_disciplines.csv"

FIELDNAMES = [
    "signal_id",
    "signal_type",
    "candidate",
    "candidate_normalized",
    "source_id",
    "source_title",
    "source_type",
    "discipline_id",
    "candidate_disciplines",
    "provenance_kind",
    "provenance_ref",
    "hierarchy_path",
    "article_number",
    "frequency",
    "source_count",
    "discipline_count",
    "specificity_score",
    "is_ambiguous",
    "quality_notes",
]

EDGE_PUNCTUATION = " \t\r\n\"'“”‘’.,;:!?[]{}<>-–—_/\\|"
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9º°./-]*")
ARTICLE_RE = re.compile(
    r"^Art\.?\s+(?P<number>\d+(?:\.\d+)?(?:[º°])?(?:-[A-Z])?)\.?\s*[-–]?\s*(?P<body>.*)$",
    re.IGNORECASE,
)
PARAGRAPH_RE = re.compile(r"^(?:§\s*\d+[º°]?(?:-[A-Z])?\.?|Parágrafo\s+único\.?)\s*", re.IGNORECASE)
LIST_MARKER_RE = re.compile(
    r"^(?P<marker>(?:[IVXLCDM]+|[a-z]|\d+))\s*(?:[)\-.]|[-–])\s+(?P<body>.+)$",
    re.IGNORECASE,
)
LEADING_HEADING_WORDS_RE = re.compile(r"^(?:d[oa]s?|de|aos?|as|os|o|a)\s+", re.IGNORECASE)
LEGAL_MARKER_RE = re.compile(
    r"^(?:parte|livro|titulo|subtitulo|capitulo|secao|subsecao|artigo|art)\s+"
    r"(?:[ivxlcdm]+(?:-[a-z])?|unico|[0-9]+[a-zºo]*(?:-[a-z])?)$",
    re.IGNORECASE,
)
LOOSE_HEADING_MARKER_RE = re.compile(
    r"^(?P<marker>(?:subt[íi]tulo|t[íi]tulo|cap[íi]tulo|subse[çc][ãa]o|se[çc][ãa]o)"
    r"\s+(?:[ivxlcdm]+(?:-[a-z])?|únic[ao]|unic[ao]))\s+(?P<title>.+)$",
    re.IGNORECASE,
)
NOISE_FRAGMENT_RE = re.compile(
    r"\b(?:redacao dada|incluido pela|incluida pela|alterado pela|revogado pela|"
    r"renumerado|vigencia|producao de efeitos|vide|vetado|revogado|nr)\b",
    re.IGNORECASE,
)
AMENDMENT_PAREN_RE = re.compile(
    r"\s*\([^)]*(?:Redação|Inclu[íi]d[oa]|Revogad[oa]|Vigência|Vide|Regulamento|"
    r"Produção de efeitos|Renumerad[oa]|NR|ADIN|ADI)[^)]*\)",
    re.IGNORECASE,
)
TRAILING_FOOTNOTE_RE = re.compile(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])\d+\b")

PHRASE_BOUNDARY_WORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
}
BOILERPLATE_NORMALIZED = {
    "disposicoes gerais",
    "disposicoes preliminares",
    "disposicao preliminar",
    "disposicoes finais",
    "disposicoes transitorias",
    "disposicoes finais e transitorias",
    "das disposicoes finais",
    "vigencia",
    "revogado",
    "revogada",
    "revogados",
    "revogadas",
    "vetado",
    "vetada",
    "vetados",
    "vetadas",
}
GENERIC_ONE_WORDS = {
    "acao",
    "ato",
    "caso",
    "competencia",
    "contrato",
    "direito",
    "forma",
    "lei",
    "norma",
    "pessoa",
    "prazo",
    "processo",
    "recurso",
    "responsabilidade",
}
BAD_TERM_STARTS = PHRASE_BOUNDARY_WORDS | {
    "quando",
    "se",
    "tambem",
    "toda",
    "todas",
    "todo",
    "todos",
    "qualquer",
}
BAD_TERM_ENDS = PHRASE_BOUNDARY_WORDS | {"que", "cuja", "cujo", "lhe", "nao"}

MAX_TOKENS_BY_TYPE = {
    "source_title": 14,
    "source_alias": 10,
    "heading_title": 12,
    "heading_topic_phrase": 5,
    "article_rubric": 8,
    "defined_term": 10,
    "enumerated_term": 10,
}
SIGNAL_TYPE_WEIGHT = {
    "defined_term": 1.0,
    "enumerated_term": 0.95,
    "article_rubric": 0.9,
    "heading_title": 0.82,
    "source_alias": 0.78,
    "heading_topic_phrase": 0.68,
    "source_title": 0.6,
}
HIERARCHY_LEVELS = ["parte", "livro", "titulo", "capitulo", "secao", "subsecao"]


@dataclass(frozen=True)
class SourceMeta:
    source_id: str
    title: str
    source_type: str
    discipline_id: str
    local_path_text: str


@dataclass
class Candidate:
    signal_type: str
    candidate: str
    candidate_normalized: str
    source_id: str
    source_title: str
    source_type: str
    discipline_id: str
    provenance_kind: str
    provenance_ref: str
    hierarchy_path: str
    article_number: str
    frequency: int = 1
    quality_notes: set[str] = field(default_factory=set)

    def key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.signal_type,
            self.candidate_normalized,
            self.source_id,
            self.provenance_kind,
            self.provenance_ref,
            self.article_number,
        )


def fold_ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", errors="ignore").decode("ascii")


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def standardize_number_marker(text: str) -> str:
    text = re.sub(r"\bn\s*\.?\s*[º°]\b", "numero", text, flags=re.IGNORECASE)
    text = re.sub(r"\bn\s*\.\s*o\b", "numero", text, flags=re.IGNORECASE)
    text = re.sub(r"\bn[uú]mero\b", "numero", text, flags=re.IGNORECASE)
    return text


def strip_amendment_noise(text: str) -> str:
    previous = None
    text = collapse_spaces(text)
    while previous != text:
        previous = text
        text = AMENDMENT_PAREN_RE.sub(" ", text)
    text = re.sub(
        r"\b(?:Redação dada|Incluído|Incluída|Revogado|Revogada|Vigência|Vide|Regulamento)\b.*$",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+\b(?:Vigência|Produção de efeitos)\b\s*$", " ", text, flags=re.IGNORECASE)
    text = TRAILING_FOOTNOTE_RE.sub("", text)
    return collapse_spaces(text)


def normalize_candidate(text: str) -> str:
    text = strip_amendment_noise(standardize_number_marker(text)).lower()
    text = fold_ascii(text)
    text = collapse_spaces(text)
    text = text.strip(EDGE_PUNCTUATION)
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    return collapse_spaces(text)


def display_candidate(text: str, *, strip_heading_prefix: bool = False) -> str:
    text = strip_amendment_noise(standardize_number_marker(text))
    text = text.strip(EDGE_PUNCTUATION)
    if strip_heading_prefix:
        previous = None
        while previous != text:
            previous = text
            text = LEADING_HEADING_WORDS_RE.sub("", text).strip(EDGE_PUNCTUATION)
    return collapse_spaces(text)


def has_balanced_parentheses(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def token_count(text: str) -> int:
    return len(normalize_candidate(text).split())


def read_stopwords(path: Path) -> set[str]:
    if not path.exists():
        return set()
    stopwords = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            stopwords.add(normalize_candidate(line))
    return stopwords


def read_manifest(path: Path) -> list[SourceMeta]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = csv.DictReader(csv_file)
        sources = []
        for row in rows:
            source_id = row.get("source_id", "").strip()
            include_in_signal_corpus = row.get("include_in_signal_corpus", "").strip().lower()
            if include_in_signal_corpus not in {"true", "1", "yes", "y"}:
                continue
            local_path_text = row.get("clean_text_path", "").strip()
            if not source_id:
                continue
            sources.append(
                SourceMeta(
                    source_id=source_id,
                    title=row.get("title", "").strip(),
                    source_type=row.get("source_type", "").strip(),
                    discipline_id=row.get("discipline_id", "").strip(),
                    local_path_text=local_path_text,
                )
            )
        return sources


def split_blocks(text: str) -> list[str]:
    blocks = [collapse_spaces(block) for block in re.split(r"\n\s*\n+", text)]
    return [block for block in blocks if block]


def hierarchy_kind(marker: str) -> str:
    folded = fold_ascii(marker).lower()
    if folded.startswith("p a r t e") or folded.startswith("parte"):
        return "parte"
    if folded.startswith("livro"):
        return "livro"
    if folded.startswith("titulo"):
        return "titulo"
    if folded.startswith("capitulo"):
        return "capitulo"
    if folded.startswith("subsecao"):
        return "subsecao"
    if folded.startswith("secao"):
        return "secao"
    return "heading"


def is_hierarchy_block(block: str) -> bool:
    return HIERARCHY_MARKER_RE.fullmatch(display_candidate(block)) is not None


def hierarchy_match(block: str) -> re.Match[str] | None:
    return HIERARCHY_MARKER_RE.fullmatch(display_candidate(block))


def article_match(block: str) -> re.Match[str] | None:
    return ARTICLE_RE.match(block)


def is_article_block(block: str) -> bool:
    return article_match(block) is not None


def is_paragraph_block(block: str) -> bool:
    return PARAGRAPH_RE.match(block) is not None


def list_match(block: str) -> re.Match[str] | None:
    return LIST_MARKER_RE.match(block)


def is_list_item(block: str) -> bool:
    return list_match(block) is not None


def strip_list_marker(block: str) -> str:
    match = list_match(block)
    if not match:
        return block
    return collapse_spaces(match.group("body"))


def strip_paragraph_marker(block: str) -> str:
    return collapse_spaces(PARAGRAPH_RE.sub("", block))


def is_all_caps_like(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    uppercase = sum(1 for char in letters if char.upper() == char)
    return uppercase / len(letters) >= 0.75


def starts_like_preamble(block: str) -> bool:
    folded = fold_ascii(block).lower()
    return folded.startswith(
        (
            "o presidente da republica",
            "a presidenta da republica",
            "o presidente do conselho",
            "faco saber",
            "decreta",
            "promulga",
            "nos, representantes",
        )
    )


def is_plain_heading(block: str) -> bool:
    cleaned = strip_amendment_noise(block)
    normalized = normalize_candidate(cleaned)
    tokens = normalized.split()
    if not tokens or len(tokens) > 14:
        return False
    if starts_like_preamble(cleaned):
        return False
    if is_article_block(cleaned) or is_paragraph_block(cleaned) or is_list_item(cleaned):
        return False
    if LEGAL_MARKER_RE.fullmatch(normalized):
        return False
    if cleaned.endswith((".", ";", ":")) and not is_all_caps_like(cleaned):
        return False
    if re.search(r"\b(?:dispõe|institui|estabelece|altera|regula|aprova)\b", normalized):
        return False
    return bool(len(tokens) <= 6 or is_all_caps_like(cleaned) or re.match(r"^d[ao]s?\b", normalized))


def is_article_rubric(block: str) -> bool:
    cleaned = display_candidate(block)
    tokens = normalize_candidate(cleaned).split()
    if not 1 <= len(tokens) <= 6:
        return False
    if LOOSE_HEADING_MARKER_RE.match(cleaned):
        return False
    if normalize_candidate(cleaned).startswith(
        ("parte ", "livro ", "titulo ", "subtitulo ", "capitulo ", "secao ", "subsecao ")
    ):
        return False
    if is_all_caps_like(cleaned):
        return False
    if normalize_candidate(cleaned) in BOILERPLATE_NORMALIZED:
        return False
    return is_plain_heading(cleaned)


def loose_heading_title(block: str) -> str | None:
    match = LOOSE_HEADING_MARKER_RE.match(block)
    if not match:
        return None
    return display_candidate(match.group("title"))


def update_hierarchy_path(
    hierarchy: dict[str, str],
    kind: str,
    title: str,
) -> dict[str, str]:
    next_hierarchy = dict(hierarchy)
    if kind in HIERARCHY_LEVELS:
        level_index = HIERARCHY_LEVELS.index(kind)
        for lower_kind in HIERARCHY_LEVELS[level_index:]:
            next_hierarchy.pop(lower_kind, None)
    next_hierarchy[kind] = title
    return next_hierarchy


def format_hierarchy_path(hierarchy: dict[str, str]) -> str:
    return " > ".join(hierarchy[kind] for kind in HIERARCHY_LEVELS if hierarchy.get(kind))


def candidate_allowed(
    signal_type: str,
    candidate: str,
    stopwords: set[str],
    *,
    strong_definition_context: bool = False,
) -> tuple[bool, set[str]]:
    normalized = normalize_candidate(candidate)
    notes: set[str] = set()
    if not normalized:
        return False, notes
    if not has_balanced_parentheses(candidate):
        return False, notes | {"unbalanced_parentheses"}
    if normalized in stopwords or normalized in BOILERPLATE_NORMALIZED:
        return False, notes
    if LEGAL_MARKER_RE.fullmatch(normalized) or NOISE_FRAGMENT_RE.search(normalized):
        return False, notes
    tokens = normalized.split()
    if not tokens:
        return False, notes
    max_tokens = MAX_TOKENS_BY_TYPE.get(signal_type, 10)
    if len(tokens) > max_tokens:
        return False, notes | {"too_long"}
    if all(token in PHRASE_BOUNDARY_WORDS for token in tokens):
        return False, notes
    if tokens[0] in BAD_TERM_STARTS or tokens[-1] in BAD_TERM_ENDS:
        return False, notes
    if len(tokens) == 1:
        if tokens[0] in GENERIC_ONE_WORDS or len(tokens[0]) < 4:
            return False, notes
        if strong_definition_context:
            notes.add("one_word_definition")
        else:
            notes.add("one_word_signal")
    if len(tokens) == 1 and tokens[0] in GENERIC_ONE_WORDS:
        return False, notes
    if tokens[0][:1].isdigit():
        return False, notes
    return True, notes


def make_candidate(
    signal_type: str,
    raw_text: str,
    source: SourceMeta,
    *,
    provenance_kind: str,
    provenance_ref: str,
    hierarchy_path: str,
    article_number: str,
    stopwords: set[str],
    strip_heading_prefix: bool = False,
    strong_definition_context: bool = False,
) -> Candidate | None:
    candidate = display_candidate(raw_text, strip_heading_prefix=strip_heading_prefix)
    allowed, notes = candidate_allowed(
        signal_type,
        candidate,
        stopwords,
        strong_definition_context=strong_definition_context,
    )
    if not allowed:
        return None
    return Candidate(
        signal_type=signal_type,
        candidate=candidate,
        candidate_normalized=normalize_candidate(candidate),
        source_id=source.source_id,
        source_title=source.title,
        source_type=source.source_type,
        discipline_id=source.discipline_id,
        provenance_kind=provenance_kind,
        provenance_ref=provenance_ref,
        hierarchy_path=hierarchy_path,
        article_number=article_number,
        quality_notes=notes,
    )


def source_aliases(source: SourceMeta, first_blocks: list[str]) -> Iterable[str]:
    yield source.title
    for block in first_blocks[:4]:
        cleaned = strip_amendment_noise(block)
        if is_article_block(cleaned) or is_hierarchy_block(cleaned) or starts_like_preamble(cleaned):
            continue
        if re.search(r"\b(?:código|estatuto|constituição|convenção|pacto|lei geral|marco civil)\b", cleaned, re.I):
            yield cleaned
        short_name_match = re.search(r"\((?P<alias>[A-Z]{2,8})\)", cleaned)
        if short_name_match:
            yield short_name_match.group("alias")


def clean_defined_term(term: str) -> str:
    term = display_candidate(term)
    term = re.sub(r"^(?:o|a|os|as|um|uma|uns|umas)\s+", "", term, flags=re.IGNORECASE)
    return display_candidate(term)


def defined_terms_from_text(text: str) -> Iterable[str]:
    cleaned = strip_amendment_noise(text)
    patterns = [
        r"\bConsidera(?:m-se|-se)?\s+([^,.;:]{3,100}?)(?:,|\s+para\b|\s+como\b|\s+aquele\b|\s+aquela\b|\s+todo\b|\s+toda\b|\s+o\b|\s+a\b|\s+os\b|\s+as\b)",
        r"\bEntende(?:m-se|-se)?\s+por\s+([^,.;:]{3,100}?)(?:,|:|\s+o\b|\s+a\b|\s+os\b|\s+as\b)",
        r"\bEntende(?:m-se|-se)?\s+como\s+([^,.;:]{3,100}?)(?:,|:|\s+o\b|\s+a\b|\s+os\b|\s+as\b)",
        r"\bDefine(?:m-se|-se)?\s+([^,.;:]{3,100}?)(?:,|:|\s+como\b)",
        r"\bDenomina(?:m-se|-se)?\s+([^,.;:]{3,100}?)(?:,|:|\s+como\b)",
        r"^([^,.;:]{3,90}?)\s+(?:é|são)\s+(?:a|o|as|os|toda|todo|qualquer|atividade|bem|bens|conjunto|direito|medida|norma|pessoa|prestação|serviço)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            term = clean_defined_term(match.group(1))
            if term:
                yield term


def enumerated_term_from_text(text: str) -> str | None:
    body = strip_list_marker(text)
    body = strip_amendment_noise(body)
    match = re.match(r"([^,;:]{3,90}?)(?:,|:)\s+\S", body)
    if not match:
        return None
    return clean_defined_term(match.group(1))


def heading_tokens(title: str) -> list[str]:
    cleaned = display_candidate(title, strip_heading_prefix=True)
    tokens = [normalize_candidate(match.group(0)) for match in TOKEN_RE.finditer(cleaned)]
    return [token for token in tokens if token and token not in PHRASE_BOUNDARY_WORDS]


def heading_topic_phrases(
    heading_occurrences: list[tuple[str, SourceMeta, str, str]],
    stopwords: set[str],
    min_frequency: int,
) -> list[Candidate]:
    occurrences: dict[str, list[tuple[str, SourceMeta, str, str]]] = defaultdict(list)
    for raw_title, source, provenance_ref, hierarchy_path in heading_occurrences:
        tokens = heading_tokens(raw_title)
        for size in range(2, 6):
            if len(tokens) < size:
                continue
            for index in range(0, len(tokens) - size + 1):
                phrase = " ".join(tokens[index : index + size])
                occurrences[phrase].append((raw_title, source, provenance_ref, hierarchy_path))

    candidates: list[Candidate] = []
    for phrase, phrase_occurrences in sorted(occurrences.items()):
        source_count = len({source.source_id for _raw, source, _ref, _path in phrase_occurrences})
        if len(phrase_occurrences) < min_frequency and source_count < min_frequency:
            continue

        by_provenance: dict[tuple[SourceMeta, str, str], int] = Counter(
            (source, provenance_ref, hierarchy_path)
            for _raw, source, provenance_ref, hierarchy_path in phrase_occurrences
        )
        for (source, provenance_ref, hierarchy_path), frequency in sorted(
            by_provenance.items(),
            key=lambda item: (item[0][0].source_id, item[0][1], item[0][2]),
        ):
            candidate = make_candidate(
                "heading_topic_phrase",
                phrase,
                source,
                provenance_kind="heading",
                provenance_ref=provenance_ref,
                hierarchy_path=hierarchy_path,
                article_number="",
                stopwords=stopwords,
            )
            if candidate is not None:
                candidate.frequency = frequency
                candidate.quality_notes.add(f"phrase_occurrences={frequency}")
                candidate.quality_notes.add("phrase_sources=1")
                candidate.candidate = phrase
                candidate.candidate_normalized = normalize_candidate(phrase)
                candidates.append(candidate)
    return candidates


def extract_source_candidates(
    source: SourceMeta,
    text_path: Path,
    stopwords: set[str],
) -> tuple[list[Candidate], list[tuple[str, SourceMeta, str, str]]]:
    blocks = split_blocks(text_path.read_text(encoding="utf-8"))
    candidates: list[Candidate] = []
    heading_occurrences: list[tuple[str, SourceMeta, str, str]] = []
    hierarchy: dict[str, str] = {}
    current_article = ""
    consumed_indexes: set[int] = set()

    for alias in source_aliases(source, blocks):
        signal_type = "source_title" if normalize_candidate(alias) == normalize_candidate(source.title) else "source_alias"
        candidate = make_candidate(
            signal_type,
            alias,
            source,
            provenance_kind="source",
            provenance_ref=source.source_id,
            hierarchy_path="",
            article_number="",
            stopwords=stopwords,
            strong_definition_context=signal_type == "source_alias",
        )
        if candidate is not None:
            candidates.append(candidate)

    for index, block in enumerate(blocks):
        if index in consumed_indexes:
            continue

        marker_match = hierarchy_match(block)
        if marker_match:
            marker = display_candidate(marker_match.group("marker"))
            inline_title = display_candidate(marker_match.group("title"))
            title = inline_title
            kind = hierarchy_kind(marker)
            folded_marker = normalize_candidate(marker)
            marker_is_self_describing_part = kind == "parte" and any(
                word in folded_marker for word in ("geral", "especial")
            )
            if (
                not title
                and not marker_is_self_describing_part
                and index + 1 < len(blocks)
                and is_plain_heading(blocks[index + 1])
            ):
                title = display_candidate(blocks[index + 1])
                consumed_indexes.add(index + 1)
            display_title = title or marker
            hierarchy = update_hierarchy_path(hierarchy, kind, display_title)
            hierarchy_path = format_hierarchy_path(hierarchy)
            if title:
                candidate = make_candidate(
                    "heading_title",
                    title,
                    source,
                    provenance_kind="heading",
                    provenance_ref=marker,
                    hierarchy_path=hierarchy_path,
                    article_number="",
                    stopwords=stopwords,
                    strip_heading_prefix=True,
                )
                if candidate is not None:
                    candidates.append(candidate)
                    heading_occurrences.append((candidate.candidate, source, marker, hierarchy_path))
            continue

        if is_article_block(block):
            match = article_match(block)
            assert match is not None
            current_article = match.group("number").rstrip(".")
            body = match.group("body")
            hierarchy_path = format_hierarchy_path(hierarchy)
            for term in defined_terms_from_text(body):
                candidate = make_candidate(
                    "defined_term",
                    term,
                    source,
                    provenance_kind="article",
                    provenance_ref=f"Art. {current_article}",
                    hierarchy_path=hierarchy_path,
                    article_number=current_article,
                    stopwords=stopwords,
                    strong_definition_context=True,
                )
                if candidate is not None:
                    candidates.append(candidate)
            continue

        if is_plain_heading(block):
            next_block = blocks[index + 1] if index + 1 < len(blocks) else ""
            hierarchy_path = format_hierarchy_path(hierarchy)
            loose_title = loose_heading_title(block)
            if loose_title:
                hierarchy = update_hierarchy_path(hierarchy, "secao", loose_title)
                hierarchy_path = format_hierarchy_path(hierarchy)
                candidate = make_candidate(
                    "heading_title",
                    loose_title,
                    source,
                    provenance_kind="heading",
                    provenance_ref=f"line {index + 1}",
                    hierarchy_path=hierarchy_path,
                    article_number="",
                    stopwords=stopwords,
                    strip_heading_prefix=True,
                )
                if candidate is not None:
                    candidates.append(candidate)
                    heading_occurrences.append((candidate.candidate, source, f"line {index + 1}", hierarchy_path))
                continue

            if next_block and is_article_block(next_block) and is_article_rubric(block):
                next_article = article_match(next_block)
                article_number = next_article.group("number").rstrip(".") if next_article else ""
                candidate = make_candidate(
                    "article_rubric",
                    block,
                    source,
                    provenance_kind="article_rubric",
                    provenance_ref=f"Art. {article_number}",
                    hierarchy_path=hierarchy_path,
                    article_number=article_number,
                    stopwords=stopwords,
                    strip_heading_prefix=False,
                )
                if candidate is not None:
                    candidates.append(candidate)
                    heading_occurrences.append((candidate.candidate, source, f"Art. {article_number}", hierarchy_path))
                continue

            if is_all_caps_like(block) or re.match(r"^d[ao]s?\b", normalize_candidate(block)):
                heading_title = display_candidate(block)
                hierarchy = update_hierarchy_path(hierarchy, "secao", heading_title)
                hierarchy_path = format_hierarchy_path(hierarchy)
                candidate = make_candidate(
                    "heading_title",
                    heading_title,
                    source,
                    provenance_kind="heading",
                    provenance_ref=f"line {index + 1}",
                    hierarchy_path=hierarchy_path,
                    article_number="",
                    stopwords=stopwords,
                    strip_heading_prefix=True,
                )
                if candidate is not None:
                    candidates.append(candidate)
                    heading_occurrences.append((candidate.candidate, source, f"line {index + 1}", hierarchy_path))
                continue

        if current_article and (is_paragraph_block(block) or is_list_item(block)):
            body = strip_paragraph_marker(block) if is_paragraph_block(block) else strip_list_marker(block)
            hierarchy_path = format_hierarchy_path(hierarchy)
            for term in defined_terms_from_text(body):
                candidate = make_candidate(
                    "defined_term",
                    term,
                    source,
                    provenance_kind="article",
                    provenance_ref=f"Art. {current_article}",
                    hierarchy_path=hierarchy_path,
                    article_number=current_article,
                    stopwords=stopwords,
                    strong_definition_context=True,
                )
                if candidate is not None:
                    candidates.append(candidate)

            if is_list_item(block):
                term = enumerated_term_from_text(block)
                if term:
                    candidate = make_candidate(
                        "enumerated_term",
                        term,
                        source,
                        provenance_kind="article",
                        provenance_ref=f"Art. {current_article}",
                        hierarchy_path=hierarchy_path,
                        article_number=current_article,
                        stopwords=stopwords,
                        strong_definition_context=True,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

    return candidates, heading_occurrences


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    grouped: dict[tuple[str, str, str, str, str, str], Candidate] = {}
    frequencies: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for candidate in candidates:
        key = candidate.key()
        frequencies[key] += candidate.frequency
        if key in grouped:
            grouped[key].quality_notes.update(candidate.quality_notes)
            continue
        grouped[key] = candidate

    deduped = []
    for key, candidate in grouped.items():
        candidate.frequency = frequencies[key]
        deduped.append(candidate)
    return sorted(
        deduped,
        key=lambda row: (
            row.source_id,
            row.signal_type,
            row.candidate_normalized,
            row.provenance_kind,
            row.provenance_ref,
        ),
    )


def signal_id_for(candidate: Candidate) -> str:
    payload = "|".join(
        [
            candidate.signal_type,
            candidate.candidate_normalized,
            candidate.source_id,
            candidate.provenance_kind,
            candidate.provenance_ref,
            candidate.article_number,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def specificity_score(candidate: Candidate, discipline_count: int) -> str:
    tokens = candidate.candidate_normalized.split()
    token_factor = min(1.0, max(0.35, len(tokens) / 6))
    ambiguity_penalty = 1 / (1 + 0.35 * max(0, discipline_count - 1))
    score = SIGNAL_TYPE_WEIGHT.get(candidate.signal_type, 0.5) * token_factor * ambiguity_penalty
    return f"{score:.3f}"


def candidate_rows(candidates: list[Candidate]) -> list[dict[str, str | int]]:
    by_form: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_form[candidate.candidate_normalized].append(candidate)

    rows: list[dict[str, str | int]] = []
    for candidate in candidates:
        form_candidates = by_form[candidate.candidate_normalized]
        source_ids = sorted({item.source_id for item in form_candidates})
        discipline_ids = sorted({item.discipline_id for item in form_candidates if item.discipline_id})
        source_count = len(source_ids)
        discipline_count = len(discipline_ids)
        is_ambiguous = discipline_count > 1 or (
            source_count > 3 and candidate.signal_type in {"heading_topic_phrase", "heading_title"}
        )
        notes = sorted(candidate.quality_notes)
        if is_ambiguous:
            notes.append("ambiguous_cross_discipline" if discipline_count > 1 else "ambiguous_cross_source")
        rows.append(
            {
                "signal_id": signal_id_for(candidate),
                "signal_type": candidate.signal_type,
                "candidate": candidate.candidate,
                "candidate_normalized": candidate.candidate_normalized,
                "source_id": candidate.source_id,
                "source_title": candidate.source_title,
                "source_type": candidate.source_type,
                "discipline_id": candidate.discipline_id,
                "candidate_disciplines": json.dumps(discipline_ids, ensure_ascii=False),
                "provenance_kind": candidate.provenance_kind,
                "provenance_ref": candidate.provenance_ref,
                "hierarchy_path": candidate.hierarchy_path,
                "article_number": candidate.article_number,
                "frequency": candidate.frequency,
                "source_count": source_count,
                "discipline_count": discipline_count,
                "specificity_score": specificity_score(candidate, discipline_count),
                "is_ambiguous": "true" if is_ambiguous else "false",
                "quality_notes": ";".join(notes),
            }
        )
    return rows


def write_candidates(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = candidate_rows(candidates)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def manifest_text_path_under_text_dir(local_path_text: str) -> Path:
    local_path = Path(local_path_text)
    if local_path.is_absolute():
        return local_path
    if len(local_path.parts) >= 3 and local_path.parts[:2] == ("extractions", "clean_text"):
        return Path(*local_path.parts[2:])
    if local_path.parts[:1] == (DEFAULT_TEXT_DIR.name,) and len(local_path.parts) > 1:
        return Path(*local_path.parts[1:])
    return local_path


def resolve_text_path(source: SourceMeta, text_dir: Path) -> Path:
    local_path = Path(source.local_path_text)
    if local_path.is_absolute():
        return local_path
    if text_dir.resolve() == DEFAULT_TEXT_DIR.resolve():
        return SOURCES_DIR / local_path

    candidates = [
        text_dir / manifest_text_path_under_text_dir(source.local_path_text),
        text_dir / local_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def extract_candidates(
    sources: list[SourceMeta],
    text_dir: Path,
    stopwords: set[str],
    min_phrase_frequency: int,
) -> tuple[list[Candidate], list[SourceMeta]]:
    candidates: list[Candidate] = []
    skipped: list[SourceMeta] = []
    all_heading_occurrences: list[tuple[str, SourceMeta, str, str]] = []

    for source in sources:
        text_path = resolve_text_path(source, text_dir)
        if not text_path.exists():
            fallback_path = text_dir / f"{source.source_id}.txt"
            text_path = fallback_path
        if not text_path.exists():
            skipped.append(source)
            continue

        source_candidates, heading_occurrences = extract_source_candidates(source, text_path, stopwords)
        candidates.extend(source_candidates)
        all_heading_occurrences.extend(heading_occurrences)

    candidates.extend(heading_topic_phrases(all_heading_occurrences, stopwords, min_phrase_frequency))
    return dedupe_candidates(candidates), skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract discipline signal candidates from cleaned LCP95-style legal text."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT_DIR)
    parser.add_argument("--stopwords", type=Path, default=DEFAULT_STOPWORDS)
    parser.add_argument("--lcp95-html", type=Path, default=DEFAULT_LCP95_HTML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-phrase-frequency", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_lcp95_reference(args.lcp95_html)
    sources = read_manifest(args.manifest)
    stopwords = read_stopwords(args.stopwords)
    candidates, skipped = extract_candidates(
        sources,
        args.text_dir,
        stopwords,
        min_phrase_frequency=args.min_phrase_frequency,
    )
    write_candidates(args.output, candidates)
    processed_count = len(sources) - len(skipped)
    print(f"Read {len(sources)} manifest sources from {args.manifest}")
    print(f"Processed {processed_count} cleaned text sources from {args.text_dir}")
    if skipped:
        for source in skipped:
            print(f"Skipped missing cleaned text: {source.source_id}: {source.local_path_text}")
    print(f"Wrote {len(candidates)} signal candidates to {args.output}")


if __name__ == "__main__":
    main()
