from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parent
EDITAIS_DIR = ROOT / "phase-1-editais"
SOURCES_DIR = ROOT / "sources"
DOCUMENTS_DIR = SOURCES_DIR / "documents"
PROCESSED_DIR = ROOT / "data" / "processed"

DISCIPLINES_CSV = PROCESSED_DIR / "exam_disciplines.csv"
VALIDATION_CSV = PROCESSED_DIR / "exam_disciplines_validation_report.csv"

EXPECTED_EXAMS = set(range(37, 47))

SOURCE_FILES = {
    "provimento_144": DOCUMENTS_DIR / "Provimento n. 144.2011.pdf",
    "rces_2018": DOCUMENTS_DIR / "rces005_18.pdf",
    "rces_2021": DOCUMENTS_DIR / "rces002_21.pdf",
    "exame_numeros": DOCUMENTS_DIR / "exame-de-ordem-em-numeros-IV.pdf",
}

EXAM_TITLE_RE = re.compile(
    r"(?P<exam>\d+)º\s+EXAME\s+DE\s+ORDEM(?:\s+UNIFICADO)?",
    re.IGNORECASE,
)
P1_RE = re.compile(r"\(P1\)\s*PROVA\s+OBJETIVA", re.IGNORECASE)
AREA_START_RE = re.compile(
    r"Disciplinas\s+profissionalizantes|Conte[uú]dos\s+de\s+forma[cç][aã]o",
    re.IGNORECASE,
)
AREA_END_RE = re.compile(
    r"N[uú]mero\s+de\s+quest[oõ]es|\(P2\)\s*PROVA\s+PR[AÁ]TICO-PROFISSIONAL",
    re.IGNORECASE,
)

CATEGORY_PRIORITY = {
    "formacao_geral": 1,
    "campo_complementar": 2,
    "formacao_tecnico_juridica": 3,
    "oab_specific": 4,
}

EXPLICIT_ALIASES = {
    "Direitos Humanos": ["Direitos Humanos"],
    "Direito do Consumidor": [
        "Direito do Consumidor",
        "Código do Consumidor",
        "Código de Defesa do Consumidor",
        "Lei 8.078/1990",
    ],
    "Direito da Criança e do Adolescente": [
        "Direito da Criança e do Adolescente",
        "Estatuto da Criança e do Adolescente",
    ],
    "Direito Ambiental": ["Direito Ambiental"],
    "Direito Internacional": ["Direito Internacional"],
    "Filosofia do Direito": ["Filosofia do Direito"],
    "Direito Financeiro": ["Direito Financeiro"],
    "Direito Previdenciário": ["Direito Previdenciário"],
    "Direito Eleitoral": ["Direito Eleitoral"],
    "Código de Ética e Estatuto da OAB": [
        "Estatuto da Advocacia",
        "Regulamento Geral",
        "Código de Ética",
        "Código de Ética e Disciplina",
    ],
}

CANONICAL_ALIASES = {
    "Código do Consumidor": "Direito do Consumidor",
    "Código de Defesa do Consumidor": "Direito do Consumidor",
    "Lei 8.078/1990": "Direito do Consumidor",
    "Estatuto da Criança e do Adolescente": "Direito da Criança e do Adolescente",
}

PROCESSUAL_SPLITS = (
    "Direito Processual Civil",
    "Direito Processual Penal",
    "Direito Processual do Trabalho",
)


@dataclass(frozen=True)
class DisciplineSpec:
    name: str
    category: str
    source_basis: frozenset[str]
    source_pdfs: frozenset[str]
    question_count_hint: str = ""


@dataclass
class DisciplineAggregate:
    name: str
    category: str
    source_basis: set[str] = field(default_factory=set)
    source_pdfs: set[str] = field(default_factory=set)
    question_count_hint: str = ""


def extract_text(path: Path) -> str:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return normalize_space(ascii_text).lower()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def canonical_name(name: str) -> str:
    cleaned = normalize_space(name).strip(" .,;:")
    cleaned = re.sub(r"^(?:e|ou)\s+", "", cleaned, flags=re.IGNORECASE)
    return CANONICAL_ALIASES.get(cleaned, cleaned)


def clean_item(item: str) -> str:
    cleaned = normalize_space(item)
    cleaned = re.sub(r"\(NR\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:e|ou)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" .,;:“”\"")


def split_items(items_text: str) -> list[str]:
    items_text = normalize_space(items_text)
    items_text = re.sub(r";\s*e\s*(?:\(NR\))?.*$", "", items_text, flags=re.IGNORECASE)
    parts = re.split(r",\s+|;\s+", items_text)
    split_parts: list[str] = []
    for part in parts:
        part = clean_item(part)
        if not part:
            continue
        trailing_split = re.split(r"\s+e\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])", part)
        split_parts.extend(clean_item(piece) for piece in trailing_split if clean_item(piece))
    return split_parts


def extract_cne_general_disciplines(text: str) -> list[str]:
    normalized = normalize_space(text)
    match = re.search(
        r"I\s*-\s*Formação geral,.*?tais como:\s*(?P<items>.*?)\s*;\s*II\s*-\s*Formação",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return []
    return [canonical_name(item) for item in split_items(match.group("items"))]


def extract_cne_technical_disciplines(text: str) -> list[str]:
    normalized = normalize_space(text)
    match = re.search(
        r"II\s*-\s*Formação.*?conteúdos essenciais referentes às áreas de\s*"
        r"(?P<items>.*?)\s*III\s*-\s*Formação",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return []

    disciplines: list[str] = []
    for item in split_items(match.group("items")):
        canonical = canonical_name(item)
        if canonical == "Direito Processual":
            disciplines.extend(PROCESSUAL_SPLITS)
        else:
            disciplines.append(canonical)
    return disciplines


def extract_cne_complementary_disciplines(text: str) -> list[str]:
    normalized = normalize_space(text)
    match = re.search(
        r"tais como:\s*(?P<items>Direito Ambiental,.*?Direito Portuário)",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return []
    return [canonical_name(item) for item in split_items(match.group("items"))]


def extract_question_count_hints(text: str) -> dict[str, str]:
    normalized = normalize_space(text).replace("ques- tões", "questões")
    start = normalized.find("As questões da primeira fase")
    end = normalized.find("Para a aprovação", start)
    if start == -1 or end == -1:
        return {}

    chunk = normalized[start:end]
    hints: dict[str, str] = {}
    for match in re.finditer(r"(?P<name>[A-ZÁÉÍÓÚÂÊÔÃÕÇ][^()]+?)\s*\((?P<count>\d+)", chunk):
        name = match.group("name")
        name = re.sub(r"^.*?áreas do conhecimento jurídico:\s*", "", name)
        canonical = canonical_name(name)
        hints[canonical] = match.group("count")
    return hints


def add_spec(
    catalog: dict[str, DisciplineSpec],
    name: str,
    category: str,
    basis: str,
    source_pdf: Path,
    question_count_hints: dict[str, str],
) -> None:
    canonical = canonical_name(name)
    source_pdf_text = relative(source_pdf)
    existing = catalog.get(canonical)
    basis_set = set(existing.source_basis) if existing else set()
    pdf_set = set(existing.source_pdfs) if existing else set()
    basis_set.add(basis)
    pdf_set.add(source_pdf_text)

    question_count_hint = question_count_hints.get(
        canonical,
        existing.question_count_hint if existing else "",
    )
    if question_count_hint:
        basis_set.add("Exame de Ordem em Números IV first-phase distribution")
        pdf_set.add(relative(SOURCE_FILES["exame_numeros"]))

    if existing and CATEGORY_PRIORITY[existing.category] > CATEGORY_PRIORITY[category]:
        category = existing.category

    catalog[canonical] = DisciplineSpec(
        name=canonical,
        category=category,
        source_basis=frozenset(basis_set),
        source_pdfs=frozenset(pdf_set),
        question_count_hint=question_count_hint,
    )


def build_catalog(source_texts: dict[str, str]) -> dict[str, DisciplineSpec]:
    question_count_hints = extract_question_count_hints(source_texts["exame_numeros"])
    catalog: dict[str, DisciplineSpec] = {}

    for discipline in extract_cne_general_disciplines(source_texts["rces_2021"]):
        add_spec(
            catalog,
            discipline,
            "formacao_geral",
            "CNE/CES Resolução 2/2021 formação geral",
            SOURCE_FILES["rces_2021"],
            question_count_hints,
        )

    for discipline in extract_cne_technical_disciplines(source_texts["rces_2021"]):
        add_spec(
            catalog,
            discipline,
            "formacao_tecnico_juridica",
            "CNE/CES Resolução 2/2021 formação técnico-jurídica",
            SOURCE_FILES["rces_2021"],
            question_count_hints,
        )

    for discipline in extract_cne_complementary_disciplines(source_texts["rces_2018"]):
        add_spec(
            catalog,
            discipline,
            "campo_complementar",
            "CNE/CES Resolução 5/2018 campos complementares",
            SOURCE_FILES["rces_2018"],
            question_count_hints,
        )

    for discipline in ("Direitos Humanos", "Filosofia do Direito", "Código de Ética e Estatuto da OAB"):
        add_spec(
            catalog,
            discipline,
            "oab_specific",
            "Provimento 144/2011 OAB exam content",
            SOURCE_FILES["provimento_144"],
            question_count_hints,
        )

    for discipline, count in question_count_hints.items():
        add_spec(
            catalog,
            discipline,
            catalog.get(discipline, DisciplineSpec(discipline, "formacao_tecnico_juridica", frozenset(), frozenset())).category,
            "Exame de Ordem em Números IV first-phase distribution",
            SOURCE_FILES["exame_numeros"],
            {discipline: count},
        )

    return catalog


def load_source_texts() -> dict[str, str]:
    missing = [relative(path) for path in SOURCE_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source PDFs: " + ", ".join(missing))
    return {key: extract_text(path) for key, path in SOURCE_FILES.items()}


def identify_edital_exam(path: Path) -> int:
    text = extract_text(path)
    match = EXAM_TITLE_RE.search(text)
    if not match:
        raise RuntimeError(f"Could not identify exam number in {relative(path)}")
    return int(match.group("exam"))


def clean_area_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = normalize_space(line)
        upper = stripped.upper()
        if not stripped:
            continue
        if "CONSELHO FEDERAL DA ORDEM DOS ADVOGADOS DO BRASIL" in upper:
            continue
        if "EXAME DE ORDEM" in upper and len(stripped) < 80:
            continue
        if upper == "EDITAL DE ABERTURA":
            continue
        if upper == "DAS PROVAS" or re.fullmatch(r"\d+\.\s*DAS PROVAS", upper):
            continue
        if re.fullmatch(r"\d+\.?", stripped):
            continue
        cleaned_lines.append(stripped)
    return normalize_space(" ".join(cleaned_lines))


def extract_first_phase_area(text: str) -> str:
    p1_match = P1_RE.search(text)
    search_start = p1_match.end() if p1_match else 0
    area_match = AREA_START_RE.search(text, search_start)
    if not area_match:
        return ""

    end_match = AREA_END_RE.search(text, area_match.end())
    end = end_match.start() if end_match else min(len(text), area_match.start() + 2_500)
    return clean_area_text(text[area_match.start() : end])


def detected_explicit_disciplines(area_text: str) -> set[str]:
    folded_area = fold_text(area_text)
    detected: set[str] = set()
    for canonical, aliases in EXPLICIT_ALIASES.items():
        if any(fold_text(alias) in folded_area for alias in aliases):
            detected.add(canonical)
    return detected


def add_aggregate(
    aggregates: dict[str, DisciplineAggregate],
    spec: DisciplineSpec,
    edital_pdf: Path,
    edital_trigger: str,
) -> None:
    aggregate = aggregates.get(spec.name)
    if aggregate is None:
        aggregate = DisciplineAggregate(name=spec.name, category=spec.category)
        aggregates[spec.name] = aggregate

    if CATEGORY_PRIORITY[spec.category] > CATEGORY_PRIORITY[aggregate.category]:
        aggregate.category = spec.category

    aggregate.source_basis.update(spec.source_basis)
    aggregate.source_basis.add(edital_trigger)
    aggregate.source_pdfs.update(spec.source_pdfs)
    if spec.question_count_hint:
        aggregate.question_count_hint = spec.question_count_hint


def build_exam_rows(
    exam: int,
    edital_pdf: Path,
    area_text: str,
    catalog: dict[str, DisciplineSpec],
) -> list[dict[str, object]]:
    aggregates: dict[str, DisciplineAggregate] = {}
    folded_area = fold_text(area_text)

    if "formacao geral" in folded_area:
        for spec in catalog.values():
            if spec.category == "formacao_geral":
                add_aggregate(aggregates, spec, edital_pdf, "edital first-phase area: formação geral")

    if "formacao tecnico" in folded_area or "disciplinas profissionalizantes" in folded_area:
        for spec in catalog.values():
            if spec.category == "formacao_tecnico_juridica":
                add_aggregate(
                    aggregates,
                    spec,
                    edital_pdf,
                    "edital first-phase area: formação técnico-jurídica/profissionalizante",
                )

    for discipline in detected_explicit_disciplines(area_text):
        spec = catalog.get(discipline)
        if spec is None:
            spec = DisciplineSpec(
                name=discipline,
                category="campo_complementar",
                source_basis=frozenset({"edital first-phase area explicit discipline"}),
                source_pdfs=frozenset(),
            )
        add_aggregate(aggregates, spec, edital_pdf, "edital first-phase area explicit discipline")

    rows: list[dict[str, object]] = []
    for aggregate in sorted(aggregates.values(), key=lambda item: (item.category, item.name)):
        rows.append(
            {
                "exam": exam,
                "discipline": aggregate.name,
                "category": aggregate.category,
                "question_count_hint": aggregate.question_count_hint,
                "source_basis": "; ".join(sorted(aggregate.source_basis)),
                "edital_source_pdf": relative(edital_pdf),
                "discipline_source_pdfs": "; ".join(sorted(aggregate.source_pdfs)),
                "parse_ok": bool(aggregate.source_pdfs),
            }
        )
    return rows


def validate_exam_rows(
    exam: int,
    area_text: str,
    rows: list[dict[str, object]],
    question_count_hints: dict[str, str],
) -> dict[str, object]:
    disciplines = [str(row["discipline"]) for row in rows]
    counts = Counter(disciplines)
    duplicates = [discipline for discipline, count in sorted(counts.items()) if count > 1]
    missing_source_matches = [
        str(row["discipline"]) for row in rows if not str(row["discipline_source_pdfs"]).strip()
    ]
    missing_count_hints = [
        discipline
        for discipline in disciplines
        if discipline in question_count_hints
        and not next(row for row in rows if row["discipline"] == discipline)["question_count_hint"]
    ]

    edital_area_found = bool(area_text)
    parse_ok = bool(
        edital_area_found
        and rows
        and not duplicates
        and not missing_source_matches
        and not missing_count_hints
        and all(bool(row["parse_ok"]) for row in rows)
    )

    notes: list[str] = []
    if not edital_area_found:
        notes.append("first-phase area text not found")
    if not rows:
        notes.append("no disciplines extracted")
    if duplicates:
        notes.append("duplicate disciplines")
    if missing_source_matches:
        notes.append("one or more disciplines have no source PDF match")
    if missing_count_hints:
        notes.append("one or more Exame em Números disciplines are missing count hints")

    return {
        "exam": exam,
        "discipline_count": len(rows),
        "edital_area_found": edital_area_found,
        "missing_source_matches": " ".join(missing_source_matches),
        "duplicate_disciplines": " ".join(duplicates),
        "parse_ok": parse_ok,
        "notes": "; ".join(notes),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source_texts = load_source_texts()
    catalog = build_catalog(source_texts)
    question_count_hints = extract_question_count_hints(source_texts["exame_numeros"])

    edital_paths = sorted(EDITAIS_DIR.glob("*.pdf"))
    editais_by_exam: dict[int, Path] = {}
    for path in edital_paths:
        exam = identify_edital_exam(path)
        if exam in editais_by_exam:
            raise RuntimeError(
                f"Duplicate edital for exam {exam}: "
                f"{relative(editais_by_exam[exam])} and {relative(path)}"
            )
        editais_by_exam[exam] = path

    found_exams = set(editais_by_exam)
    if found_exams != EXPECTED_EXAMS:
        missing = " ".join(str(exam) for exam in sorted(EXPECTED_EXAMS - found_exams))
        extra = " ".join(str(exam) for exam in sorted(found_exams - EXPECTED_EXAMS))
        raise RuntimeError(f"Unexpected edital exam set. Missing: {missing or '-'} Extra: {extra or '-'}")

    discipline_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for exam, edital_pdf in sorted(editais_by_exam.items()):
        edital_text = extract_text(edital_pdf)
        area_text = extract_first_phase_area(edital_text)
        rows = build_exam_rows(exam, edital_pdf, area_text, catalog)
        discipline_rows.extend(rows)
        validation_rows.append(validate_exam_rows(exam, area_text, rows, question_count_hints))

    write_csv(
        DISCIPLINES_CSV,
        discipline_rows,
        [
            "exam",
            "discipline",
            "category",
            "question_count_hint",
            "source_basis",
            "edital_source_pdf",
            "discipline_source_pdfs",
            "parse_ok",
        ],
    )
    write_csv(
        VALIDATION_CSV,
        validation_rows,
        [
            "exam",
            "discipline_count",
            "edital_area_found",
            "missing_source_matches",
            "duplicate_disciplines",
            "parse_ok",
            "notes",
        ],
    )

    failed = [row for row in validation_rows if not row["parse_ok"]]
    print(f"Wrote {len(discipline_rows)} rows to {DISCIPLINES_CSV.relative_to(ROOT)}")
    print(f"Wrote validation report to {VALIDATION_CSV.relative_to(ROOT)}")
    if failed:
        print("Validation failed for exams:", ", ".join(str(row["exam"]) for row in failed))
        raise SystemExit(1)
    print("Exam disciplines validated.")


if __name__ == "__main__":
    main()
