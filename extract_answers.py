from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parent
EXAMS_DIR = ROOT / "phase-1-exams"
PROCESSED_DIR = ROOT / "data" / "processed"
DEBUG_DIR = ROOT / "data" / "debug"

ANSWERS_CSV = PROCESSED_DIR / "answers_raw.csv"
VALIDATION_CSV = PROCESSED_DIR / "answers_validation_report.csv"

EXAM_DIR_RE = re.compile(r"^(?P<exam>\d+)-exam$")
QUESTION_MARKER_RE = re.compile(r"(?m)^[ \t]*(?P<number>[1-9]|[1-7][0-9]|80)[ \t]*$")
ALT_MARKER_RE = re.compile(r"(?m)^[ \t]*\(?([ABCD])\)[ \t]*")
PROVA_TYPE_RE = re.compile(r"\bTIPO\s*1\b.*\bBRANC[AO]\b|\bBRANC[AO]\b.*\bTIPO\s*1\b", re.I)
TYPE_1_HEADER_RE = re.compile(r"\bTIPO\s*1\b|\bPROVA\s+TIPO\s*1\b|\bPROVA\s+1\b", re.I)
SECTION_END_RE = re.compile(
    r"\bTIPO\s*2\b|\bPROVA\s+TIPO\s*2\b|\bPROVA\s+2\b|GABARITOS\s+PRELIMINARES|"
    r"TABELA\s+DE\s+CORRESPOND",
    re.I,
)
ANSWER_TOKEN_RE = re.compile(r"^(?:[1-9]|[1-7][0-9]|80|[ABCD]|\*)$")
ANSWER_VALUES = {"A", "B", "C", "D", "*"}


@dataclass(frozen=True)
class PdfCandidate:
    path: Path
    pages: int
    chars: int
    gabarito_hits: int
    question_marker_hits: int
    alternative_marker_hits: int
    type_1_header_hits: int
    prova_type_ok: bool


def iter_exam_dirs() -> list[tuple[int, Path]]:
    exam_dirs: list[tuple[int, Path]] = []
    for path in EXAMS_DIR.iterdir():
        if not path.is_dir():
            continue
        match = EXAM_DIR_RE.fullmatch(path.name)
        if match:
            exam_dirs.append((int(match.group("exam")), path))
    return sorted(exam_dirs)


def extract_text(path: Path) -> str:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def score_pdf(path: Path) -> PdfCandidate:
    text = extract_text(path)
    upper_text = text.upper()
    with pymupdf.open(path) as doc:
        page_count = doc.page_count

    return PdfCandidate(
        path=path,
        pages=page_count,
        chars=len(text),
        gabarito_hits=upper_text.count("GABARITO"),
        question_marker_hits=len(QUESTION_MARKER_RE.findall(text)),
        alternative_marker_hits=len(ALT_MARKER_RE.findall(text)),
        type_1_header_hits=len(TYPE_1_HEADER_RE.findall(text)),
        prova_type_ok=bool(PROVA_TYPE_RE.search(text)),
    )


def identify_exam_pdfs(exam_dir: Path) -> tuple[Path, Path]:
    candidates = [score_pdf(path) for path in sorted(exam_dir.glob("*.pdf"))]
    prova_candidates = [
        candidate
        for candidate in candidates
        if candidate.pages >= 10
        and candidate.chars >= 50_000
        and candidate.alternative_marker_hits >= 300
        and candidate.gabarito_hits == 0
    ]
    gabarito_candidates = [
        candidate
        for candidate in candidates
        if candidate.pages <= 5 and candidate.gabarito_hits > 0 and candidate.type_1_header_hits > 0
    ]

    if len(prova_candidates) == 1 and len(gabarito_candidates) == 1:
        return prova_candidates[0].path, gabarito_candidates[0].path

    print(f"Could not confidently identify PDFs in {exam_dir}:")
    for candidate in candidates:
        print(
            "  "
            f"{candidate.path.name}: pages={candidate.pages}, chars={candidate.chars}, "
            f"gabarito_hits={candidate.gabarito_hits}, "
            f"question_markers={candidate.question_marker_hits}, "
            f"alternative_markers={candidate.alternative_marker_hits}, "
            f"type_1_headers={candidate.type_1_header_hits}, "
            f"prova_type_ok={candidate.prova_type_ok}"
        )
    raise RuntimeError(f"Uncertain PDF detection for {exam_dir.name}")


def normalize_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def find_type_1_answer_section(lines: list[str]) -> list[str]:
    definitive_start = next(
        (index for index, line in enumerate(lines) if "GABARITOS DEFINITIVOS" in line.upper()),
        0,
    )
    section_start = next(
        (
            index
            for index, line in enumerate(lines[definitive_start:], definitive_start)
            if TYPE_1_HEADER_RE.search(line)
        ),
        None,
    )
    if section_start is None:
        return []

    section_end = next(
        (
            index
            for index, line in enumerate(lines[section_start + 1 :], section_start + 1)
            if SECTION_END_RE.search(line)
        ),
        len(lines),
    )
    return lines[section_start:section_end]


def tokenize_answer_section(lines: list[str]) -> list[str]:
    tokens: list[str] = []
    for line in lines:
        for token in line.split():
            if ANSWER_TOKEN_RE.fullmatch(token):
                tokens.append(token)
    return tokens


def extract_type_1_answers(text: str) -> list[tuple[int, str]]:
    lines = normalize_lines(text)
    section_lines = find_type_1_answer_section(lines)
    tokens = tokenize_answer_section(section_lines)

    answers: list[tuple[int, str]] = []
    search_start = 0
    for block_start in (1, 21, 41, 61):
        expected_numbers = [str(number) for number in range(block_start, block_start + 20)]
        found_block = False
        for index in range(search_start, max(search_start, len(tokens) - 39) + 1):
            answer_slice = tokens[index + 20 : index + 40]
            if tokens[index : index + 20] == expected_numbers and all(
                token in ANSWER_VALUES for token in answer_slice
            ):
                answers.extend(
                    (question_number, answer)
                    for question_number, answer in zip(
                        range(block_start, block_start + 20), answer_slice
                    )
                )
                search_start = index + 40
                found_block = True
                break
        if not found_block:
            break
    return answers


def build_answer_rows(
    exam: int,
    answers: list[tuple[int, str]],
    source_pdf: Path,
    prova_pdf: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for question_number, answer in answers:
        is_annulled = answer == "*"
        rows.append(
            {
                "exam": exam,
                "question_number": question_number,
                "answer": answer,
                "is_annulled": is_annulled,
                "source_pdf": str(source_pdf.relative_to(ROOT)),
                "prova_pdf": str(prova_pdf.relative_to(ROOT)),
                "parse_ok": answer in ANSWER_VALUES,
            }
        )
    return rows


def validate_answers(
    exam: int,
    rows: list[dict[str, object]],
    prova_type_ok: bool,
    answer_text: str,
) -> dict[str, object]:
    question_numbers = [int(row["question_number"]) for row in rows]
    counts = Counter(question_numbers)
    missing = [number for number in range(1, 81) if number not in counts]
    duplicates = [number for number, count in sorted(counts.items()) if count > 1]
    invalid_answers = [
        row for row in rows if row["answer"] not in ANSWER_VALUES or row["is_annulled"] != (row["answer"] == "*")
    ]
    annulled_count = sum(1 for row in rows if row["answer"] == "*")

    answer_count_ok = len(rows) == 80
    number_set_ok = not missing and not duplicates and set(question_numbers) == set(range(1, 81))
    rows_parse_ok = all(bool(row["parse_ok"]) for row in rows)
    parse_ok = (
        prova_type_ok
        and answer_count_ok
        and number_set_ok
        and not invalid_answers
        and rows_parse_ok
    )

    notes: list[str] = []
    if not prova_type_ok:
        notes.append("prova PDF is not confirmed as TIPO 1 / BRANCA")
    if not answer_count_ok:
        notes.append(f"expected 80 answers, found {len(rows)}")
    if missing:
        notes.append("missing question numbers")
    if duplicates:
        notes.append("duplicate question numbers")
    if invalid_answers:
        notes.append("one or more answers are invalid")
    if not rows_parse_ok:
        notes.append("one or more answer rows failed to parse")

    debug_path = DEBUG_DIR / f"exam_{exam}_answers_extracted_text.txt"
    if not parse_ok:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(answer_text, encoding="utf-8")
    elif debug_path.exists():
        debug_path.unlink()

    return {
        "exam": exam,
        "answer_count": len(rows),
        "missing_questions": " ".join(str(number) for number in missing),
        "duplicate_questions": " ".join(str(number) for number in duplicates),
        "annulled_count": annulled_count,
        "prova_type_ok": prova_type_ok,
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
    answer_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for exam, exam_dir in iter_exam_dirs():
        prova_pdf, gabarito_pdf = identify_exam_pdfs(exam_dir)
        prova_text = extract_text(prova_pdf)
        answer_text = extract_text(gabarito_pdf)
        prova_type_ok = bool(PROVA_TYPE_RE.search(prova_text))
        answers = extract_type_1_answers(answer_text)
        rows = build_answer_rows(exam, answers, gabarito_pdf, prova_pdf)

        answer_rows.extend(rows)
        validation_rows.append(validate_answers(exam, rows, prova_type_ok, answer_text))

    write_csv(
        ANSWERS_CSV,
        answer_rows,
        [
            "exam",
            "question_number",
            "answer",
            "is_annulled",
            "source_pdf",
            "prova_pdf",
            "parse_ok",
        ],
    )
    write_csv(
        VALIDATION_CSV,
        validation_rows,
        [
            "exam",
            "answer_count",
            "missing_questions",
            "duplicate_questions",
            "annulled_count",
            "prova_type_ok",
            "parse_ok",
            "notes",
        ],
    )

    failed = [row for row in validation_rows if not row["parse_ok"]]
    print(f"Wrote {len(answer_rows)} rows to {ANSWERS_CSV.relative_to(ROOT)}")
    print(f"Wrote validation report to {VALIDATION_CSV.relative_to(ROOT)}")
    if failed:
        print("Validation failed for exams:", ", ".join(str(row["exam"]) for row in failed))
        print(f"Debug text written to {DEBUG_DIR.relative_to(ROOT)}")
        raise SystemExit(1)
    print("All answer keys validated.")


if __name__ == "__main__":
    main()
