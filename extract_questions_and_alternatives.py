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

QUESTIONS_CSV = PROCESSED_DIR / "questions_raw.csv"
VALIDATION_CSV = PROCESSED_DIR / "validation_report.csv"

EXAM_DIR_RE = re.compile(r"^(?P<exam>\d+)-exam$")
QUESTION_MARKER_RE = re.compile(r"(?m)^[ \t]*(?P<number>[1-9]|[1-7][0-9]|80)[ \t]*$")
ALT_MARKER_RE = re.compile(r"(?m)^[ \t]*\(?([ABCD])\)[ \t]*")


@dataclass(frozen=True)
class PdfCandidate:
    path: Path
    pages: int
    chars: int
    title_hits: int
    applied_hits: int
    question_marker_hits: int
    alternative_marker_hits: int
    gabarito_hits: int
    score: int


def iter_exam_dirs() -> list[tuple[int, Path]]:
    exam_dirs: list[tuple[int, Path]] = []
    for path in EXAMS_DIR.iterdir():
        if not path.is_dir():
            continue
        match = EXAM_DIR_RE.fullmatch(path.name)
        if match:
            exam_dirs.append((int(match.group("exam")), path))
    return sorted(exam_dirs)


def extract_pages(path: Path) -> list[str]:
    with pymupdf.open(path) as doc:
        return [page.get_text("text") for page in doc]


def score_pdf(path: Path) -> PdfCandidate:
    pages = extract_pages(path)
    text = "\n".join(pages)
    upper_text = text.upper()
    title_hits = upper_text.count("EXAME DE ORDEM UNIFICADO")
    applied_hits = upper_text.count("PROVA APLICADA")
    question_marker_hits = len(QUESTION_MARKER_RE.findall(text))
    alternative_marker_hits = len(ALT_MARKER_RE.findall(text))
    gabarito_hits = upper_text.count("GABARITO")

    score = 0
    if title_hits:
        score += 5
    if applied_hits:
        score += 5
    if len(pages) >= 10:
        score += 20
    if len(text) >= 50_000:
        score += 20
    if question_marker_hits >= 80:
        score += 10
    if alternative_marker_hits >= 300:
        score += 30
    if gabarito_hits:
        score -= 40
    if len(pages) <= 5:
        score -= 15

    return PdfCandidate(
        path=path,
        pages=len(pages),
        chars=len(text),
        title_hits=title_hits,
        applied_hits=applied_hits,
        question_marker_hits=question_marker_hits,
        alternative_marker_hits=alternative_marker_hits,
        gabarito_hits=gabarito_hits,
        score=score,
    )


def identify_prova_pdf(exam_dir: Path) -> Path:
    candidates = [score_pdf(path) for path in sorted(exam_dir.glob("*.pdf"))]
    likely = [
        candidate
        for candidate in candidates
        if candidate.pages >= 10
        and candidate.chars >= 50_000
        and candidate.alternative_marker_hits >= 300
        and candidate.gabarito_hits == 0
    ]

    if len(likely) == 1:
        return likely[0].path

    print(f"Could not confidently identify prova PDF in {exam_dir}:")
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        print(
            "  "
            f"{candidate.path.name}: score={candidate.score}, pages={candidate.pages}, "
            f"chars={candidate.chars}, title_hits={candidate.title_hits}, "
            f"applied_hits={candidate.applied_hits}, "
            f"question_markers={candidate.question_marker_hits}, "
            f"alternative_markers={candidate.alternative_marker_hits}, "
            f"gabarito_hits={candidate.gabarito_hits}"
        )
    raise RuntimeError(f"Uncertain prova PDF detection for {exam_dir.name}")


def is_header_line(line: str) -> bool:
    stripped = line.strip()
    upper = stripped.upper()
    if not stripped:
        return False
    if "EXAME" in upper and "ORDEM UNIFICADO" in upper and "TIPO" in upper:
        return True
    if "EXAME" in upper and "ORDEM UNIFICADO" in upper and len(stripped) < 80:
        return True
    if upper.startswith("PROVA APLICADA EM "):
        return True
    if re.fullmatch(r"TIPO\s+1\s*[–-].*P[ÁA]GINA\s+\d+", upper):
        return True
    return False


def remove_repeated_page_headers(pages: list[str]) -> str:
    cleaned_pages: list[str] = []
    for page_index, page_text in enumerate(pages, start=1):
        cleaned_lines: list[str] = []
        for line_index, line in enumerate(page_text.splitlines()):
            stripped = line.strip()
            if is_header_line(line):
                continue
            if line_index <= 20 and stripped == str(page_index):
                continue
            cleaned_lines.append(line.rstrip())
        cleaned_pages.append("\n".join(cleaned_lines))
    return "\n".join(cleaned_pages)


def trim_to_questions(text: str) -> str:
    questionnaire_match = re.search(
        r"(?im)^[ \t]*Questionário de percepção sobre a prova[ \t]*$", text
    )
    if questionnaire_match:
        text = text[: questionnaire_match.start()]

    first_question_match = QUESTION_MARKER_RE.search(text)
    if not first_question_match:
        return text

    while first_question_match and first_question_match.group("number") != "1":
        first_question_match = QUESTION_MARKER_RE.search(text, first_question_match.end())

    if not first_question_match:
        return text
    return text[first_question_match.start() :]


def split_question_blocks(text: str) -> list[tuple[int, str]]:
    matches = list(QUESTION_MARKER_RE.finditer(text))
    accepted: list[re.Match[str]] = []
    search_start = 0

    for expected_number in range(1, 81):
        candidates = [
            match
            for match in matches
            if int(match.group("number")) == expected_number and match.start() >= search_start
        ]
        if not candidates:
            break

        selected = candidates[0]
        next_expected = expected_number + 1
        for candidate in candidates:
            if next_expected > 80:
                block_end = len(text)
            else:
                next_marker = next(
                    (
                        match
                        for match in matches
                        if int(match.group("number")) == next_expected
                        and match.start() > candidate.end()
                    ),
                    None,
                )
                block_end = next_marker.start() if next_marker else len(text)
            block = text[candidate.end() : block_end]
            if set(ALT_MARKER_RE.findall(block)) >= {"A", "B", "C", "D"}:
                selected = candidate
                break

        accepted.append(selected)
        search_start = selected.end()

    blocks: list[tuple[int, str]] = []
    for index, match in enumerate(accepted):
        number = int(match.group("number"))
        start = match.end()
        end = accepted[index + 1].start() if index + 1 < len(accepted) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append((number, block))
    return blocks


def clean_field(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def parse_question_block(exam: int, number: int, raw_block: str, source_pdf: Path) -> dict[str, object]:
    alt_matches = list(ALT_MARKER_RE.finditer(raw_block))
    alt_positions: dict[str, re.Match[str]] = {}
    search_start = 0
    for label in ("A", "B", "C", "D"):
        next_match = next(
            (
                match
                for match in alt_matches
                if match.group(1) == label and match.start() >= search_start
            ),
            None,
        )
        if next_match is None:
            break
        alt_positions[label] = next_match
        search_start = next_match.end()

    has_all_alternatives = set(alt_positions) == {"A", "B", "C", "D"}
    question_text = ""
    alternatives = {"A": "", "B": "", "C": "", "D": ""}

    if has_all_alternatives:
        question_text = clean_field(raw_block[: alt_positions["A"].start()])
        ordered_labels = ("A", "B", "C", "D")
        for index, label in enumerate(ordered_labels):
            start = alt_positions[label].end()
            if index + 1 < len(ordered_labels):
                end = alt_positions[ordered_labels[index + 1]].start()
            else:
                end = len(raw_block)
            alternatives[label] = clean_field(raw_block[start:end])

    parse_ok = bool(
        has_all_alternatives
        and question_text
        and all(alternatives[label] for label in ("A", "B", "C", "D"))
    )

    return {
        "exam": exam,
        "question_number": number,
        "question_text": question_text,
        "alt_a": alternatives["A"],
        "alt_b": alternatives["B"],
        "alt_c": alternatives["C"],
        "alt_d": alternatives["D"],
        "raw_block": clean_field(raw_block),
        "source_pdf": str(source_pdf.relative_to(ROOT)),
        "parse_ok": parse_ok,
    }


def validate_exam(exam: int, rows: list[dict[str, object]], extracted_text: str) -> dict[str, object]:
    question_numbers = [int(row["question_number"]) for row in rows]
    counts = Counter(question_numbers)
    missing = [number for number in range(1, 81) if number not in counts]
    duplicates = [number for number, count in sorted(counts.items()) if count > 1]
    alternatives_ok_count = sum(
        1
        for row in rows
        if all(row[field] for field in ("alt_a", "alt_b", "alt_c", "alt_d"))
    )
    question_count_ok = len(rows) == 80
    number_set_ok = not missing and not duplicates and set(question_numbers) == set(range(1, 81))
    rows_parse_ok = all(bool(row["parse_ok"]) for row in rows)
    parse_ok = question_count_ok and number_set_ok and alternatives_ok_count == 80 and rows_parse_ok

    notes: list[str] = []
    if not question_count_ok:
        notes.append(f"expected 80 questions, found {len(rows)}")
    if missing:
        notes.append("missing question numbers")
    if duplicates:
        notes.append("duplicate question numbers")
    if alternatives_ok_count != 80:
        notes.append(f"{80 - alternatives_ok_count} question(s) missing alternatives")
    if not rows_parse_ok:
        notes.append("one or more question blocks failed to parse")

    debug_path = DEBUG_DIR / f"exam_{exam}_extracted_text.txt"
    if not parse_ok:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(extracted_text, encoding="utf-8")
    elif debug_path.exists():
        debug_path.unlink()

    return {
        "exam": exam,
        "question_count": len(rows),
        "missing_questions": " ".join(str(number) for number in missing),
        "duplicate_questions": " ".join(str(number) for number in duplicates),
        "alternatives_ok_count": alternatives_ok_count,
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
    question_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for exam, exam_dir in iter_exam_dirs():
        source_pdf = identify_prova_pdf(exam_dir)
        pages = extract_pages(source_pdf)
        extracted_text = trim_to_questions(remove_repeated_page_headers(pages))
        blocks = split_question_blocks(extracted_text)
        rows = [
            parse_question_block(exam, question_number, raw_block, source_pdf)
            for question_number, raw_block in blocks
        ]

        question_rows.extend(rows)
        validation_rows.append(validate_exam(exam, rows, extracted_text))

    write_csv(
        QUESTIONS_CSV,
        question_rows,
        [
            "exam",
            "question_number",
            "question_text",
            "alt_a",
            "alt_b",
            "alt_c",
            "alt_d",
            "raw_block",
            "source_pdf",
            "parse_ok",
        ],
    )
    write_csv(
        VALIDATION_CSV,
        validation_rows,
        [
            "exam",
            "question_count",
            "missing_questions",
            "duplicate_questions",
            "alternatives_ok_count",
            "parse_ok",
            "notes",
        ],
    )

    failed = [row for row in validation_rows if not row["parse_ok"]]
    print(f"Wrote {len(question_rows)} rows to {QUESTIONS_CSV.relative_to(ROOT)}")
    print(f"Wrote validation report to {VALIDATION_CSV.relative_to(ROOT)}")
    if failed:
        print("Validation failed for exams:", ", ".join(str(row["exam"]) for row in failed))
        print(f"Debug text written to {DEBUG_DIR.relative_to(ROOT)}")
        raise SystemExit(1)
    print("All exams validated.")


if __name__ == "__main__":
    main()
