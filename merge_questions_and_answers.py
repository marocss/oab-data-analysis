from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"

QUESTIONS_CSV = PROCESSED_DIR / "questions_raw.csv"
ANSWERS_CSV = PROCESSED_DIR / "answers_raw.csv"
MERGED_CSV = PROCESSED_DIR / "questions_and_answers_raw.csv"
VALIDATION_CSV = PROCESSED_DIR / "questions_and_answers_validation_report.csv"

MERGED_FIELDNAMES = [
    "exam",
    "question_number",
    "question_text",
    "alt_a",
    "alt_b",
    "alt_c",
    "alt_d",
    "answer",
    "question_source_pdf",
    "answer_source_pdf",
    "question_parse_ok",
    "answer_parse_ok",
]

VALIDATION_FIELDNAMES = [
    "question_rows",
    "answer_rows",
    "merged_rows",
    "annulled_skipped",
    "missing_answers",
    "missing_questions",
    "duplicate_question_keys",
    "duplicate_answer_keys",
    "parse_ok",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path.relative_to(ROOT)}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> tuple[int, int]:
    return int(row["exam"]), int(row["question_number"])


def format_keys(keys: list[tuple[int, int]]) -> str:
    return " ".join(f"{exam}:{question_number}" for exam, question_number in sorted(keys))


def duplicate_keys(rows: list[dict[str, str]]) -> list[tuple[int, int]]:
    counts = Counter(row_key(row) for row in rows)
    return [key for key, count in sorted(counts.items()) if count > 1]


def bool_text(value: str) -> bool:
    return value.strip().lower() == "true"


def is_annulled(row: dict[str, str]) -> bool:
    return row["answer"] == "*" or bool_text(row["is_annulled"])


def build_validation_row(
    question_rows: list[dict[str, str]],
    answer_rows: list[dict[str, str]],
    merged_rows: list[dict[str, object]],
    annulled_skipped: int,
) -> dict[str, object]:
    question_keys = {row_key(row) for row in question_rows}
    answer_keys = {row_key(row) for row in answer_rows}
    missing_answers = sorted(question_keys - answer_keys)
    missing_questions = sorted(answer_keys - question_keys)
    duplicate_question_keys = duplicate_keys(question_rows)
    duplicate_answer_keys = duplicate_keys(answer_rows)
    merged_has_annulled = any(row["answer"] == "*" for row in merged_rows)
    expected_merged_rows = len(question_rows) - annulled_skipped
    merged_count_ok = len(merged_rows) == expected_merged_rows
    input_parse_ok = all(
        bool_text(row.get("parse_ok", "False")) for row in question_rows + answer_rows
    )

    notes: list[str] = []
    if missing_answers:
        notes.append("one or more questions are missing answers")
    if missing_questions:
        notes.append("one or more answers are missing questions")
    if duplicate_question_keys:
        notes.append("duplicate question keys")
    if duplicate_answer_keys:
        notes.append("duplicate answer keys")
    if merged_has_annulled:
        notes.append("merged output contains annulled answers")
    if not merged_count_ok:
        notes.append(f"expected {expected_merged_rows} merged rows")
    if not input_parse_ok:
        notes.append("one or more input rows have parse_ok=False")

    parse_ok = not any(
        [
            missing_answers,
            missing_questions,
            duplicate_question_keys,
            duplicate_answer_keys,
            merged_has_annulled,
            not merged_count_ok,
            not input_parse_ok,
        ]
    )

    return {
        "question_rows": len(question_rows),
        "answer_rows": len(answer_rows),
        "merged_rows": len(merged_rows),
        "annulled_skipped": annulled_skipped,
        "missing_answers": format_keys(missing_answers),
        "missing_questions": format_keys(missing_questions),
        "duplicate_question_keys": format_keys(duplicate_question_keys),
        "duplicate_answer_keys": format_keys(duplicate_answer_keys),
        "parse_ok": parse_ok,
        "notes": "; ".join(notes),
    }


def merge_rows(
    question_rows: list[dict[str, str]],
    answer_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], int]:
    answers_by_key = {row_key(row): row for row in answer_rows}
    merged_rows: list[dict[str, object]] = []
    annulled_skipped = 0

    for question in sorted(question_rows, key=row_key):
        answer = answers_by_key.get(row_key(question))
        if answer is None:
            continue
        if is_annulled(answer):
            annulled_skipped += 1
            continue
        merged_rows.append(
            {
                "exam": question["exam"],
                "question_number": question["question_number"],
                "question_text": question["question_text"],
                "alt_a": question["alt_a"],
                "alt_b": question["alt_b"],
                "alt_c": question["alt_c"],
                "alt_d": question["alt_d"],
                "answer": answer["answer"],
                "question_source_pdf": question["source_pdf"],
                "answer_source_pdf": answer["source_pdf"],
                "question_parse_ok": question["parse_ok"],
                "answer_parse_ok": answer["parse_ok"],
            }
        )
    return merged_rows, annulled_skipped


def main() -> None:
    question_rows = read_csv(QUESTIONS_CSV)
    answer_rows = read_csv(ANSWERS_CSV)
    merged_rows, annulled_skipped = merge_rows(question_rows, answer_rows)
    validation_row = build_validation_row(
        question_rows,
        answer_rows,
        merged_rows,
        annulled_skipped,
    )

    write_csv(MERGED_CSV, merged_rows, MERGED_FIELDNAMES)
    write_csv(VALIDATION_CSV, [validation_row], VALIDATION_FIELDNAMES)

    print(f"Wrote {len(merged_rows)} rows to {MERGED_CSV.relative_to(ROOT)}")
    print(f"Wrote validation report to {VALIDATION_CSV.relative_to(ROOT)}")
    if not validation_row["parse_ok"]:
        print("Merge validation failed:", validation_row["notes"])
        raise SystemExit(1)
    print("Merged questions and answers validated.")


if __name__ == "__main__":
    main()
