# OAB Data Analysis

A data-driven map of what the OAB 1st phase exam actually tests.

The long-term goal is to analyze recent OAB 1st phase exams and show which subjects, topics, laws, concepts, and question styles appear most often. The current milestone is intentionally narrower: extract the 80 objective questions, extract the TIPO 1 answer keys, and create a non-annulled combined raw CSV.

## Current Milestone

The repository currently parses prova PDFs and answer-key PDFs for exams 37 through 46.

Implemented in `extract_questions_and_alternatives.py`:

- scans folders under `phase-1-exams/` matching `<number>-exam`
- extracts the exam number from the folder name
- identifies the prova PDF among the PDFs in each exam folder
- reads selectable PDF text with PyMuPDF using `page.get_text("text")`
- removes repeated page headers and page numbers
- ignores the cover/instruction section before question 1
- stops before the perception questionnaire
- splits question blocks using standalone question numbers from 1 to 80
- parses `question_text`, `alt_a`, `alt_b`, `alt_c`, and `alt_d`
- preserves `raw_block` for debugging
- validates each exam before reporting success

Implemented in `extract_answers.py`:

- scans the same exam folders independently from `extract_questions_and_alternatives.py`
- identifies the prova PDF and the gabarito PDF in each folder
- confirms the prova PDF is TIPO 1 / BRANCA before accepting TIPO 1 answers
- extracts only definitive TIPO 1 / PROVA 1 answers
- supports annulled questions marked with `*`
- keeps answers separate from `questions_raw.csv`
- validates each answer key before reporting success

Implemented in `merge_questions_and_answers.py`:

- reads `questions_raw.csv` and `answers_raw.csv`
- merges rows by `exam` and `question_number`
- excludes annulled questions from the combined output
- writes a combined raw CSV with question text, alternatives, answer, and source PDF provenance
- validates the merge before reporting success

Not implemented yet:

- subject/topic labels
- legal citation extraction
- OCR or image processing
- LLM classification

## Folder Structure

Expected input layout:

```text
phase-1-exams/
  37-exam/
    <original random filename>.pdf
    <original random filename>.pdf
  38-exam/
    <original random filename>.pdf
    <original random filename>.pdf
  ...
  46-exam/
    <original random filename>.pdf
    <original random filename>.pdf
```

Each exam folder currently contains two PDFs: the prova and the answer key. Question extraction and answer extraction are intentionally separate scripts and outputs.

Generated output layout:

```text
data/
  processed/
    questions_raw.csv
    validation_report.csv
    answers_raw.csv
    answers_validation_report.csv
    questions_and_answers_raw.csv
    questions_and_answers_validation_report.csv
  debug/
    exam_<number>_extracted_text.txt
    exam_<number>_answers_extracted_text.txt
```

Debug text files are only written when validation fails.

## PyMuPDF Environment

This project uses PyMuPDF because the official PDF text is selectable and `page.get_text("text")` returns usable ordered text. The parser expects PyMuPDF to be available in a local virtual environment named `pymupdf-venv`.

Create the environment:

```bash
python3 -m venv pymupdf-venv
```

Install PyMuPDF:

```bash
pymupdf-venv/bin/python -m pip install --upgrade pip
pymupdf-venv/bin/python -m pip install PyMuPDF
```

Check the installation:

```bash
pymupdf-venv/bin/python -c "import pymupdf; print(pymupdf.version)"
```

Run the question parser:

```bash
pymupdf-venv/bin/python extract_questions_and_alternatives.py
```

Expected successful output:

```text
Wrote 800 rows to data/processed/questions_raw.csv
Wrote validation report to data/processed/validation_report.csv
All exams validated.
```

Run the answer-key parser:

```bash
pymupdf-venv/bin/python extract_answers.py
```

Expected successful output:

```text
Wrote 800 rows to data/processed/answers_raw.csv
Wrote validation report to data/processed/answers_validation_report.csv
All answer keys validated.
```

Run the merge step:

```bash
pymupdf-venv/bin/python merge_questions_and_answers.py
```

Expected successful output:

```text
Wrote 788 rows to data/processed/questions_and_answers_raw.csv
Wrote validation report to data/processed/questions_and_answers_validation_report.csv
Merged questions and answers validated.
```

## Outputs

`data/processed/questions_raw.csv` contains one row per extracted question.

Columns:

- `exam`
- `question_number`
- `question_text`
- `alt_a`
- `alt_b`
- `alt_c`
- `alt_d`
- `raw_block`
- `source_pdf`
- `parse_ok`

`data/processed/validation_report.csv` contains one row per exam.

Columns:

- `exam`
- `question_count`
- `missing_questions`
- `duplicate_questions`
- `alternatives_ok_count`
- `parse_ok`
- `notes`

`data/processed/answers_raw.csv` contains one row per extracted TIPO 1 answer.

Columns:

- `exam`
- `question_number`
- `answer`
- `is_annulled`
- `source_pdf`
- `prova_pdf`
- `parse_ok`

`answer` is one of `A`, `B`, `C`, `D`, or `*`. Annulled questions keep the literal `*` and have `is_annulled=True`.

`data/processed/answers_validation_report.csv` contains one row per answer-key PDF.

Columns:

- `exam`
- `answer_count`
- `missing_questions`
- `duplicate_questions`
- `annulled_count`
- `prova_type_ok`
- `parse_ok`
- `notes`

`data/processed/questions_and_answers_raw.csv` contains one row per non-annulled question with its correct answer.

Columns:

- `exam`
- `question_number`
- `question_text`
- `alt_a`
- `alt_b`
- `alt_c`
- `alt_d`
- `answer`
- `question_source_pdf`
- `answer_source_pdf`
- `question_parse_ok`
- `answer_parse_ok`

Annulled questions are excluded from this combined output.

`data/processed/questions_and_answers_validation_report.csv` contains one row for the merge run.

Columns:

- `question_rows`
- `answer_rows`
- `merged_rows`
- `annulled_skipped`
- `missing_answers`
- `missing_questions`
- `duplicate_question_keys`
- `duplicate_answer_keys`
- `parse_ok`
- `notes`

## Validation Rules

For each prova PDF, question validation requires:

- exactly 80 question rows
- question numbers 1 through 80
- no duplicate question numbers
- no missing question numbers
- alternatives A, B, C, and D for every question
- `parse_ok=True` for every parsed question block

If validation fails, the full extracted text for that exam is written to `data/debug/` so the split/parsing issue can be inspected without rerunning PDF extraction manually.

For each answer-key PDF, answer validation requires:

- exactly one prova PDF and one gabarito PDF in the exam folder
- the prova PDF is confirmed as TIPO 1 / BRANCA
- exactly 80 answer rows
- question numbers 1 through 80
- no duplicate question numbers
- no missing question numbers
- every answer is `A`, `B`, `C`, `D`, or `*`
- `is_annulled=True` exactly when `answer == "*"`

If answer validation fails, the full extracted answer-key text is written to `data/debug/exam_<number>_answers_extracted_text.txt`.

For the merge step, validation requires:

- `questions_raw.csv` exists
- `answers_raw.csv` exists
- both inputs have exactly one row per `exam` and `question_number`
- every question has a matching answer
- every answer has a matching question
- annulled answers are skipped
- no merged row has `answer == "*"`
- all input rows have `parse_ok=True`

## Future Work

Planned later steps:

1. Add subject and topic labels.
2. Detect cited laws and articles.
3. Build frequency and trend charts.

Example future analyses:

- Which legal subjects appear most often?
- Which topics repeat across exams?
- Which laws and articles are cited most?
- Which subjects are stable, rising, or falling?
- Which question styles are common?

## Reference

- https://examedeordem.oab.org.br/EditaisProvas?NumeroExame=0
