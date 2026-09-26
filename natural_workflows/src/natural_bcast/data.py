from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import string
import tarfile
import urllib.request
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


HOTPot_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    "1908d6afbbead072334abe2965f91bd2709910ab/"
    "distractor/validation-00000-of-00001.parquet"
)
TATQA_URL = (
    "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/"
    "870accc41953dcde885aabeb963d94aabdc0fbc3/"
    "dataset_raw/tatqa_dataset_dev.json"
)
HEARSAY_URL = (
    "https://huggingface.co/datasets/nguha/legalbench/resolve/"
    "daec8237410aa23e3faf4bc41ad8b3a7e1696826/data/hearsay/test.tsv"
)
HEARSAY_PROMPT_URL = (
    "https://raw.githubusercontent.com/HazyResearch/legalbench/"
    "b46bf4ffae90524b2b72aaa30e7745fe9db64481/tasks/hearsay/base_prompt.txt"
)
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DROP_URL = "https://ai2-public-datasets.s3.amazonaws.com/drop/drop_dataset.zip"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "3101c7d5072418e28b9008a6636bde82a006892c/"
    "grade_school_math/data/test.jsonl"
)
MUSIQUE_URL = (
    "https://huggingface.co/datasets/bdsaglam/musique/resolve/"
    "22873a405dd809893b22ada0b499299fb612d2df/"
    "musique_ans_v1.0_dev.jsonl"
)
TABFACT_URL = (
    "https://github.com/wenhuchen/Table-Fact-Checking/archive/"
    "948b5560e2f7f8c9139bd91c7f093346a2bb56a8.zip"
)


@dataclass(frozen=True)
class SourceBlock:
    source_id: str
    text: str


@dataclass(frozen=True)
class TaskCase:
    case_id: str
    task: str
    dataset: str
    split: str
    group_id: str
    question: str
    sources: tuple[SourceBlock, ...]
    gold_answers: tuple[str, ...]
    gold_scale: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [asdict(item) for item in self.sources]
        payload["gold_answers"] = list(self.gold_answers)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskCase":
        return cls(
            case_id=str(payload["case_id"]),
            task=str(payload["task"]),
            dataset=str(payload["dataset"]),
            split=str(payload["split"]),
            group_id=str(payload["group_id"]),
            question=str(payload["question"]),
            sources=tuple(SourceBlock(**item) for item in payload["sources"]),
            gold_answers=tuple(str(item) for item in payload["gold_answers"]),
            gold_scale=str(payload.get("gold_scale", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def visible_payload(self, *, include_source_text: bool = True) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task": self.task,
            "question": self.question,
            "sources": [
                {
                    "source_id": source.source_id,
                    **({"text": source.text} if include_source_text else {}),
                }
                for source in self.sources
            ],
            "answer_format": answer_format(self.task),
        }


def answer_format(task: str) -> str:
    formats = {
        "hotpotqa": "Return the shortest answer span that resolves the multi-hop question.",
        "tatqa": (
            "Return the answer and scale. Allowed scales are empty, thousand, million, "
            "billion, and percent."
        ),
        "legal_hearsay": "Return exactly Yes if the evidence is hearsay; otherwise return No.",
        "scifact": "Return exactly SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO.",
        "drop": "Return the shortest answer span, number, date, or list required by the passage.",
        "gsm8k": "Return only the final numeric answer; do not include units or prose.",
        "musique": "Return the shortest answer span that resolves the complete multi-hop chain.",
        "tabfact": "Return exactly ENTAILED or REFUTED.",
    }
    return formats[task]


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "natural-bcast/1.0"})
    with urllib.request.urlopen(request, timeout=240) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_raw_data(raw_dir: Path, legacy_raw_dir: Path | None = None) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "hotpotqa": raw_dir / "hotpotqa_validation.parquet",
        "tatqa": raw_dir / "tatqa_dev.json",
        "legal_hearsay": raw_dir / "hearsay_test.tsv",
        "legal_prompt": raw_dir / "hearsay_base_prompt.txt",
        "scifact_archive": raw_dir / "scifact_data.tar.gz",
        "drop_archive": raw_dir / "drop_dataset.zip",
        "gsm8k": raw_dir / "gsm8k_test.jsonl",
        "musique": raw_dir / "musique_ans_dev.jsonl",
        "tabfact_archive": raw_dir / "tabfact.zip",
    }
    legacy_names = {
        "hotpotqa": "hotpotqa_validation.parquet",
        "tatqa": "tatqa_dev.json",
        "legal_hearsay": "hearsay_test.tsv",
        "legal_prompt": "hearsay_base_prompt.txt",
    }
    if legacy_raw_dir:
        for key, name in legacy_names.items():
            source = legacy_raw_dir / name
            if source.exists() and not files[key].exists():
                shutil.copy2(source, files[key])
    for key, url in (
        ("hotpotqa", HOTPot_URL),
        ("tatqa", TATQA_URL),
        ("legal_hearsay", HEARSAY_URL),
        ("legal_prompt", HEARSAY_PROMPT_URL),
        ("scifact_archive", SCIFACT_URL),
        ("drop_archive", DROP_URL),
        ("gsm8k", GSM8K_URL),
        ("musique", MUSIQUE_URL),
        ("tabfact_archive", TABFACT_URL),
    ):
        _download(url, files[key])

    scifact_dir = raw_dir / "scifact"
    if not (scifact_dir / "data" / "claims_dev.jsonl").exists():
        scifact_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(files["scifact_archive"], "r:gz") as archive:
            archive.extractall(scifact_dir, filter="data")

    drop_dir = raw_dir / "drop"
    if not (drop_dir / "drop_dataset" / "drop_dataset_dev.json").exists():
        drop_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(files["drop_archive"]) as archive:
            archive.extractall(drop_dir)

    tabfact_dir = raw_dir / "tabfact"
    tabfact_root = tabfact_dir / "Table-Fact-Checking-948b5560e2f7f8c9139bd91c7f093346a2bb56a8"
    if not (tabfact_root / "tokenized_data" / "val_examples.json").exists():
        tabfact_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(files["tabfact_archive"]) as archive:
            archive.extractall(tabfact_dir)

    manifest = {
        "files": {
            key: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for key, path in files.items()
        }
    }
    return manifest


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _hash(seed: int, *parts: str) -> str:
    return hashlib.sha256("|".join((str(seed), *parts)).encode("utf-8")).hexdigest()


def _table(rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    cleaned = [[_clean(cell).replace("|", "\\|") for cell in row] for row in rows]
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    return "\n".join(
        [
            "| " + " | ".join(padded[0]) + " |",
            "| " + " | ".join(["---"] * width) + " |",
            *("| " + " | ".join(row) + " |" for row in padded[1:]),
        ]
    )


def load_hotpotqa(path: Path, max_chars: int) -> list[TaskCase]:
    import pyarrow.parquet as pq

    cases: list[TaskCase] = []
    for row in pq.read_table(path).to_pylist():
        context = row["context"]
        pairs = zip(context.get("title", []), context.get("sentences", []))
        sources = tuple(
            SourceBlock(f"wiki:{title}", f"{title}: " + " ".join(_clean(s) for s in sentences))
            for title, sentences in pairs
        )
        total = sum(len(source.text) for source in sources)
        if not (500 <= total <= max_chars):
            continue
        cases.append(
            TaskCase(
                case_id=f"hotpot:{row['id']}",
                task="hotpotqa",
                dataset="HotpotQA-distractor",
                split="validation",
                group_id=f"hotpot:{row['id']}",
                question=_clean(row["question"]),
                sources=sources,
                gold_answers=(_clean(row["answer"]),),
                gold_scale="",
                metadata={
                    "level": str(row.get("level", "")),
                    "question_type": str(row.get("type", "")),
                    "supporting_facts": row.get("supporting_facts", {}),
                },
            )
        )
    return cases


def load_tatqa(path: Path, max_chars: int) -> list[TaskCase]:
    cases: list[TaskCase] = []
    for document in json.loads(path.read_text(encoding="utf-8")):
        table = document["table"]
        sources = [SourceBlock(f"table:{table['uid']}", _table(table["table"]))]
        sources.extend(
            SourceBlock(f"paragraph:{item['uid']}", _clean(item["text"]))
            for item in sorted(document.get("paragraphs", []), key=lambda value: value.get("order", 0))
        )
        total = sum(len(source.text) for source in sources)
        if not (300 <= total <= max_chars):
            continue
        for question in document.get("questions", []):
            if question.get("answer_type") not in {"arithmetic", "span", "count"}:
                continue
            answer = question.get("answer", [])
            answer = answer if isinstance(answer, list) else [answer]
            gold = tuple(_clean(item) for item in answer if _clean(item))
            if not gold:
                continue
            cases.append(
                TaskCase(
                    case_id=f"tatqa:{question['uid']}",
                    task="tatqa",
                    dataset="TAT-QA",
                    split="dev",
                    group_id=f"tatqa-doc:{table['uid']}",
                    question=_clean(question["question"]),
                    sources=tuple(sources),
                    gold_answers=gold,
                    gold_scale=_clean(question.get("scale", "")).lower(),
                    metadata={
                        "answer_type": question.get("answer_type", ""),
                        "answer_from": question.get("answer_from", ""),
                        "derivation": _clean(question.get("derivation", "")),
                        "requires_comparison": bool(question.get("req_comparison", False)),
                    },
                )
            )
    return cases


def load_hearsay(path: Path, prompt_path: Path) -> list[TaskCase]:
    official = prompt_path.read_text(encoding="utf-8").replace("{{text}}", "").strip()
    rule = (
        "Hearsay is an out-of-court statement introduced to prove the truth of the matter asserted. "
        "Determine whether there was a statement, whether it was made outside the trial or hearing, "
        "and whether it is offered for its truth. Statements used for notice, effect on the listener, "
        "speaker identity, or another non-truth purpose are not hearsay. Ignore exceptions.\n\n"
        + official
    )
    cases = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            index = str(row["index"])
            cases.append(
                TaskCase(
                    case_id=f"hearsay:{index}",
                    task="legal_hearsay",
                    dataset="LegalBench-hearsay",
                    split="test",
                    group_id=f"hearsay:{index}",
                    question="Under the supplied rule, is the described evidence hearsay?",
                    sources=(
                        SourceBlock("rule:hearsay", rule),
                        SourceBlock(f"fact-pattern:{index}", _clean(row["text"])),
                    ),
                    gold_answers=(_clean(row["answer"]),),
                    gold_scale="",
                    metadata={"slice": _clean(row.get("slice", ""))},
                )
            )
    return cases


def load_scifact(root: Path, max_chars: int, seed: int) -> list[TaskCase]:
    corpus_path = root / "data" / "corpus.jsonl"
    claims_path = root / "data" / "claims_dev.jsonl"
    corpus = {
        int(item["doc_id"]): item
        for item in (json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines())
    }
    doc_ids = sorted(corpus)
    cases = []
    for claim in (json.loads(line) for line in claims_path.read_text(encoding="utf-8").splitlines()):
        evidence = claim.get("evidence", {})
        labels = {
            item["label"]
            for entries in evidence.values()
            for item in entries
            if item.get("label")
        }
        label = "NOT_ENOUGH_INFO" if not labels else sorted(labels)[0]
        cited = [int(item) for item in claim.get("cited_doc_ids", []) if int(item) in corpus]
        distractors = [
            doc_id
            for doc_id in sorted(doc_ids, key=lambda value: _hash(seed, str(claim["id"]), str(value)))
            if doc_id not in cited
        ][:2]
        selected = cited + distractors
        sources = []
        for doc_id in selected:
            doc = corpus[doc_id]
            text = f"{doc.get('title', '')}: " + " ".join(_clean(item) for item in doc.get("abstract", []))
            sources.append(SourceBlock(f"paper:{doc_id}", text))
        total = sum(len(source.text) for source in sources)
        if not sources or total > max_chars:
            continue
        cases.append(
            TaskCase(
                case_id=f"scifact:{claim['id']}",
                task="scifact",
                dataset="SciFact",
                split="dev",
                group_id=f"scifact:{claim['id']}",
                question=f"Classify this scientific claim: {_clean(claim['claim'])}",
                sources=tuple(sources),
                gold_answers=(label,),
                gold_scale="",
                metadata={"evidence_doc_ids": sorted(evidence), "cited_doc_ids": cited},
            )
        )
    return cases


def _drop_answer(annotation: dict[str, Any]) -> tuple[str, ...]:
    spans = tuple(_clean(item) for item in annotation.get("spans", []) if _clean(item))
    if spans:
        return spans
    number = _clean(annotation.get("number", ""))
    if number:
        return (number,)
    date = annotation.get("date", {})
    date_parts = [_clean(date.get(key, "")) for key in ("month", "day", "year")]
    joined = " ".join(part for part in date_parts if part)
    return (joined,) if joined else tuple()


def load_drop(root: Path, max_chars: int) -> list[TaskCase]:
    path = root / "drop_dataset" / "drop_dataset_dev.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for passage_id, passage in payload.items():
        source_text = _clean(passage["passage"])
        if not (400 <= len(source_text) <= max_chars):
            continue
        for item in passage.get("qa_pairs", []):
            answers = _drop_answer(item.get("answer", {}))
            if not answers:
                continue
            annotation = item.get("answer", {})
            kind = "span" if annotation.get("spans") else "number" if annotation.get("number") else "date"
            cases.append(
                TaskCase(
                    case_id=f"drop:{item['query_id']}",
                    task="drop",
                    dataset="DROP",
                    split="dev",
                    group_id=f"drop-passage:{passage_id}",
                    question=_clean(item["question"]),
                    sources=(SourceBlock(f"passage:{passage_id}", source_text),),
                    gold_answers=answers,
                    gold_scale="",
                    metadata={"answer_kind": kind},
                )
            )
    return cases


def load_gsm8k(path: Path) -> list[TaskCase]:
    cases = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        item = json.loads(line)
        final = item["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        question = _clean(item["question"])
        cases.append(
            TaskCase(
                case_id=f"gsm8k:{index}",
                task="gsm8k",
                dataset="GSM8K",
                split="test",
                group_id=f"gsm8k:{index}",
                question=question,
                sources=(SourceBlock(f"problem:{index}", question),),
                gold_answers=(final,),
                gold_scale="",
                metadata={"gold_rationale": item["answer"].rsplit("####", 1)[0].strip()},
            )
        )
    return cases


def load_musique(path: Path, max_chars: int) -> list[TaskCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("answerable", True):
            continue
        sources = tuple(
            SourceBlock(
                f"paragraph:{paragraph['idx']}:{_clean(paragraph.get('title', ''))}",
                f"{_clean(paragraph.get('title', ''))}: "
                f"{_clean(paragraph.get('paragraph_text', ''))}",
            )
            for paragraph in item.get("paragraphs", [])
            if _clean(paragraph.get("paragraph_text", ""))
        )
        total = sum(len(source.text) for source in sources)
        if not sources or not (500 <= total <= max_chars):
            continue
        aliases = tuple(_clean(value) for value in item.get("answer_aliases", []) if _clean(value))
        cases.append(
            TaskCase(
                case_id=f"musique:{item['id']}",
                task="musique",
                dataset="MuSiQue-Ans",
                split="dev",
                group_id=f"musique:{item['id']}",
                question=_clean(item["question"]),
                sources=sources,
                gold_answers=(_clean(item["answer"]), *aliases),
                gold_scale="",
                metadata={
                    "hop_count": len(item.get("question_decomposition", [])),
                    "question_decomposition": item.get("question_decomposition", []),
                    "supporting_paragraph_indices": [
                        int(paragraph["idx"])
                        for paragraph in item.get("paragraphs", [])
                        if paragraph.get("is_supporting")
                    ],
                },
            )
        )
    return cases


def load_tabfact(root: Path, max_chars: int) -> list[TaskCase]:
    statements_path = root / "tokenized_data" / "val_examples.json"
    tables_root = root / "data" / "all_csv"
    examples = json.loads(statements_path.read_text(encoding="utf-8"))
    cases = []
    for table_id, value in examples.items():
        statements, labels, caption = value
        table_path = tables_root / table_id
        raw_table = table_path.read_text(encoding="utf-8")
        first_line = raw_table.splitlines()[0] if raw_table.splitlines() else ""
        delimiter = "#" if first_line.count("#") >= first_line.count(",") else ","
        rows = list(csv.reader(raw_table.splitlines(), delimiter=delimiter))
        rendered = f"Caption: {_clean(caption)}\n{_table(rows)}"
        if not (100 <= len(rendered) <= max_chars):
            continue
        for statement_index, (statement, label) in enumerate(zip(statements, labels)):
            label_name = "ENTAILED" if int(label) == 1 else "REFUTED"
            cases.append(
                TaskCase(
                    case_id=f"tabfact:{table_id}:{statement_index}",
                    task="tabfact",
                    dataset="TabFact",
                    split="validation",
                    group_id=f"tabfact-table:{table_id}",
                    question=(
                        "Determine whether this statement is entailed or refuted by the supplied table: "
                        f"{_clean(statement)}"
                    ),
                    sources=(SourceBlock(f"table:{table_id}", rendered),),
                    gold_answers=(label_name,),
                    gold_scale="",
                    metadata={
                        "table_id": table_id,
                        "statement_index": statement_index,
                        "statement_tokens": len(_clean(statement).split()),
                    },
                )
            )
    return cases


def load_all(raw_dir: Path, *, max_chars: int, seed: int) -> dict[str, list[TaskCase]]:
    tabfact_root = (
        raw_dir
        / "tabfact"
        / "Table-Fact-Checking-948b5560e2f7f8c9139bd91c7f093346a2bb56a8"
    )
    return {
        "hotpotqa": load_hotpotqa(raw_dir / "hotpotqa_validation.parquet", max_chars),
        "tatqa": load_tatqa(raw_dir / "tatqa_dev.json", max_chars),
        "legal_hearsay": load_hearsay(raw_dir / "hearsay_test.tsv", raw_dir / "hearsay_base_prompt.txt"),
        "scifact": load_scifact(raw_dir / "scifact", max_chars, seed),
        "drop": load_drop(raw_dir / "drop", max_chars),
        "gsm8k": load_gsm8k(raw_dir / "gsm8k_test.jsonl"),
        "musique": load_musique(raw_dir / "musique_ans_dev.jsonl", max_chars),
        "tabfact": load_tabfact(tabfact_root, max_chars),
    }


def _one_per_group(cases: Iterable[TaskCase], seed: int) -> list[TaskCase]:
    grouped: dict[str, list[TaskCase]] = defaultdict(list)
    for case in cases:
        grouped[case.group_id].append(case)
    return [min(items, key=lambda item: _hash(seed, item.case_id)) for items in grouped.values()]


def _stratified_order(
    cases: Iterable[TaskCase], *, seed: int, stratum: Callable[[TaskCase], str]
) -> list[TaskCase]:
    groups: dict[str, deque[TaskCase]] = {}
    raw: dict[str, list[TaskCase]] = defaultdict(list)
    for case in cases:
        raw[stratum(case)].append(case)
    for key, values in raw.items():
        groups[key] = deque(sorted(values, key=lambda item: _hash(seed, key, item.case_id)))
    ordered = []
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                ordered.append(groups[key].popleft())
    return ordered


def freeze_cases(
    all_cases: dict[str, list[TaskCase]], counts: dict[str, int], *, seed: int
) -> dict[str, list[TaskCase]]:
    strata: dict[str, Callable[[TaskCase], str]] = {
        "hotpotqa": lambda case: f"{case.metadata.get('level')}:{case.metadata.get('question_type')}",
        "tatqa": lambda case: f"{case.metadata.get('answer_type')}:{case.metadata.get('answer_from')}",
        "legal_hearsay": lambda case: f"{case.metadata.get('slice')}:{normalize_answer(case.gold_answers[0])}",
        "scifact": lambda case: case.gold_answers[0],
        "drop": lambda case: str(case.metadata.get("answer_kind")),
        "gsm8k": lambda case: magnitude_bucket(case.gold_answers[0]),
        "musique": lambda case: f"{case.metadata.get('hop_count', 0)}hop",
        "tabfact": lambda case: (
            f"{normalize_answer(case.gold_answers[0])}:"
            f"{'long' if int(case.metadata.get('statement_tokens', 0)) >= 18 else 'short'}"
        ),
    }
    frozen = {}
    for task, count in counts.items():
        unique = _one_per_group(all_cases[task], seed)
        if task == "hotpotqa":
            unique = [case for case in unique if case.metadata.get("level") in {"medium", "hard"}]
        if task == "tatqa":
            unique = sorted(
                unique,
                key=lambda case: (
                    case.metadata.get("answer_type") != "arithmetic",
                    case.metadata.get("answer_from") != "table-text",
                    _hash(seed, case.case_id),
                ),
            )
        ordered = _stratified_order(unique, seed=seed, stratum=strata[task])
        if len(ordered) < count:
            raise ValueError(f"{task}: requested {count}, only {len(ordered)} available")
        frozen[task] = ordered[:count]
    return frozen


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def magnitude_bucket(value: str) -> str:
    try:
        number = abs(float(value.replace(",", "")))
    except ValueError:
        return "non_numeric"
    if number < 10:
        return "lt10"
    if number < 100:
        return "lt100"
    if number < 1000:
        return "lt1000"
    return "ge1000"


def normalize_answer(value: Any) -> str:
    text = str(value).lower().strip().replace("%", " percent ")
    text = text.replace("–", "-").replace("—", "-")
    text = "".join(char if char not in string.punctuation else " " for char in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _number_forms(value: Any) -> set[str]:
    raw = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
    forms = {normalize_answer(value)}
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return forms
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    forms.update({normalized, normalize_answer(normalized)})
    return {item for item in forms if item}


def token_f1(prediction: Any, golds: Sequence[str]) -> float:
    predicted = normalize_answer(prediction).split()
    if not predicted:
        return 0.0
    scores = []
    for gold in golds:
        expected = normalize_answer(gold).split()
        common = Counter(predicted) & Counter(expected)
        overlap = sum(common.values())
        if not expected or overlap == 0:
            scores.append(0.0)
            continue
        precision = overlap / len(predicted)
        recall = overlap / len(expected)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


def exact_match(prediction: Any, golds: Sequence[str]) -> float:
    predicted = _number_forms(prediction)
    return float(any(predicted & _number_forms(gold) for gold in golds))


def canonical_label(task: str, value: Any) -> str:
    normalized = normalize_answer(value)
    if task == "legal_hearsay":
        if re.search(r"\byes\b", normalized):
            return "yes"
        if re.search(r"\bno\b", normalized):
            return "no"
    if task == "scifact":
        if "not enough" in normalized or normalized in {"nei", "unknown"}:
            return "not_enough_info"
        if "contradict" in normalized or "refute" in normalized:
            return "contradict"
        if "support" in normalized:
            return "support"
    if task == "tabfact":
        if "entail" in normalized or normalized in {"1", "true"}:
            return "entailed"
        if "refut" in normalized or normalized in {"0", "false"}:
            return "refuted"
    return normalized


def score(case: TaskCase, answer: Any, scale: Any = "") -> dict[str, float | str]:
    answer_text = str(answer).strip()
    scale_text = str(scale).strip().lower()
    if case.task in {"legal_hearsay", "scifact", "tabfact"}:
        prediction = canonical_label(case.task, answer_text)
        expected = canonical_label(case.task, case.gold_answers[0])
        correct = float(prediction == expected)
        return {"prediction": prediction, "scale": "", "correct": correct, "f1": correct, "loss": 1 - correct}
    if case.task == "tatqa":
        gold_answers = tuple(case.gold_answers)
        gold_scale = case.gold_scale
        if not gold_scale and any("%" in value for value in gold_answers):
            gold_scale = "percent"
            gold_answers = tuple(value.replace("%", "") for value in gold_answers)
        if not scale_text and "%" in answer_text:
            scale_text = "percent"
            answer_text = answer_text.replace("%", "")
        em = exact_match(answer_text, gold_answers)
        f1 = token_f1(answer_text, gold_answers)
        scale_correct = float(scale_text == gold_scale)
        task_score = f1 * scale_correct
    else:
        em = exact_match(answer_text, case.gold_answers)
        f1 = token_f1(answer_text, case.gold_answers)
    if case.task in {"hotpotqa", "drop", "musique"}:
        scale_correct = 1.0
        task_score = f1
    elif case.task != "tatqa":
        scale_correct = 1.0
        task_score = em
    return {
        "prediction": answer_text,
        "scale": scale_text,
        "em": em,
        "f1": f1,
        "scale_correct": scale_correct,
        "correct": em * scale_correct,
        "task_score": task_score,
        "loss": 1.0 - task_score,
    }


def split_cases(
    cases: Sequence[TaskCase], *, seed: int, discovery_fraction: float
) -> tuple[list[TaskCase], list[TaskCase]]:
    ordered = sorted(cases, key=lambda case: _hash(seed, "split", case.case_id))
    boundary = int(math.floor(len(ordered) * discovery_fraction))
    return ordered[:boundary], ordered[boundary:]
