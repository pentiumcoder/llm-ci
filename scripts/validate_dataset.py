"""Validate the golden dataset against schema and content assertions."""

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import GoldenDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"

CATEGORIES = ["billing", "technical", "account", "general"]
DIFFICULTIES = ["easy", "medium", "hard", "edge"]

EXPECTED_CATEGORY_COUNTS: dict[str, int] = {
    "billing": 27,
    "technical": 25,
    "account": 24,
    "general": 24,
}

EXPECTED_DIFFICULTY_COUNTS: dict[str, int] = {
    "easy": 20,
    "medium": 20,
    "hard": 20,
    "edge": 40,
}

ID_PATTERN = re.compile(r"^case_(\d{3})$")


def load_dataset(path: Path) -> GoldenDataset:
    """Load and validate the golden dataset from JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GoldenDataset(**raw)


def validate(dataset: GoldenDataset) -> list[str]:
    """Run all assertions, return list of failure messages (empty = all pass)."""
    failures: list[str] = []
    cases = dataset.cases

    if len(cases) != 100:
        failures.append(f"Dataset must contain exactly 100 cases, found {len(cases)}")

    cat_counts: Counter[str] = Counter(c.expected_category for c in cases)
    for cat, expected in EXPECTED_CATEGORY_COUNTS.items():
        actual = cat_counts.get(cat, 0)
        if actual != expected:
            failures.append(f"Category '{cat}': expected {expected}, found {actual}")

    diff_counts: Counter[str] = Counter(c.difficulty for c in cases)
    for diff, expected in EXPECTED_DIFFICULTY_COUNTS.items():
        actual = diff_counts.get(diff, 0)
        if actual != expected:
            failures.append(f"Difficulty '{diff}': expected {expected}, found {actual}")

    expected_ids = {f"case_{i:03d}" for i in range(1, 101)}
    actual_ids = {c.id for c in cases}

    missing_ids = expected_ids - actual_ids
    if missing_ids:
        failures.append(f"Missing IDs: {sorted(missing_ids)[:5]}...")

    extra_ids = actual_ids - expected_ids
    if extra_ids:
        failures.append(f"Unexpected IDs: {sorted(extra_ids)[:5]}...")

    non_sequential = []
    for c in cases:
        match = ID_PATTERN.match(c.id)
        if not match:
            non_sequential.append(c.id)
            continue
        num = int(match.group(1))
        if num < 1 or num > 100:
            non_sequential.append(c.id)
    if non_sequential:
        failures.append(f"Non-sequential IDs: {non_sequential[:5]}")

    empty_inputs = [c.id for c in cases if not c.input.strip()]
    if empty_inputs:
        failures.append(f"Empty inputs: {empty_inputs}")

    adversarial_cases = [c for c in cases if "adversarial" in c.tags]
    adversarial_no_target = [
        c.id for c in adversarial_cases if c.adversarial_target is None
    ]
    if adversarial_no_target:
        failures.append(f"Adversarial cases missing adversarial_target: {adversarial_no_target}")

    return failures


def print_breakdown(dataset: GoldenDataset) -> None:
    """Print breakdown table of counts per category, difficulty, and tag."""
    cat_counts: Counter[str] = Counter()
    diff_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()

    for case in dataset.cases:
        cat_counts[case.expected_category] += 1
        diff_counts[case.difficulty] += 1
        for tag in case.tags:
            tag_counts[tag] += 1

    print("\n=== Dataset Breakdown ===")
    print(f"Total cases: {len(dataset.cases)}")

    print("\nBy category:")
    for cat in CATEGORIES:
        expected = EXPECTED_CATEGORY_COUNTS[cat]
        actual = cat_counts.get(cat, 0)
        mark = "" if actual == expected else f" (expected {expected})"
        print(f"  {cat}: {actual}{mark}")

    print("\nBy difficulty:")
    for diff in DIFFICULTIES:
        expected = EXPECTED_DIFFICULTY_COUNTS[diff]
        actual = diff_counts.get(diff, 0)
        mark = "" if actual == expected else f" (expected {expected})"
        print(f"  {diff}: {actual}{mark}")

    print("\nBy tag:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")

    if dataset.metadata:
        m = dataset.metadata
        print("\n=== Generation Metadata ===")
        print(f"Model: {m.generator_model}")
        print(f"Temperature: {m.temperature}")
        print(f"API calls: {m.total_api_calls}")
        print(f"Tokens: {m.total_input_tokens} in / {m.total_output_tokens} out")
        print(f"Estimated cost: AUD ${m.estimated_cost_aud:.4f}")
        print(f"Mode counts: {m.generation_mode_counts}")


def main() -> None:
    """Load dataset, run validations, print results."""
    if not DATASET_PATH.exists():
        logger.error("Dataset not found at %s", DATASET_PATH)
        sys.exit(1)

    dataset = load_dataset(DATASET_PATH)
    print_breakdown(dataset)

    failures = validate(dataset)

    print("\n=== Validation Results ===")
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  X {f}")
        sys.exit(1)
    else:
        print("PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
