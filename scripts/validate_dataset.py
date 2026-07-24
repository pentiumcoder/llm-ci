"""Validate the golden dataset against schema and content assertions."""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from src.models import GoldenCase, GoldenDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"

CATEGORIES = ["billing", "technical", "account", "general"]
DIFFICULTIES = ["easy", "medium", "hard", "edge"]


def load_dataset(path: Path) -> GoldenDataset:
    """Load and validate the golden dataset from JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GoldenDataset(**raw)


def validate(dataset: GoldenDataset) -> list[str]:
    """Run all assertions, return list of failure messages (empty = all pass)."""
    failures: list[str] = []
    cases = dataset.cases

    if len(cases) < 100:
        failures.append(f"Total cases {len(cases)} < 100")

    adversarial_cases = [c for c in cases if "adversarial" in c.tags]
    multilingual_cases = [c for c in cases if "multilingual" in c.tags or "non-english" in c.tags]
    standard_cases = [
        c
        for c in cases
        if "adversarial" not in c.tags
        and "multilingual" not in c.tags
        and "non-english" not in c.tags
    ]

    if len(standard_cases) < 80:
        failures.append(f"Standard cases {len(standard_cases)} < 80")

    if len(adversarial_cases) < 8:
        failures.append(f"Adversarial tag count {len(adversarial_cases)} < 8")

    if len(multilingual_cases) < 8:
        failures.append(f"Non-english/multilingual tag count {len(multilingual_cases)} < 8")

    present_categories = {c.expected_category for c in cases}
    missing = set(CATEGORIES) - present_categories
    if missing:
        failures.append(f"Missing categories: {missing}")

    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        failures.append(f"Duplicate IDs found: {len(ids)} total, {len(set(ids))} unique")

    empty_inputs = [c.id for c in cases if not c.input.strip()]
    if empty_inputs:
        failures.append(f"Empty inputs: {empty_inputs}")

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
        print(f"  {cat}: {cat_counts.get(cat, 0)}")

    print("\nBy difficulty:")
    for diff in DIFFICULTIES:
        print(f"  {diff}: {diff_counts.get(diff, 0)}")

    print("\nBy tag:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")


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
