"""Dataset loader and filter functions for the golden evaluation dataset."""

from __future__ import annotations

import json
import logging

from src.models import GoldenCase, GoldenDataset

logger = logging.getLogger(__name__)


def load_dataset(path: str) -> GoldenDataset:
    """Load and validate the golden dataset from a JSON file."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    dataset = GoldenDataset(**raw)
    logger.info("Loaded dataset with %d cases from %s", len(dataset.cases), path)
    return dataset


def get_cases_by_difficulty(dataset: GoldenDataset, difficulty: str) -> list[GoldenCase]:
    """Return all cases matching the given difficulty level."""
    return [c for c in dataset.cases if c.difficulty == difficulty]


def get_cases_by_category(dataset: GoldenDataset, category: str) -> list[GoldenCase]:
    """Return all cases matching the given expected category."""
    return [c for c in dataset.cases if c.expected_category == category]


def get_cases_by_tag(dataset: GoldenDataset, tag: str) -> list[GoldenCase]:
    """Return all cases that contain the given tag."""
    return [c for c in dataset.cases if tag in c.tags]
