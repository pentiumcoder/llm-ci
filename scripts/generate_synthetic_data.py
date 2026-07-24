"""Generate 100 golden dataset cases via OpenAI gpt-4o in three modes."""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cost import calculate_cost_aud
from src.models import GenerationMetadata, GoldenCase, GoldenDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"

CATEGORIES = ["billing", "technical", "account", "general"]
DIFFICULTIES = ["easy", "medium", "hard", "edge"]

REQUIRED_CASE_FIELDS = {
    "input",
    "expected_category",
    "expected_summary_keywords",
    "difficulty",
    "tags",
    "notes",
}

ADVERSARIAL_PAIRS: list[tuple[str, str]] = [
    ("billing", "technical"),
    ("technical", "account"),
    ("account", "billing"),
    ("general", "billing"),
    ("technical", "general"),
]

MULTILINGUAL_DISTRIBUTION: dict[str, int] = {
    "billing": 3,
    "technical": 3,
    "account": 2,
    "general": 2,
}

GENERATOR_MODEL = "gpt-4o"
GENERATION_TEMPERATURE = 0.9
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0

STANDARD_PROMPT = """\
You are generating a diverse, realistic golden dataset for evaluating a customer support email classifier.

Generate {count} customer support emails for the category: {category}
Difficulty level: {difficulty}

Difficulty definitions:
- easy: Clear, unambiguous emails where the category is obvious
- medium: Realistic emails with some ambiguity or mixed content
- hard: Emails where the correct category requires inference; could plausibly be another category
- edge: Deliberately tricky — sarcastic, very short, typos, mixed languages, no clear issue, or emotionally charged

For each email, respond ONLY with a JSON array. Each object must have:
- "input": the raw email text (realistic, varied length, no template feel)
- "expected_category": "{category}"
- "expected_summary_keywords": array of 2-4 keywords a good summary should contain
- "difficulty": "{difficulty}"
- "tags": array of relevant tags from ["ambiguous", "sarcastic", "multilingual", "short", "typo", "emotional", "vague", "multi-issue"]
- "notes": one sentence explaining what makes this case useful for evaluation

Rules:
- Vary email length: some 1–2 sentences, some 3–4 paragraphs
- Vary formality: casual, professional, frustrated, polite
- Do NOT use placeholder names like [Name] — invent realistic names
- Do NOT start every email with "I am writing to"
- Make 20% of emails contain at least one typo or grammar error
- For edge cases: at least one email should be entirely in a language other than English

No markdown. No preamble. Output only the JSON array."""

ADVERSARIAL_PROMPT = """\
You are generating adversarial test cases for a customer support email classifier.

An adversarial case is an email that a poorly-tuned classifier will likely misclassify, but which has a clear correct label when read carefully.

Generate {count} adversarial emails. Each email should be designed to trick the classifier into predicting {wrong_category}, but the correct label is {correct_category}.

For each email, respond ONLY with a JSON array. Each object must have:
- "input": the raw email text
- "expected_category": "{correct_category}"
- "adversarial_target": "{wrong_category}" (the category it is designed to fool the classifier into predicting)
- "expected_summary_keywords": array of 2-4 keywords a good summary should contain
- "difficulty": "edge"
- "tags": ["adversarial"]
- "notes": one sentence explaining the specific linguistic trick used (e.g., "uses billing vocabulary while describing a login issue")

Rules:
- The trick must be linguistic, not just ambiguous — there should be a clear correct answer
- Vary the technique: vocabulary mismatch, topic switching mid-email, misleading subject line references, category-adjacent terminology
- Do NOT invent fake product features — keep emails plausibly real

No markdown. No preamble. Output only the JSON array."""

MULTILINGUAL_PROMPT = """\
You are generating multilingual test cases for a customer support email classifier that should work regardless of language.

Generate {count} customer support emails. Write each email in a different non-English language. Distribute across: Spanish, French, Hindi, Japanese, Portuguese, Arabic (pick the most varied set for the count requested).

The correct category for each email is {category}.

For each email, respond ONLY with a JSON array. Each object must have:
- "input": the raw email text in the target language (do NOT include a translation)
- "expected_category": "{category}"
- "expected_summary_keywords": array of 2-4 keywords IN ENGLISH that a good summary should contain
- "difficulty": "edge"
- "tags": ["multilingual", "non-english"]
- "notes": one sentence in English explaining what the email says and why this category is correct

No markdown. No preamble. Output only the JSON array."""


def call_gpt4o(client: OpenAI, prompt: str) -> tuple[str, int, int]:
    """Call gpt-4o with retries; return (text, input_tokens, output_tokens)."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GENERATION_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            return text, input_tokens, output_tokens
        except (APIStatusError, APITimeoutError, APIConnectionError, OSError) as exc:
            last_exc = exc
            is_transient = isinstance(exc, (APITimeoutError, APIConnectionError)) or (
                isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503)
            )
            if not is_transient or attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("  Retry %d/%d after %.1fs: %s", attempt + 1, MAX_RETRIES, delay, exc)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def extract_cases(raw_text: str) -> list[dict[str, object]]:
    """Parse JSON and extract a list of case dicts with required fields validated."""
    data = json.loads(raw_text)
    items: list[dict[str, object]] | None = None

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                items = value
                break

    if items is None:
        raise ValueError("Response contains no JSON array")

    validated: list[dict[str, object]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            logger.warning("Skipping non-object item at index %d", idx)
            continue
        missing = REQUIRED_CASE_FIELDS - set(item.keys())
        if missing:
            logger.warning("Skipping item at index %d: missing fields %s", idx, missing)
            continue
        validated.append(item)

    if not validated:
        raise ValueError("No valid case objects found in response")

    return validated


def validate_and_collect(
    raw_cases: list[dict[str, object]],
    case_counter: list[int],
    cases: list[GoldenCase],
) -> int:
    """Validate raw dicts against GoldenCase, append valid ones, return skip count."""
    skipped = 0
    for raw in raw_cases:
        case_counter[0] += 1
        raw["id"] = f"case_{case_counter[0]:03d}"
        try:
            validated = GoldenCase(**raw)
            cases.append(validated)
        except ValidationError as exc:
            logger.warning("Skipping case_%03d: %s", case_counter[0], exc)
            skipped += 1
    return skipped


def generate_standard_cases(
    client: OpenAI,
) -> tuple[list[GoldenCase], int, int, int]:
    """Generate 80 standard cases; return (cases, skipped, input_tokens, output_tokens)."""
    logger.info("=== MODE 1: Standard cases (80 total) ===")
    cases: list[GoldenCase] = []
    counter = [0]
    total_skipped = 0
    total_input = 0
    total_output = 0

    for category in CATEGORIES:
        for difficulty in DIFFICULTIES:
            prompt = STANDARD_PROMPT.format(count=5, category=category, difficulty=difficulty)
            try:
                raw_text, inp, out = call_gpt4o(client, prompt)
                total_input += inp
                total_output += out
                raw_cases = extract_cases(raw_text)
                skipped = validate_and_collect(raw_cases, counter, cases)
                total_skipped += skipped
                logger.info(
                    "  %s/%s: got %d cases (skipped %d)",
                    category,
                    difficulty,
                    len(raw_cases) - skipped,
                    skipped,
                )
            except Exception:
                logger.exception("  %s/%s: FAILED", category, difficulty)
                total_skipped += 5

    logger.info("Standard cases generated: %d (skipped %d)", len(cases), total_skipped)
    return cases, total_skipped, total_input, total_output


def generate_adversarial_cases(
    client: OpenAI, start_counter: int
) -> tuple[list[GoldenCase], int, int, int]:
    """Generate 10 adversarial cases; return (cases, skipped, input_tokens, output_tokens)."""
    logger.info("=== MODE 2: Adversarial cases (10 total) ===")
    cases: list[GoldenCase] = []
    counter = [start_counter]
    total_skipped = 0
    total_input = 0
    total_output = 0

    for wrong_cat, correct_cat in ADVERSARIAL_PAIRS:
        prompt = ADVERSARIAL_PROMPT.format(
            count=2, wrong_category=wrong_cat, correct_category=correct_cat
        )
        try:
            raw_text, inp, out = call_gpt4o(client, prompt)
            total_input += inp
            total_output += out
            raw_cases = extract_cases(raw_text)
            skipped = validate_and_collect(raw_cases, counter, cases)
            total_skipped += skipped
            logger.info(
                "  %s->%s: got %d cases (skipped %d)",
                correct_cat,
                wrong_cat,
                len(raw_cases) - skipped,
                skipped,
            )
        except Exception:
            logger.exception("  %s->%s: FAILED", correct_cat, wrong_cat)
            total_skipped += 2

    logger.info("Adversarial cases generated: %d (skipped %d)", len(cases), total_skipped)
    return cases, total_skipped, total_input, total_output


def generate_multilingual_cases(
    client: OpenAI, start_counter: int
) -> tuple[list[GoldenCase], int, int, int]:
    """Generate 10 multilingual cases; return (cases, skipped, input_tokens, output_tokens)."""
    logger.info("=== MODE 3: Multilingual cases (10 total) ===")
    cases: list[GoldenCase] = []
    counter = [start_counter]
    total_skipped = 0
    total_input = 0
    total_output = 0

    for category, count in MULTILINGUAL_DISTRIBUTION.items():
        prompt = MULTILINGUAL_PROMPT.format(count=count, category=category)
        try:
            raw_text, inp, out = call_gpt4o(client, prompt)
            total_input += inp
            total_output += out
            raw_cases = extract_cases(raw_text)
            skipped = validate_and_collect(raw_cases, counter, cases)
            total_skipped += skipped
            logger.info(
                "  %s: got %d cases (skipped %d)",
                category,
                len(raw_cases) - skipped,
                skipped,
            )
        except Exception:
            logger.exception("  %s: FAILED", category)
            total_skipped += count

    logger.info("Multilingual cases generated: %d (skipped %d)", len(cases), total_skipped)
    return cases, total_skipped, total_input, total_output


def print_breakdown(dataset: GoldenDataset) -> None:
    """Print breakdown table of counts per category, difficulty, and tag."""
    from collections import Counter

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
    """Generate all 100 golden dataset cases and write to JSON."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    client = OpenAI()

    all_cases: list[GoldenCase] = []
    total_api_calls = 0
    cum_input = 0
    cum_output = 0

    standard_cases, s_skip, s_in, s_out = generate_standard_cases(client)
    all_cases.extend(standard_cases)
    total_api_calls += 16
    cum_input += s_in
    cum_output += s_out

    adversarial_cases, a_skip, a_in, a_out = generate_adversarial_cases(
        client, start_counter=len(all_cases)
    )
    all_cases.extend(adversarial_cases)
    total_api_calls += 5
    cum_input += a_in
    cum_output += a_out

    multilingual_cases, m_skip, m_in, m_out = generate_multilingual_cases(
        client, start_counter=len(all_cases)
    )
    all_cases.extend(multilingual_cases)
    total_api_calls += 4
    cum_input += m_in
    cum_output += m_out

    total_skipped = s_skip + a_skip + m_skip
    cost_aud = calculate_cost_aud("openai", GENERATOR_MODEL, cum_input, cum_output)

    metadata = GenerationMetadata(
        generator_model=GENERATOR_MODEL,
        prompt_version="1.0.0",
        temperature=GENERATION_TEMPERATURE,
        generation_timestamp=datetime.now(timezone.utc),
        total_api_calls=total_api_calls,
        total_input_tokens=cum_input,
        total_output_tokens=cum_output,
        estimated_cost_aud=cost_aud,
        generation_mode_counts={
            "standard": len(standard_cases),
            "adversarial": len(adversarial_cases),
            "multilingual": len(multilingual_cases),
        },
    )

    dataset = GoldenDataset(
        version="1.0.0",
        created_at=datetime.now(timezone.utc),
        cases=all_cases,
        metadata=metadata,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Dataset written to %s", OUTPUT_PATH)
    logger.info("Total cases: %d (skipped: %d)", len(all_cases), total_skipped)

    print_breakdown(dataset)

    print("\n=== Generation Summary ===")
    print(f"API calls: {total_api_calls}")
    print(f"Input tokens: {cum_input}")
    print(f"Output tokens: {cum_output}")
    print(f"Estimated cost: AUD ${cost_aud:.4f}")


if __name__ == "__main__":
    main()
