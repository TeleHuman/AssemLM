#!/usr/bin/env python3
"""

 ``aggregate_eval_results.py``  seed
 seed  JSON

*  split  summary  per_category
* overall_by_split  per_category

 seed /ddof=0
/ddof=1 seed  seed
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


def _parse_result_spec(spec: str) -> tuple[int, Path]:
    parts = spec.split(":", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Invalid --result {spec!r}; expected SEED:/path/to/eval_results.json."
        )
    seed_text, result_path = parts
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise ValueError(f"Invalid seed {seed_text!r} in --result {spec!r}.") from exc
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed} in --result {spec!r}.")
    return seed, Path(result_path).expanduser().resolve()


def _parse_expected_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(
            f"Invalid --expected_seeds {value!r}; expected comma-separated integers."
        ) from exc
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("--expected_seeds must contain non-negative integers.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--expected_seeds contains duplicates: {seeds}")
    return seeds


def _iter_numeric_leaves(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], float]]:
    """ JSON """
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_numeric_leaves(child, path + (str(key),))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Non-finite numeric value at {'.'.join(path)}: {value!r}")
        yield path, numeric_value


def _validate_finite_numbers(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject JSON NaN/Infinity values before cross-seed aggregation."""
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_numbers(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_numbers(child, path + (str(index),))
    elif isinstance(value, float) and not math.isfinite(value):
        location = ".".join(path) or "<root>"
        raise ValueError(f"Non-finite numeric value at {location}: {value!r}")


def _metric_statistics(
    values_by_seed: dict[int, float], expected_seeds: list[int]
) -> dict[str, Any]:
    ordered_values = [values_by_seed[seed] for seed in expected_seeds if seed in values_by_seed]
    count = len(ordered_values)
    mean = math.fsum(ordered_values) / count
    squared_deviations = math.fsum((value - mean) ** 2 for value in ordered_values)
    population_variance = squared_deviations / count
    sample_variance = squared_deviations / (count - 1) if count > 1 else None

    return {
        "num_seeds": count,
        "missing_seeds": [seed for seed in expected_seeds if seed not in values_by_seed],
        "values_by_seed": {
            str(seed): values_by_seed[seed]
            for seed in expected_seeds
            if seed in values_by_seed
        },
        "mean": mean,
        "variance_population": population_variance,
        "variance_sample": sample_variance,
        "std_population": math.sqrt(population_variance),
        "std_sample": math.sqrt(sample_variance) if sample_variance is not None else None,
    }


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if not path:
        raise ValueError("Cannot assign an empty metric path.")
    cursor = target
    for key in path[:-1]:
        existing = cursor.setdefault(key, {})
        if not isinstance(existing, dict):
            raise ValueError(f"Metric path collision at {'.'.join(path)}")
        cursor = existing
    cursor[path[-1]] = value


def _build_statistics(
    payloads_by_seed: dict[int, dict[str, Any]], expected_seeds: list[int]
) -> dict[str, Any]:
    """-seed """
    values_by_path: dict[tuple[str, ...], dict[int, float]] = defaultdict(dict)
    for seed, payload in payloads_by_seed.items():
        sections = {
            "datasets": payload["datasets"],
            "overall_by_split": payload["overall_by_split"],
        }
        for path, value in _iter_numeric_leaves(sections):
            values_by_path[path][seed] = value

    statistics: dict[str, Any] = {}
    for path in sorted(values_by_path):
        _set_nested(
            statistics,
            path,
            _metric_statistics(values_by_path[path], expected_seeds),
        )
    return statistics


def _atomic_write_json(output_file: Path, payload: dict[str, Any]) -> None:
    """Publish a cross-seed aggregate only after complete JSON serialization."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(output_file.name + ".partial")
    temporary_file.unlink(missing_ok=True)
    try:
        with temporary_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temporary_file.replace(output_file)
    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="SEED:JSON",
        help="A single-seed aggregate eval_results.json. Repeat once per seed.",
    )
    parser.add_argument(
        "--expected_seeds",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated complete seed set; missing or extra input seeds are rejected.",
    )
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--run_timestamp", type=str, default=None)
    args = parser.parse_args()

    try:
        expected_seeds = _parse_expected_seeds(args.expected_seeds)
    except ValueError as exc:
        parser.error(str(exc))

    payloads_by_seed: dict[int, dict[str, Any]] = {}
    source_paths: dict[int, Path] = {}
    for spec in args.result:
        try:
            seed, result_path = _parse_result_spec(spec)
        except ValueError as exc:
            parser.error(str(exc))
        if seed in payloads_by_seed:
            parser.error(f"Duplicate input for seed {seed}.")
        if not result_path.is_file():
            parser.error(f"Result file does not exist: {result_path}")

        try:
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"Unable to read valid JSON from {result_path}: {exc}")
        if not isinstance(payload, dict):
            parser.error(f"Malformed aggregate result (top-level JSON must be an object): {result_path}")
        try:
            _validate_finite_numbers(payload)
        except ValueError as exc:
            parser.error(f"Malformed aggregate result: {result_path}: {exc}")
        if not isinstance(payload.get("datasets"), dict):
            parser.error(f"Malformed aggregate result (datasets missing): {result_path}")
        if not isinstance(payload.get("overall_by_split"), dict):
            parser.error(f"Malformed aggregate result (overall_by_split missing): {result_path}")
        checkpoint = payload.get("ckpt_path")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            parser.error(f"Malformed aggregate result (ckpt_path missing): {result_path}")
        payload_seed = payload.get("seed")
        if (
            not isinstance(payload_seed, int)
            or isinstance(payload_seed, bool)
            or payload_seed != seed
        ):
            parser.error(
                f"Seed mismatch for {result_path}: file={payload_seed!r}, argument={seed}."
            )
        payloads_by_seed[seed] = payload
        source_paths[seed] = result_path

    actual_seeds = sorted(payloads_by_seed)
    if actual_seeds != sorted(expected_seeds):
        missing = sorted(set(expected_seeds) - set(actual_seeds))
        extra = sorted(set(actual_seeds) - set(expected_seeds))
        parser.error(
            f"Input seed set does not match expected seeds; missing={missing}, extra={extra}."
        )

    ckpt_paths = {str(payload.get("ckpt_path")) for payload in payloads_by_seed.values()}
    if len(ckpt_paths) != 1:
        parser.error(f"All seed results must use the same checkpoint, got: {sorted(ckpt_paths)}")

    dataset_sets = {tuple(sorted(payload["datasets"])) for payload in payloads_by_seed.values()}
    split_sets = {
        tuple(sorted(payload["overall_by_split"]))
        for payload in payloads_by_seed.values()
    }
    if len(dataset_sets) != 1:
        parser.error(f"Dataset sets differ across seeds: {sorted(dataset_sets)}")
    if len(split_sets) != 1:
        parser.error(f"Split sets differ across seeds: {sorted(split_sets)}")

    try:
        statistics = _build_statistics(payloads_by_seed, expected_seeds)
    except ValueError as exc:
        parser.error(str(exc))

    output = {
        "format_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_timestamp": args.run_timestamp,
        "ckpt_path": next(iter(ckpt_paths)),
        "seed_count": len(expected_seeds),
        "seeds": expected_seeds,
        "datasets": list(next(iter(dataset_sets))),
        "splits": list(next(iter(split_sets))),
        "variance_definition": {
            "variance_population": "sum((x - mean)^2) / N (ddof=0)",
            "variance_sample": "sum((x - mean)^2) / (N - 1) (ddof=1)",
        },
        "aggregation_note": (
            "Statistics are computed across seeds for every numeric leaf under datasets and "
            "overall_by_split. Dataset/split/category levels remain separate. A metric with "
            "num_seeds < seed_count is based only on the listed values_by_seed and reports the "
            "other seeds in missing_seeds."
        ),
        "source_results": [
            {
                "seed": seed,
                "path": str(source_paths[seed]),
                "run_timestamp": payloads_by_seed[seed].get("run_timestamp"),
            }
            for seed in expected_seeds
        ],
        "statistics": statistics,
    }

    output_file = args.output_file.expanduser().resolve()
    _atomic_write_json(output_file, output)
    print(f"Cross-seed mean/variance results saved to {output_file}")


if __name__ == "__main__":
    main()
