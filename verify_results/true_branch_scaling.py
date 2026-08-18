#!/usr/bin/env python3
"""Parallel fresh Monte Carlo sweep of the authors' full-postselection circuit."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np
import pymatching
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# The authors' circuit module imports Sinter for its CLI. This script constructs
# the circuit directly and does not use that CLI, so a stub avoids adding an
# otherwise unused runtime dependency.
sys.modules["sinter"] = types.ModuleType("sinter")
sys.path.insert(0, str(SRC))

from lattice_surgery_error_detection import (  # noqa: E402
    InitialValue,
    QubitMapping,
    SteanePlusSurfaceCode,
    SteaneSyndromeExtractionPattern,
)


P_VALUES = np.array([0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0014, 0.0017, 0.0020])


@dataclass(frozen=True)
class Job:
    probe: str
    p: float
    shard: int
    target_errors: int
    max_shots: int
    batch_shots: int
    seed: int


@dataclass(frozen=True)
class Result:
    probe: str
    p: float
    shard: int
    shots: int
    accepted: int
    errors: int


def run_job(job: Job) -> Result:
    initial_value = InitialValue.Plus if job.probe == "+" else InitialValue.Zero
    experiment = SteanePlusSurfaceCode(
        QubitMapping(30, 30),
        5,
        initial_value,
        SteaneSyndromeExtractionPattern.ZXZ,
        job.p,
        True,
    )
    experiment.run()
    circuit = experiment.circuit.circuit
    if len(experiment.circuit.detectors_for_post_selection) != circuit.num_detectors:
        raise AssertionError("full_post_selection=True did not select every detector")

    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    if np.any(matching.decode(np.zeros(circuit.num_detectors, dtype=np.uint8))):
        raise AssertionError("PyMatching applies a logical correction to the accepted zero syndrome")

    sampler = circuit.detector_error_model(decompose_errors=False).compile_sampler(seed=job.seed)
    shots = 0
    accepted = 0
    errors = 0
    while errors < job.target_errors and shots < job.max_shots:
        batch = min(job.batch_shots, job.max_shots - shots)
        detection_events, observable_flips, _ = sampler.sample(batch, bit_packed=True)
        accepted_mask = ~np.any(detection_events, axis=1)
        accepted += int(np.count_nonzero(accepted_mask))
        errors += int(np.count_nonzero(observable_flips[accepted_mask, 0] & 1))
        shots += batch
    return Result(job.probe, job.p, job.shard, shots, accepted, errors)


def power_law(p: np.ndarray, coefficient: float, exponent: float) -> np.ndarray:
    return coefficient * p**exponent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors-per-shard", type=int, default=5)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--max-shots-per-shard", type=int, default=1_000_000_000)
    parser.add_argument("--batch-shots", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-p", type=float, default=0.0005)
    parser.add_argument("--seed-base", type=int, default=50_000)
    args = parser.parse_args()

    fit_p_values = P_VALUES[P_VALUES >= args.min_p]
    jobs = [
        Job(
            probe,
            float(p),
            shard,
            args.errors_per_shard,
            args.max_shots_per_shard,
            args.batch_shots,
            args.seed_base + probe_index * 10_000 + p_index * 100 + shard,
        )
        for probe_index, probe in enumerate(("+", "0"))
        for p_index, p in enumerate(fit_p_values)
        for shard in range(args.shards)
    ]

    results: list[Result] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"# completed {result.probe} p={result.p:.4g} shard={result.shard}: "
                f"shots={result.shots}, errors={result.errors}",
                flush=True,
            )

    print("probe,p,shots,accepted,errors,acceptance,logical_error_rate", flush=True)
    aggregated: dict[tuple[str, float], Result] = {}
    for probe in ("+", "0"):
        for p in fit_p_values:
            point = [result for result in results if result.probe == probe and result.p == p]
            aggregated[(probe, float(p))] = Result(
                probe,
                float(p),
                -1,
                sum(result.shots for result in point),
                sum(result.accepted for result in point),
                sum(result.errors for result in point),
            )
            result = aggregated[(probe, float(p))]
            print(
                f"{probe},{p:.4g},{result.shots},{result.accepted},{result.errors},"
                f"{result.accepted / result.shots:.10g},{result.errors / result.accepted:.10g}",
                flush=True,
            )

    for probe in ("+", "0"):
        rates = np.array([
            aggregated[(probe, float(p))].errors / aggregated[(probe, float(p))].accepted
            for p in fit_p_values
        ])
        if np.any(rates == 0):
            print(f"# {probe}: fit unavailable because at least one point has zero errors")
            continue
        fit, covariance = curve_fit(
            power_law,
            fit_p_values,
            rates,
            p0=(1_000, 3),
            maxfev=100_000,
        )
        print(
            f"# {probe}: C={fit[0]:.10g}, alpha={fit[1]:.10g}, "
            f"alpha_fit_se={np.sqrt(covariance[1, 1]):.10g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
