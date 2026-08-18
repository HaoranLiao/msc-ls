#!/usr/bin/env python3
"""Fit the released circuit's full_post_selection=False logical-error scaling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import types

import numpy as np
import pymatching


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.modules["sinter"] = types.ModuleType("sinter")
sys.path.insert(0, str(SRC))

from lattice_surgery_error_detection import (  # noqa: E402
    InitialValue,
    QubitMapping,
    SteanePlusSurfaceCode,
    SteaneSyndromeExtractionPattern,
)


P_VALUES = np.array([0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0014, 0.0017, 0.0020])


def run_point(
    probe: str,
    p: float,
    *,
    target_errors: int,
    max_shots: int,
    batch_shots: int,
    seed: int,
) -> tuple[int, int, int]:
    """Sample one physical error rate using the authors' complete false-branch circuit.

    The circuit has a noisy surface preparation/initial syndrome round, an
    ideal encoded Steane input, three noisy ZXZ merge rounds, five noisy
    post-merge surface-code rounds, and a noisy destructive logical readout.
    The released false branch does not append a noise-free closing round.
    """
    initial_value = InitialValue.Plus if probe == "+" else InitialValue.Zero
    experiment = SteanePlusSurfaceCode(
        QubitMapping(30, 30),
        5,
        initial_value,
        SteaneSyndromeExtractionPattern.ZXZ,
        p,
        False,
    )
    experiment.run()
    circuit = experiment.circuit.circuit
    postselected_detectors = [
        detector.id for detector in experiment.circuit.detectors_for_post_selection
    ]
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    sampler = circuit.compile_detector_sampler(seed=seed)

    # Cumulative Monte Carlo counters across all sampled batches:
    # - shots: every circuit execution, including executions later discarded;
    # - accepted: executions with zero on every hard-postselection detector;
    # - errors: accepted executions whose decoded observable is incorrect.
    shots = 0
    accepted = 0
    errors = 0

    # Stop after obtaining the requested logical-error statistics or exhausting
    # the shot budget. The last batch is shortened to end exactly at max_shots.
    while errors < target_errors and shots < max_shots:
        batch = min(batch_shots, max_shots - shots)

        # detection_events[shot, detector] is the sampled detector syndrome;
        # observable_flips[shot, observable] is the actual logical-frame flip.
        detection_events, observable_flips = sampler.sample(
            batch,
            separate_observables=True,
        )

        # accepted_mask selects shots that pass all of Hirano's hard
        # postselection conditions. Non-postselected surface detectors may be 1
        # and remain available to the matching decoder.
        accepted_mask = ~np.any(
            detection_events[:, postselected_detectors],
            axis=1,
        )

        # predictions is PyMatching's predicted logical-observable correction
        # for each accepted syndrome. A mismatch with the sampled observable
        # flip is one logical failure.
        predictions = matching.decode_batch(detection_events[accepted_mask])
        errors += int(
            np.count_nonzero(
                np.any(predictions != observable_flips[accepted_mask], axis=1)
            )
        )
        accepted += int(np.count_nonzero(accepted_mask))
        shots += batch
    return shots, accepted, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-errors", type=int, default=500)
    parser.add_argument("--max-shots", type=int, default=5_000_000)
    parser.add_argument("--batch-shots", type=int, default=250_000)
    args = parser.parse_args()

    print("probe,p,shots,accepted,errors,acceptance,logical_error_rate", flush=True)
    for probe_index, probe in enumerate(("+", "0")):
        rates: list[float] = []
        error_counts: list[int] = []
        for p_index, p in enumerate(P_VALUES):
            shots, accepted, errors = run_point(
                probe,
                float(p),
                target_errors=args.target_errors,
                max_shots=args.max_shots,
                batch_shots=args.batch_shots,
                seed=10_000 + probe_index * 100 + p_index,
            )
            rate = errors / accepted
            rates.append(rate)
            error_counts.append(errors)
            print(
                f"{probe},{p:.4g},{shots},{accepted},{errors},"
                f"{accepted / shots:.10g},{rate:.10g}",
                flush=True,
            )
        unweighted_fit = np.polyfit(np.log(P_VALUES), np.log(rates), 1)
        weighted_fit = np.polyfit(
            np.log(P_VALUES),
            np.log(rates),
            1,
            w=np.sqrt(error_counts),
        )
        print(
            f"# {probe} unweighted: C={np.exp(unweighted_fit[1]):.10g}, "
            f"alpha={unweighted_fit[0]:.10g}",
            flush=True,
        )
        print(
            f"# {probe} error-weighted: C={np.exp(weighted_fit[1]):.10g}, "
            f"alpha={weighted_fit[0]:.10g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
