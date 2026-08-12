#!/usr/bin/env python3
"""Audit Figure 9 using only Hirano et al.'s released circuit builder."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import stim


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PLOT_SCRIPT = ROOT / "fig" / "plot_error_detection.py"
SHEET_EXPORT = ROOT / "audit" / "fig9_google_sheet_gid_573444527.xlsx"
UPSTREAM_COMMIT = "c89a53a7a7062b552d31f4c41ae2ba857756359a"

# lattice_surgery_error_detection imports sinter for its CLI, but this audit
# directly constructs the circuit and does not use sinter.
if "sinter" not in sys.modules:
    sys.modules["sinter"] = types.ModuleType("sinter")
sys.path.insert(0, str(SRC))

from lattice_surgery_error_detection import (  # noqa: E402
    InitialValue,
    QubitMapping,
    SteanePlusSurfaceCode,
    SteaneSyndromeExtractionPattern,
)


@dataclass(frozen=True)
class FaultEffect:
    detector_mask: int
    logical_mask: int
    explanation: stim.ExplainedError


@dataclass(frozen=True)
class BuiltExperiment:
    wrapper: SteanePlusSurfaceCode
    circuit: stim.Circuit
    postselection_mask: int


# These indices identify compact witnesses in the undecomposed detector error
# model of the exact upstream commit. They are verified below, not assumed.
KNOWN_AMBIGUITIES = {
    "+": ((186, 246), (206, 341)),
    "0": ((212, 818), (185, 1350)),
}


def build_experiment(probe: str, *, full_post_selection: bool) -> BuiltExperiment:
    initial_value = InitialValue.Plus if probe == "+" else InitialValue.Zero
    wrapper = SteanePlusSurfaceCode(
        QubitMapping(30, 30),
        5,
        initial_value,
        SteaneSyndromeExtractionPattern.ZXZ,
        0.001,
        full_post_selection,
    )
    wrapper.run()
    postselection_mask = sum(
        1 << detector.id for detector in wrapper.circuit.detectors_for_post_selection
    )
    return BuiltExperiment(wrapper, wrapper.circuit.circuit, postselection_mask)


def mask_to_indices(mask: int) -> list[int]:
    indices: list[int] = []
    while mask:
        lowest_bit = mask & -mask
        indices.append(lowest_bit.bit_length() - 1)
        mask ^= lowest_bit
    return indices


def dem_targets_to_masks(targets: list[stim.DemTarget]) -> tuple[int, int]:
    detector_mask = 0
    logical_mask = 0
    for target in targets:
        if target.is_relative_detector_id():
            detector_mask ^= 1 << target.val
        elif target.is_logical_observable_id():
            logical_mask ^= 1 << target.val
    return detector_mask, logical_mask


def extract_fault_effects(circuit: stim.Circuit) -> list[FaultEffect]:
    """Extract every elementary fault's detector and logical effect."""
    detector_error_model = circuit.detector_error_model(
        decompose_errors=False,
        flatten_loops=True,
    )
    model_effects = [
        dem_targets_to_masks(instruction.targets_copy())
        for instruction in detector_error_model
        if instruction.type == "error"
    ]
    explanations = circuit.explain_detector_error_model_errors(
        reduce_to_one_representative_error=True,
    )
    if len(model_effects) != len(explanations):
        raise AssertionError("Detector-error effects and explanations have different lengths")

    effects: list[FaultEffect] = []
    for model_effect, explanation in zip(model_effects, explanations):
        explained_effect = dem_targets_to_masks(
            [term.dem_target for term in explanation.dem_error_terms]
        )
        if model_effect != explained_effect:
            raise AssertionError("Stim explanation order differs from detector-error-model order")
        effects.append(FaultEffect(*model_effect, explanation))
    return effects


def combine_pair(effects: list[FaultEffect], pair: tuple[int, int]) -> tuple[int, int]:
    first, second = (effects[index] for index in pair)
    return (
        first.detector_mask ^ second.detector_mask,
        first.logical_mask ^ second.logical_mask,
    )


def verify_ambiguity(
    effects: list[FaultEffect],
    postselection_mask: int,
    pairs: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    first_effect = combine_pair(effects, pairs[0])
    second_effect = combine_pair(effects, pairs[1])
    if first_effect[0] & postselection_mask or second_effect[0] & postselection_mask:
        raise AssertionError("A witness pair triggers a postselected detector")
    if first_effect[0] != second_effect[0]:
        raise AssertionError("Witness pairs do not produce the same detector syndrome")
    if first_effect[1] == second_effect[1]:
        raise AssertionError("Witness pairs do not have opposite logical outcomes")
    return first_effect[0], first_effect[1] ^ second_effect[1]


def rediscover_ambiguity(
    effects: list[FaultEffect],
    postselection_mask: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find two accepted two-fault pairs with one syndrome and opposite logicals."""
    first_pair_by_syndrome_and_logical: dict[tuple[int, int], tuple[int, int]] = {}
    for first_index, first in enumerate(effects):
        for second_index in range(first_index + 1, len(effects)):
            second = effects[second_index]
            detector_mask = first.detector_mask ^ second.detector_mask
            if detector_mask & postselection_mask:
                continue
            logical_mask = first.logical_mask ^ second.logical_mask
            opposite = first_pair_by_syndrome_and_logical.get(
                (detector_mask, logical_mask ^ 1)
            )
            if opposite is not None:
                return opposite, (first_index, second_index)
            first_pair_by_syndrome_and_logical.setdefault(
                (detector_mask, logical_mask),
                (first_index, second_index),
            )
    raise AssertionError("No accepted two-fault ambiguity found")


def print_ambiguity(
    probe: str,
    effects: list[FaultEffect],
    postselection_mask: int,
    pairs: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    syndrome_mask, logical_difference = verify_ambiguity(
        effects,
        postselection_mask,
        pairs,
    )
    print(f"\n{probe} probe: accepted same-syndrome/opposite-logical witness")
    print(f"  common detection events: {mask_to_indices(syndrome_mask)}")
    print(f"  logical difference: L0={logical_difference}")
    for pair_number, pair in enumerate(pairs, start=1):
        detector_mask, logical_mask = combine_pair(effects, pair)
        print(f"  pair {pair_number}: DEM effect indices {pair}, L0={logical_mask}")
        print(f"    combined detection events: {mask_to_indices(detector_mask)}")
        for fault_number, effect_index in enumerate(pair, start=1):
            effect = effects[effect_index]
            print(
                f"    fault {fault_number}: effect index {effect_index}, "
                f"D={mask_to_indices(effect.detector_mask)}, L0={effect.logical_mask}"
            )
            for line in str(effect.explanation).splitlines():
                print(f"      {line}")


def read_sheet_chart_title() -> str:
    namespaces = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    }
    with zipfile.ZipFile(SHEET_EXPORT) as workbook:
        chart = ET.fromstring(workbook.read("xl/charts/chart1.xml"))
    text_runs = [
        node.text or ""
        for node in chart.findall("./c:chart/c:title//a:t", namespaces)
    ]
    if not text_runs:
        raise AssertionError("No chart title found in the spreadsheet export")
    title = "".join(text_runs)
    if not title.startswith("full post selection"):
        raise AssertionError(f"Unexpected spreadsheet chart title: {title!r}")
    return title


def read_released_acceptance() -> dict[str, float]:
    module = ast.parse(PLOT_SCRIPT.read_text())
    acceptance: dict[str, float] = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "Task":
            continue
        values = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
        if values.get("p") != 0.001:
            continue
        accepted = values["num_valid_cases"] + values["num_wrong_cases"]
        shots = accepted + values["num_discarded_cases"]
        acceptance[values["initial"]] = accepted / shots
    if set(acceptance) != {"+", "0"}:
        raise AssertionError("Could not extract both p=0.001 Figure 9 data points")
    return acceptance


def sample_acceptance(experiment: BuiltExperiment, shots: int, seed: int) -> float:
    detection_events = experiment.circuit.compile_detector_sampler(seed=seed).sample(shots)
    postselected_detectors = mask_to_indices(experiment.postselection_mask)
    accepted = ~np.any(detection_events[:, postselected_detectors], axis=1)
    return float(np.mean(accepted))


def audit_postselection(sample_shots: int) -> None:
    print(f"Upstream commit: {UPSTREAM_COMMIT}")
    print(f"Spreadsheet chart title: {read_sheet_chart_title()!r}")
    released_acceptance = read_released_acceptance()
    for probe_index, probe in enumerate(("+", "0")):
        print(f"\n{probe} probe at p=0.001")
        print(f"  released Figure 9 acceptance: {released_acceptance[probe]:.6%}")
        sampled_acceptance: dict[bool, float] = {}
        for full_post_selection in (False, True):
            experiment = build_experiment(
                probe,
                full_post_selection=full_post_selection,
            )
            postselection_count = experiment.postselection_mask.bit_count()
            print(
                f"  full_post_selection={full_post_selection}: "
                f"detectors={experiment.circuit.num_detectors}, "
                f"postselected={postselection_count}"
            )
            if full_post_selection and postselection_count != experiment.circuit.num_detectors:
                raise AssertionError("The full branch did not postselect every detector")
            if sample_shots:
                acceptance = sample_acceptance(
                    experiment,
                    sample_shots,
                    seed=1234 + probe_index * 2 + int(full_post_selection),
                )
                sampled_acceptance[full_post_selection] = acceptance
                print(f"    sampled acceptance ({sample_shots:,} shots): {acceptance:.6%}")
        if sample_shots:
            full_difference = abs(sampled_acceptance[True] - released_acceptance[probe])
            non_full_difference = abs(sampled_acceptance[False] - released_acceptance[probe])
            if full_difference >= non_full_difference:
                raise AssertionError("Released acceptance is not closer to the full branch")


def audit_fault_pairs(rediscover: bool) -> None:
    for probe in ("+", "0"):
        experiment = build_experiment(probe, full_post_selection=False)
        effects = extract_fault_effects(experiment.circuit)
        pairs = KNOWN_AMBIGUITIES[probe]
        verify_ambiguity(effects, experiment.postselection_mask, pairs)
        if rediscover:
            rediscovered_pairs = rediscover_ambiguity(effects, experiment.postselection_mask)
            verify_ambiguity(effects, experiment.postselection_mask, rediscovered_pairs)
            print(f"\n{probe} probe: independently rediscovered witness {rediscovered_pairs}")
        print_ambiguity(probe, effects, experiment.postselection_mask, pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-shots",
        type=int,
        default=100_000,
        help="shots per probe and postselection mode; use 0 to skip sampling",
    )
    parser.add_argument(
        "--rediscover",
        action="store_true",
        help="search the full elementary-effect list instead of only verifying saved witnesses",
    )
    args = parser.parse_args()
    audit_postselection(args.sample_shots)
    audit_fault_pairs(args.rediscover)


if __name__ == "__main__":
    main()
