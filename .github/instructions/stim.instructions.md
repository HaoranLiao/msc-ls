---
applyTo: '**/*_stim*'
---

# Stim Usage Guide

Use these rules when writing or reviewing Stim code.

## Circuit Model

- Stim simulates stabilizer circuits; label any Clifford proxy for a
  non-Clifford protocol explicitly.
- Qubits are integer indices. Add `QUBIT_COORDS` when geometry or diagrams
  matter.
- Prefer Stim's native circuit generators for standard experiments.
- Append batched targets for operations in the same layer instead of emitting
  one instruction per target.
- Represent repeated rounds with `REPEAT` blocks rather than unrolling large
  circuits.

## Measurements and Scheduling

- Measurements enter one global record in execution order. Sampler column `0`
  is the first measurement appended.
- Result `0` means eigenvalue `+1` and result `1` means `-1`, unless the target
  is inverted with `stim.target_inv(...)`.
- Treat record positions as part of the circuit contract. Centralize their
  layout; avoid scattering unexplained `rec[-k]` offsets through the code.
- `TICK` separates intended time layers. Put parallel operations in the same
  tick and never place colliding two-qubit gates in one layer.
- A tick is not a detector boundary; detectors require explicit record
  comparisons.
- For destructive logical readout, parity the support measurements and apply
  any Pauli-string phase correction, especially for odd-weight Y operators.

## Detectors and Observables

- `DETECTOR` and `OBSERVABLE_INCLUDE` reference existing measurements; neither
  performs a measurement.
- Compare syndromes with the correct preparation, previous-round, or final
  boundary. A nonzero raw syndrome is not automatically a detection event.
- `rec[-k]` is relative to the current record end. Recheck offsets whenever
  earlier measurements change.
- `L0`, `L1`, and so on are observable indices, not inherently logical X or Z.
- Compile detector samplers, measurement-to-detection converters, or detector
  error models only after detector and observable annotations are complete.
- Complete Stim protocol builders always emit their applicable detector
  annotations, and emit `OBSERVABLE_INCLUDE` whenever they append a terminal
  logical readout. When protocols share a schedule but differ in detector
  relations, select the policy with an explicit parameter on the one builder
  so every branch stays fully annotated; do not add opt-out flags that can
  leave a circuit under-annotated, split the schedule into a separate
  measurement-only primitive, or duplicate lower-level annotations in an
  enclosing builder.

## Noise and Performance

- Build the noiseless circuit first, then apply the noise model once.
- Model single-qubit, two-qubit, reset, and measurement faults separately when
  their physical rates differ.
- Handle `REPEAT` blocks deliberately during circuit transformations.
- Compile samplers and converters once, outside shot or parameter-sweep loops,
  then reuse them.
- Use deterministic Pauli faults to debug propagation before stochastic noise.

## Verification and Debugging

- Never inspect a circuit with `print(circuit)`. Use
  `show_timeline(circuit)` from `msc/utils/circuit_stim.py`, which renders the
  timeline diagram on a white background with enlarged record and detector
  annotations. Tune `zoom` and `annotation_font_size` for wide circuits, and
  fall back to the raw text form only when a diagram cannot be rendered.
- Use `stim.TableauSimulator` and `peek_observable_expectation()` for exact
  stabilizer checks.
- Seed stochastic regression tests, and assert record count and semantic order
  in addition to sampled values.
- Use timeline diagrams for operation order and timeslice diagrams for
  tick-by-tick geometry.
- For detector circuits, inspect the detector error model and use
  `shortest_graphlike_error()` as a useful distance sanity check when its
  graphlike assumptions apply.
- Temporarily flatten repeat blocks for inspection, not as the default circuit
  representation.
