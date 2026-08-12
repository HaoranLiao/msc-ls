"""Build the executed-record notebook for the Hirano Figure 9 audit."""

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


AUDIT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = AUDIT_DIR / "fig9_postselection_and_fault_pairs.ipynb"


notebook = nbf.v4.new_notebook()
notebook.metadata.update(
    {
        "kernelspec": {
            "display_name": "Python 3 (msc)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
)
notebook.cells = [
    nbf.v4.new_markdown_cell(
        """# Hirano Figure 9: postselection and two-fault audit

This notebook is an executed record of the audit of Hirano *et al.* commit
`c89a53a7a7062b552d31f4c41ae2ba857756359a`. It uses only the vendored authors'
circuit builder, their released Figure 9 counts, the linked spreadsheet export,
and Stim. No circuit or decoder from the surrounding repository is imported.

The `+` experiment is the logical-Z-error probe; the `0` experiment is the
logical-X-error probe."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import inspect
import sys

import numpy as np
import pymatching
import stim
from scipy.optimize import curve_fit

AUDIT_DIR = Path.cwd()
assert (AUDIT_DIR / "fig9_postselection_audit.py").exists(), "Run this notebook from its audit directory"
sys.path.insert(0, str(AUDIT_DIR))

from fig9_postselection_audit import (
    UPSTREAM_COMMIT,
    audit_postselection,
    build_experiment,
    extract_fault_effects,
    print_ambiguity,
    rediscover_ambiguity,
    verify_ambiguity,
)
from false_branch_scaling import P_VALUES, run_point
from true_branch_scaling import power_law as true_power_law, run_job as run_true_job

print(f"upstream commit: {UPSTREAM_COMMIT}")
print(f"Stim: {stim.__version__}")
print(f"PyMatching: {pymatching.__version__}")
print(f"NumPy: {np.__version__}")"""
    ),
    nbf.v4.new_markdown_cell(
        """## Figure 9 postselection provenance

The next cell reads the chart title from the saved export of the Google Sheet
linked by the authors' plotting script, extracts its released (p=10^{-3})
acceptance rates, builds both branches of the authors' circuit, and samples each
with fixed seeds. The full branch is required to mark every detector for
postselection."""
    ),
    nbf.v4.new_code_cell("audit_postselection(sample_shots=100_000)"),
    nbf.v4.new_markdown_cell(
        r"""## Scaling of the authors' non-full branch

This cell directly constructs and samples the authors' exact
`full_post_selection=False` circuit with its postselection mask and PyMatching
decoder. Sampling continues to 1000 logical errors or ten million shots at
each of the eight Figure 9 physical-error values. The quoted exponent is an
effective finite-window log-log fit; it is distinct from the asymptotic order
established by fault enumeration below."""
    ),
    nbf.v4.new_code_cell(
        """scaling_results = {}
for probe_index, (probe, error_name) in enumerate((("+", "Z"), ("0", "X"))):
    rows = []
    print(f"{error_name}-error probe ({probe}):")
    for p_index, p in enumerate(P_VALUES):
        shots, accepted, errors = run_point(
            probe,
            float(p),
            target_errors=1000,
            max_shots=10_000_000,
            batch_shots=100_000,
            seed=10_000 + probe_index * 100 + p_index,
        )
        rate = errors / accepted
        rows.append((float(p), shots, accepted, errors, rate))
        print(
            f"  p={p:.4g}, shots={shots:,}, accepted={accepted:,}, "
            f"errors={errors}, acceptance={accepted / shots:.6%}, pL={rate:.10g}",
            flush=True,
        )
    scaling_results[probe] = rows
    p_values = np.array([row[0] for row in rows])
    rates = np.array([row[4] for row in rows])
    error_counts = np.array([row[3] for row in rows])
    unweighted = np.polyfit(np.log(p_values), np.log(rates), 1)
    weighted = np.polyfit(
        np.log(p_values),
        np.log(rates),
        1,
        w=np.sqrt(error_counts),
    )
    print(f"{error_name}-error fits:")
    print(f"  unweighted: pL ≈ {np.exp(unweighted[1]):.3g} p^{unweighted[0]:.3f}")
    print(f"  error-weighted: pL ≈ {np.exp(weighted[1]):.3g} p^{weighted[0]:.3f}")
"""
    ),
    nbf.v4.new_markdown_cell(
        r"""## Fresh Monte Carlo for the authors' full branch

This records an independent simulation of the authors' exact
`full_post_selection=True` circuit. The helper shown below builds that circuit,
asserts that all 135 detectors are postselected, verifies that PyMatching makes
no logical correction for the accepted all-zero syndrome, and samples the
undecomposed Stim detector error model with fixed independent seeds.

The high-statistics run uses the six probabilities from (8\times10^{-4})
through (2\times10^{-3}), omitting the two smallest Figure 9 probabilities. It
targets 50 logical errors in each of ten independent shards, producing 500--506
logical errors per point. The quoted uncertainty is the standard deviation of
20,000 Poisson counting-noise bootstrap refits using the same unweighted
nonlinear fit of (C p^\alpha) that reproduces the paper's reported exponents
from its released counts."""
    ),
    nbf.v4.new_code_cell("print(inspect.getsource(run_true_job))"),
    nbf.v4.new_code_cell(
        """true_scaling_results = {
    "+": [
        (0.0008, 3_954_000_000, 1_580_710_343, 504),
        (0.0010, 2_242_000_000, 712_784_921, 503),
        (0.0012, 1_640_000_000, 414_613_001, 503),
        (0.0014, 1_420_000_000, 285_469_396, 506),
        (0.0017, 996_000_000, 142_038_124, 505),
        (0.0020, 884_000_000, 89_404_785, 506),
    ],
    "0": [
        (0.0008, 19_630_000_000, 7_866_015_738, 500),
        (0.0010, 11_244_000_000, 3_585_192_687, 501),
        (0.0012, 8_066_000_000, 2_046_361_433, 500),
        (0.0014, 7_132_000_000, 1_439_832_353, 501),
        (0.0017, 5_422_000_000, 777_027_748, 501),
        (0.0020, 4_812_000_000, 489_572_832, 501),
    ],
}

for probe, error_name in (("+", "Z"), ("0", "X")):
    rows = true_scaling_results[probe]
    print(f"{error_name}-error probe ({probe}), fresh full-postselection run:")
    for p, shots, accepted, errors in rows:
        print(
            f"  p={p:.4g}, shots={shots:,}, accepted={accepted:,}, "
            f"errors={errors}, acceptance={accepted / shots:.6%}, "
            f"pL={errors / accepted:.10g}"
        )
    p_values = np.array([row[0] for row in rows])
    accepted_counts = np.array([row[2] for row in rows])
    error_counts = np.array([row[3] for row in rows])
    rates = error_counts / accepted_counts
    fit, covariance = curve_fit(
        true_power_law,
        p_values,
        rates,
        p0=(1000, 3),
        maxfev=100_000,
    )
    rng = np.random.default_rng(20260809 + (probe == "0"))
    bootstrap_exponents = []
    for _ in range(20_000):
        bootstrap_rates = rng.poisson(error_counts) / accepted_counts
        bootstrap_fit, _ = curve_fit(
            true_power_law,
            p_values,
            bootstrap_rates,
            p0=fit,
            maxfev=10_000,
        )
        bootstrap_exponents.append(bootstrap_fit[1])
    bootstrap_exponents = np.asarray(bootstrap_exponents)
    bootstrap_se = np.std(bootstrap_exponents, ddof=1)
    bootstrap_interval = np.quantile(bootstrap_exponents, [0.025, 0.975])
    print(
        f"  six-point paper-style fit: pL ≈ {fit[0]:.3g} p^{fit[1]:.3f} "
        f"(bootstrap 1σ={bootstrap_se:.3f}, "
        f"95% CI=[{bootstrap_interval[0]:.3f}, {bootstrap_interval[1]:.3f}])"
    )
    print(
        f"  six-point totals: shots={sum(row[1] for row in rows):,}, "
        f"logical errors={sum(row[3] for row in rows)}"
    )
"""
    ),
    nbf.v4.new_markdown_cell(
        r"""## How the fault pairs are found

For each elementary error mechanism (i), the undecomposed Stim detector error
model supplies a binary detector mask (d_i) and logical mask \(\ell_i\). A
two-fault combination ((i,j)) has

\[
d_{ij}=d_i\mathbin{\mathrm{XOR}}d_j,\qquad
\ell_{ij}=\ell_i\mathbin{\mathrm{XOR}}\ell_j.
\]

It is accepted when (d_{ij}) has no overlap with the authors' postselection
mask. The search groups accepted pairs by their combined detector mask and
looks for two pairs with the same mask but opposite logical masks. Stim then
maps each DEM effect back to a representative physical Pauli fault and exact
circuit instruction. The witnesses below use distinct circuit locations, so
each combination is a physically possible two-fault event with nonzero
(O(p^2)) probability."""
    ),
    nbf.v4.new_markdown_cell(
        "### Elementary-effect extraction from the authors' circuit"
    ),
    nbf.v4.new_code_cell("print(inspect.getsource(extract_fault_effects))"),
    nbf.v4.new_markdown_cell("### Exhaustive same-syndrome search"),
    nbf.v4.new_code_cell("print(inspect.getsource(rediscover_ambiguity))"),
    nbf.v4.new_markdown_cell(
        """## Independently rediscovered fault pairs

This reruns the search on each authors' `full_post_selection=False` circuit.
`print_ambiguity` verifies acceptance, equal detector syndromes, and opposite
logical outcomes before printing Stim's physical circuit locations."""
    ),
    nbf.v4.new_code_cell(
        """for probe in ("+", "0"):
    experiment = build_experiment(probe, full_post_selection=False)
    effects = extract_fault_effects(experiment.circuit)
    pairs = rediscover_ambiguity(effects, experiment.postselection_mask)
    verify_ambiguity(effects, experiment.postselection_mask, pairs)
    print(f"\\n{probe} probe: rediscovered {pairs}")
    print_ambiguity(probe, effects, experiment.postselection_mask, pairs)"""
    ),
    nbf.v4.new_markdown_cell(
        r"""## Interpretation

Across the finite Figure 9 window, the non-full circuit has an effective fitted
exponent of approximately (2.7) for both probes. Separately, for each probe,
one accepted detector syndrome is compatible with both logical classes at
two-fault order. A decoder that receives only this syndrome must choose one
class and therefore fails on the other. This establishes a nonzero asymptotic
(O(p^2)) contribution for the authors' `full_post_selection=False` circuit.
Full postselection rejects these detector events and addresses a different
mode. The fresh high-statistics full-branch run independently reproduces the
approximately 31.8% retention at (p=10^{-3}) and gives six-point paper-style
fits of (\alpha_Z=3.077\pm0.131) and (\alpha_X=2.909\pm0.126), where the quoted
one-standard-deviation uncertainties come from counting-noise bootstrap
refits."""
    ),
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--execute",
    action="store_true",
    help="execute every cell from the audit directory before saving",
)
args = parser.parse_args()

if args.execute:
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(AUDIT_DIR)}},
    ).execute()

nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
