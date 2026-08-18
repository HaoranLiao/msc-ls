# Figure 9 postselection and scaling audit

This directory compares the authors' released `full_post_selection=False` and
`full_post_selection=True` circuits at commit
`c89a53a7a7062b552d31f4c41ae2ba857756359a`. The upstream source files in the
parent directories are copied without modification.

The spreadsheet snapshot was exported from the Google Sheet linked by
`../fig/plot_error_detection.py`:

<https://docs.google.com/spreadsheets/d/1NV1PXI3OCJHO0QLHeb4nQcJ7v9jxXbYIJyWrJgQPSQg/edit?gid=573444527>

## Contents

- `fig9_postselection_audit.py` checks the spreadsheet title, released
  acceptance rates, and detector postselection counts for both branches.
- `false_branch_scaling.py` runs the authors' non-full circuit with its hard
  postselection mask and PyMatching decoder.
- `true_branch_scaling.py` runs the authors' full-postselection circuit in
  parallel shards.
- `false_branch_scaling_results.csv` and `true_branch_scaling_results.csv`
  preserve the Monte Carlo counts used for the comparison.
- `fig9_postselection_and_scaling.ipynb` loads the CSV files, fits both branches
  to `p_L = C p^alpha`, and plots the data and fitted curves.
- `fig9_true_false_scaling.png` is the plot exported by the notebook.

The notebook uses the same unweighted nonlinear least-squares fit of
`C p^alpha` for both branches. The false branch has eight points from
`p=0.0005` through `0.002`; the higher-statistics true branch has six points
from `p=0.0008` through `0.002`. Quoted uncertainties are the standard
deviation of 20,000 Poisson counting-noise bootstrap refits.

## Postselection provenance

Run the lightweight provenance audit from the repository root:

```bash
MPLCONFIGDIR=/tmp/mpl_msc \
XDG_CACHE_HOME=/tmp/xdg_msc \
NUMBA_CACHE_DIR=/tmp/numba_msc \
conda run -n msc python \
  references/hirano_code/verify_results/fig9_postselection_audit.py \
  --sample-shots 100000
```

It verifies that the released `p=0.001` acceptance rate is consistent with the
full branch, where all 135 detectors are postselected, rather than the non-full
branch, where 40 of 231 detectors are postselected.

## Reproducing the Monte Carlo data

The non-full sweep can require tens of millions of shots:

```bash
MPLCONFIGDIR=/tmp/mpl_msc \
XDG_CACHE_HOME=/tmp/xdg_msc \
NUMBA_CACHE_DIR=/tmp/numba_msc \
conda run -n msc python \
  references/hirano_code/verify_results/false_branch_scaling.py \
  --target-errors 500 \
  --max-shots 5000000
```

The recorded full-postselection run used 500--506 logical errors per point:

```bash
MPLCONFIGDIR=/tmp/mpl_msc \
XDG_CACHE_HOME=/tmp/xdg_msc \
conda run --no-capture-output -n msc python \
  references/hirano_code/verify_results/true_branch_scaling.py \
  --errors-per-shard 50 \
  --shards 10 \
  --max-shots-per-shard 8000000000 \
  --batch-shots 2000000 \
  --workers 8 \
  --min-p 0.0008 \
  --seed-base 90000
```

The notebook reads the recorded CSVs and does not rerun either Monte Carlo
sweep.
