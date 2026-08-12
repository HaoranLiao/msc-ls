# Figure 9 postselection audit

This directory audits Figure 9 using the authors' released circuit builder at
commit `c89a53a7a7062b552d31f4c41ae2ba857756359a`. The upstream source files in
the parent directories are copied without modification.

The spreadsheet snapshot was exported from the Google Sheet linked by
`../fig/plot_error_detection.py`:

<https://docs.google.com/spreadsheets/d/1NV1PXI3OCJHO0QLHeb4nQcJ7v9jxXbYIJyWrJgQPSQg/edit?gid=573444527>

Run the complete audit from the repository root:

```bash
MPLCONFIGDIR=/tmp/mpl_msc \
XDG_CACHE_HOME=/tmp/xdg_msc \
NUMBA_CACHE_DIR=/tmp/numba_msc \
conda run -n msc python \
  references/hirano_code/audit/fig9_postselection_audit.py \
  --sample-shots 100000 \
  --rediscover
```

The script performs four independent checks:

1. Reads the chart title from the linked spreadsheet export.
2. Extracts the released `p=0.001` acceptance rates directly from
   `../fig/plot_error_detection.py`.
3. Builds the authors' `full_post_selection=False` and `True` circuits and
   prints both the total and postselected detector counts. It asserts that the
   full branch postselects every detector, then samples both branches.
4. Builds the authors' `full_post_selection=False` circuit, extracts every
   elementary fault effect from Stim's undecomposed detector error model, and
   verifies explicit pairs of two-fault combinations having the same accepted
   detector syndrome but opposite `L0` outcomes. With `--rediscover`, it also
   searches for such a witness from scratch.

The two-fault result is decoder-independent: one detector syndrome is
consistent with both logical classes, so no decoder receiving only those
detectors can correct every two-fault case.

The non-full-branch finite-window scaling sweep is implemented in
`false_branch_scaling.py`. Its recorded raw counts are stored in
`false_branch_scaling_results.csv`. A full rerun can require tens of millions
of shots:

```bash
MPLCONFIGDIR=/tmp/mpl_msc \
XDG_CACHE_HOME=/tmp/xdg_msc \
NUMBA_CACHE_DIR=/tmp/numba_msc \
conda run -n msc python \
  references/hirano_code/audit/false_branch_scaling.py \
  --target-errors 500 \
  --max-shots 5000000
```

The independent full-postselection sweep is implemented in
`true_branch_scaling.py`, with its high-statistics six-point results
stored in `true_branch_scaling_results.csv`. The recorded run used 500--506
logical errors per point and omitted the two smallest Figure 9 probabilities:

```bash
MPLCONFIGDIR=/tmp/mpl_msc \
XDG_CACHE_HOME=/tmp/xdg_msc \
conda run --no-capture-output -n msc python \
  references/hirano_code/audit/true_branch_scaling.py \
  --errors-per-shard 50 \
  --shards 10 \
  --max-shots-per-shard 8000000000 \
  --batch-shots 2000000 \
  --workers 8 \
  --min-p 0.0008 \
  --seed-base 90000
```

An executed notebook record containing the provenance check, a direct
eight-point non-full-branch simulation, the recorded high-statistics
full-postselection simulation and bootstrap fit, the search algorithm, and
explicit fault locations is available at
`fig9_postselection_and_fault_pairs.ipynb`. Rebuilding it runs tens of millions
of non-full-branch shots and may take several minutes:

```bash
MPLCONFIGDIR=/tmp/mpl_msc \
XDG_CACHE_HOME=/tmp/xdg_msc \
NUMBA_CACHE_DIR=/tmp/numba_msc \
IPYTHONDIR=/tmp/ipython_msc \
conda run -n msc python \
  references/hirano_code/audit/scripts/build_fig9_fault_pair_audit_notebook.py \
  --execute
```
