"""Visualization and exact-fault helpers for the Hirano MSC-LS audit notebook.

The functions here deliberately use Stim diagrams instead of printing circuits.
They are presentation/audit helpers; the protocol itself is still built by the
authors' unmodified ``src/lattice_surgery_error_detection.py`` implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from html import escape
from pathlib import Path
import re
import sys
import types

from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import pymatching
import stim


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
AUDIT = ROOT / "audit"
if "sinter" not in sys.modules:
    sys.modules["sinter"] = types.ModuleType("sinter")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(AUDIT) not in sys.path:
    sys.path.insert(0, str(AUDIT))

from lattice_surgery_error_detection import (  # noqa: E402
    InitialValue,
    QubitMapping,
    SteanePlusSurfaceCode,
    SteaneSyndromeExtractionPattern,
)

from fig9_postselection_audit import (  # noqa: E402
    KNOWN_AMBIGUITIES,
    FaultEffect,
    combine_pair,
    extract_fault_effects,
    mask_to_indices,
    verify_ambiguity,
)


STEANE_COORDS = {
    (3, 13): "St0",
    (1, 15): "St1",
    (6, 12): "St2",
    (5, 15): "St3",
    (2, 12): "St4",
    (3, 15): "St5",
    (2, 10): "St6",
    (5, 11): "St2' (temporary)",
}

COLOR_ANCILLA_COORDS = {
    (1, 13): "C0145 half A (RX/MX)",
    (2, 14): "C0145 half B (RZ/MZ)",
    (4, 14): "C0235 half A (RX/MX)",
    (5, 13): "C0235 half B (RZ/MZ)",
    (3, 11): "C0246 half A (RX/MX)",
    (4, 12): "C0246 half B (RZ/MZ)",
}

INTERFACE_COORDS = {
    (0, 16): "I: Z1a left ancilla",
    (2, 16): "I: Z1a right / demolition Xab",
    (4, 16): "I: Z35bc ancilla",
    (8, 16): "I: Zde ancilla",
    (6, 16): "surface top Xcd ancilla",
}


def build_protocol(
    probe: str,
    *,
    error_probability: float = 0.0,
    full_post_selection: bool = False,
) -> SteanePlusSurfaceCode:
    """Build the released distance-5/ZXZ Figure-9 protocol."""
    initial_value = InitialValue.Plus if probe == "+" else InitialValue.Zero
    wrapper = SteanePlusSurfaceCode(
        QubitMapping(30, 30),
        5,
        initial_value,
        SteaneSyndromeExtractionPattern.ZXZ,
        error_probability,
        full_post_selection,
    )
    wrapper.run()
    return wrapper


def qubit_id_by_coord(wrapper: SteanePlusSurfaceCode) -> dict[tuple[int, int], int]:
    return {coord: q for q, coord in wrapper.mapping.mapping}


def surface_data_label(coord: tuple[int, int]) -> str | None:
    x, y = coord
    if x not in range(1, 10, 2) or y not in range(17, 26, 2):
        return None
    row = (y - 17) // 2
    col = (x - 1) // 2
    if row == 0:
        return f"Su-{chr(ord('a') + col)}"
    return f"Su-r{row}c{col}"


def surface_support(measurement: object) -> list[tuple[int, int]]:
    """Return the data support of one Surface{X,Z}SyndromeMeasurement."""
    x, y = measurement.ancilla_position
    corners = {
        "left_top": (x - 1, y - 1),
        "left_bottom": (x - 1, y + 1),
        "right_top": (x + 1, y - 1),
        "right_bottom": (x + 1, y + 1),
    }
    selected = {
        "FOUR_WEIGHT": tuple(corners),
        "TWO_WEIGHT_UP": ("left_top", "right_top"),
        "TWO_WEIGHT_DOWN": ("left_bottom", "right_bottom"),
        "TWO_WEIGHT_LEFT": ("left_top", "left_bottom"),
        "TWO_WEIGHT_RIGHT": ("right_top", "right_bottom"),
    }[measurement.pattern.name]
    return [corners[name] for name in selected]


def surface_ancilla_roles(wrapper: SteanePlusSurfaceCode) -> dict[tuple[int, int], str]:
    roles: dict[tuple[int, int], str] = {}
    for coord, measurement in wrapper.surface_syndrome_measurements.items():
        kind = "X" if measurement.__class__.__name__.startswith("SurfaceX") else "Z"
        support = ",".join(
            (surface_data_label(pos) or str(pos)).removeprefix("Su-")
            for pos in surface_support(measurement)
        )
        roles[coord] = f"Su-{kind}[{support}]"
    return roles


def semantic_labels(wrapper: SteanePlusSurfaceCode) -> dict[int, str]:
    """Label every protocol qubit by code block and role."""
    by_coord = qubit_id_by_coord(wrapper)
    labels: dict[int, str] = {}
    for coord, label in STEANE_COORDS.items():
        labels[by_coord[coord]] = label
    for row in range(wrapper.surface_distance):
        for col in range(wrapper.surface_distance):
            coord = (wrapper.surface_offset_x + 2 * col, wrapper.surface_offset_y + 2 * row)
            labels[by_coord[coord]] = surface_data_label(coord) or f"Su-{coord}"
    for coord, label in surface_ancilla_roles(wrapper).items():
        labels[by_coord[coord]] = label
    for coord, label in COLOR_ANCILLA_COORDS.items():
        labels[by_coord[coord]] = label
    for coord, label in INTERFACE_COORDS.items():
        labels[by_coord[coord]] = label
    return labels


def qubit_legend(wrapper: SteanePlusSurfaceCode) -> pd.DataFrame:
    labels = semantic_labels(wrapper)
    coords = {
        q: tuple(int(value) for value in coord)
        for q, coord in wrapper.circuit.circuit.get_final_qubit_coordinates().items()
    }
    rows = []
    for q, label in labels.items():
        coord = coords[q]
        if label.startswith("St"):
            family = "Steane data"
        elif label.startswith("Su-r") or label.startswith("Su-") and len(label) == 4:
            family = "surface data"
        elif label.startswith("Su-") or label.startswith("surface"):
            family = "surface-check ancilla"
        elif label.startswith("C"):
            family = "color-check ancilla"
        else:
            family = "interface / demolition"
        rows.append({"q": q, "coordinate": coord, "label": label, "family": family})
    return pd.DataFrame(rows).sort_values(["family", "coordinate"]).reset_index(drop=True)


def plot_protocol_layout(
    wrapper: SteanePlusSurfaceCode,
    *,
    highlight: Iterable[tuple[int, int]] = (),
    xlim: tuple[float, float] = (-1, 11),
    ylim: tuple[float, float] = (27, 9),
) -> plt.Figure:
    """Draw a semantic layout; this complements Stim's gate-order diagrams."""
    highlighted = set(highlight)
    roles = surface_ancilla_roles(wrapper)
    fig, ax = plt.subplots(figsize=(12, 10))

    categories: list[tuple[str, dict[tuple[int, int], str], str, str, int]] = []
    steane = {coord: label for coord, label in STEANE_COORDS.items()}
    surface_data = {}
    for row in range(wrapper.surface_distance):
        for col in range(wrapper.surface_distance):
            coord = (wrapper.surface_offset_x + 2 * col, wrapper.surface_offset_y + 2 * row)
            surface_data[coord] = surface_data_label(coord) or str(coord)
    surface_x = {coord: label for coord, label in roles.items() if label.startswith("Su-X")}
    surface_z = {coord: label for coord, label in roles.items() if label.startswith("Su-Z")}
    categories.extend(
        [
            ("Steane data", steane, "o", "#d1495b", 170),
            ("surface data", surface_data, "o", "#2b6cb0", 140),
            ("surface X ancilla", surface_x, "s", "#ef767a", 105),
            ("surface Z ancilla", surface_z, "s", "#79c267", 105),
            ("color superdense ancilla", COLOR_ANCILLA_COORDS, "D", "#e0a82e", 120),
            ("interface / demolition", INTERFACE_COORDS, "P", "#7b61a8", 155),
        ]
    )

    for family, mapping, marker, color, size in categories:
        coords = list(mapping)
        ax.scatter(
            [coord[0] for coord in coords],
            [coord[1] for coord in coords],
            s=[size * (1.7 if coord in highlighted else 1) for coord in coords],
            marker=marker,
            color=color,
            edgecolors=["#111111" if coord in highlighted else "white" for coord in coords],
            linewidths=[2.5 if coord in highlighted else 0.8 for coord in coords],
            label=family,
            zorder=3,
        )
        for coord, label in mapping.items():
            short = label
            if family == "surface X ancilla":
                short = "X"
            elif family == "surface Z ancilla":
                short = "Z"
            elif family == "color superdense ancilla":
                short = label.split(" half")[0] + ("-A" if "half A" in label else "-B")
            elif family == "interface / demolition":
                short = {
                    (0, 16): "Z1a-L",
                    (2, 16): "Z1a-R / Xab",
                    (4, 16): "Z35bc",
                    (6, 16): "Xcd",
                    (8, 16): "Zde",
                }[coord]
            ax.annotate(short, coord, xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set(xlabel="lattice x", ylabel="lattice y")
    ax.set_xticks([tick for tick in range(0, 11) if min(xlim) <= tick <= max(xlim)])
    ax.set_yticks([tick for tick in range(10, 28) if min(ylim) <= tick <= max(ylim)])
    # Set limits after explicit ticks; Matplotlib otherwise expands a requested
    # witness close-up back out to include every full-layout tick.
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.18)
    ax.set_aspect("equal")
    title = "Hirano MSC-LS qubit roles"
    if highlighted:
        title += " (black outlines mark witness locations)"
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return fig


def interface_repetition_one_macro_table() -> pd.DataFrame:
    """Describe the deinterleaved operation-family view of Hirano round 1."""
    return pd.DataFrame(
        [
            {
                "macro-tick": "I",
                "only operation family shown": "interface ZZ checks",
                "checks / actions": "Z1a, Z35bc, Zde",
                "relation to executable schedule": (
                    "Uses Z1a-R at (2,16); Z1a-L is idle. Each ancilla circuit is physically spread over the six-layer round."
                ),
            },
            {
                "macro-tick": "S",
                "only operation family shown": "compatible surface checks",
                "checks / actions": "all ordinary surface X/Z checks except the conflicting top X checks",
                "relation to executable schedule": (
                    "Collapsed stabilizer view; reset, four CNOT sublayers, and readout are not shown separately."
                ),
            },
            {
                "macro-tick": "B",
                "only operation family shown": "color-ancilla Bell preparation",
                "checks / actions": "prepare the 0145, 0235, and 0246 two-ancilla pairs",
                "relation to executable schedule": "One conceptual preparation group.",
            },
            {
                "macro-tick": "Z",
                "only operation family shown": "Steane Z-check coupling",
                "checks / actions": "Z0145, Z0235, Z0246",
                "relation to executable schedule": (
                    "All data-to-ancilla couplings are collected here; hardware requires several collision-free sublayers."
                ),
            },
            {
                "macro-tick": "X",
                "only operation family shown": "Steane X-check coupling",
                "checks / actions": "X0235, X0246; X0145 is deliberately omitted",
                "relation to executable schedule": (
                    "All ancilla-to-data couplings are collected here and are pipelined into Hirano round 2."
                ),
            },
            {
                "macro-tick": "M",
                "only operation family shown": "color-ancilla disentangling/readout",
                "checks / actions": "MX/MZ ancilla readouts and postselected color detectors",
                "relation to executable schedule": "Readouts are staggered by the released pipeline.",
            },
        ]
    )


def hirano_interface_round_macro_table(round_number: int) -> pd.DataFrame:
    """Describe the barrier-collapsed protocol groups in a Hirano LS round."""
    if round_number == 1:
        table = interface_repetition_one_macro_table().copy()
        table.insert(0, "Hirano interface round", round_number)
        return table
    if round_number == 2:
        families = [
            ("I", "interface ZZ checks", "Z1a (left site), Z35bc, Zde"),
            ("S", "compatible surface checks", "all ordinary surface X/Z checks except top Xab/Xcd"),
            ("B", "color-ancilla Bell preparation", "prepare 0145, 0235, and 0246 pairs for the second direct color extraction"),
            ("Z", "Steane Z-check coupling", "Z0145, Z0235, Z0246"),
            ("X", "Steane X-check coupling", "none in the second direct color extraction"),
            ("M", "color-ancilla disentangling/readout", "MX/MZ readouts; retain the direct Z-check results"),
        ]
    elif round_number == 3:
        families = [
            ("I", "interface ZZ checks", "Z1a (right site), Z35bc, Zde"),
            ("S", "compatible surface checks", "all ordinary surface X/Z checks except top Xab/Xcd"),
            ("D", "Steane demolition", "destructively MX-measure Steane data 0–6 (with temporary 2′ bookkeeping)"),
            ("A", "demolition Xab ancilla", "RX q23; CX q23→a,b; MX q23"),
            ("P", "derive color-X checks", "X0145ab, X0235, X0246 and logical XL from demolition parities"),
            ("R", "recover top surface-X checks", "restart Xab and Xcd in the post-merge surface record"),
        ]
    else:
        raise ValueError("round_number must be 1, 2, or 3")
    return pd.DataFrame(
        [
            {
                "Hirano interface round": round_number,
                "macro-tick": tick,
                "only operation family shown": family,
                "checks / actions": actions,
            }
            for tick, family, actions in families
        ]
    )


def plot_grouped_hirano_interface_round(
    wrapper: SteanePlusSurfaceCode,
    round_number: int,
) -> plt.Figure:
    """Plot one Hirano lattice-surgery round as conceptual operation groups.

    These panels are intentionally *not* executable Stim TICKs. In particular,
    a panel may collect several CNOTs sharing an ancilla; it is an operation-
    family atlas that sits beside, not in place of, the literal Stim timeline.
    """
    if round_number not in (1, 2, 3):
        raise ValueError("round_number must be 1, 2, or 3")

    steane = {coord: label.removeprefix("St") for coord, label in STEANE_COORDS.items()}
    surface_data = {
        (wrapper.surface_offset_x + 2 * col, wrapper.surface_offset_y + 2 * row):
            (surface_data_label((wrapper.surface_offset_x + 2 * col, wrapper.surface_offset_y + 2 * row)) or "").removeprefix("Su-")
        for row in range(wrapper.surface_distance)
        for col in range(wrapper.surface_distance)
    }
    color_pairs = [
        ((1, 13), (2, 14), "0145"),
        ((4, 14), (5, 13), "0235"),
        ((3, 11), (4, 12), "0246"),
    ]

    z_connections = [
        ((3, 13), (2, 14), "0145"),
        ((1, 15), (2, 14), "0145"),
        ((2, 12), (1, 13), "0145"),
        ((3, 15), (2, 14), "0145"),
        ((3, 13), (4, 14), "0235"),
        ((5, 15), (4, 14), "0235"),
        ((3, 15), (4, 14), "0235"),
        ((6, 12), (5, 13), "0235"),
        ((3, 13), (4, 12), "0246"),
        ((5, 11), (4, 12), "0246"),
        ((2, 12), (3, 11), "0246"),
        ((2, 10), (3, 11), "0246"),
    ]
    x_connections = [
        ((4, 14), (3, 13), "0235"),
        ((4, 14), (5, 15), "0235"),
        ((4, 14), (3, 15), "0235"),
        ((5, 13), (6, 12), "0235"),
        ((4, 12), (3, 13), "0246"),
        ((4, 12), (5, 11), "0246"),
        ((3, 11), (2, 12), "0246"),
        ((3, 11), (2, 10), "0246"),
    ]
    check_colors = {"0145": "#d1495b", "0235": "#2b6cb0", "0246": "#6a4c93"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    def setup_axis(ax: plt.Axes, title: str, *, surface: bool = False) -> None:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)
        ax.set_xticks([])
        ax.set_yticks([])
        if surface:
            ax.set_xlim(-1, 11)
            ax.set_ylim(27, 14)
        else:
            ax.set_xlim(-1, 11)
            ax.set_ylim(18.5, 9)

    def draw_data(ax: plt.Axes, *, include_all_surface: bool = False) -> None:
        for coord, label in steane.items():
            ax.scatter(*coord, s=85, color="#d1495b", edgecolor="white", zorder=4)
            ax.annotate(label, coord, xytext=(4, 4), textcoords="offset points", fontsize=8)
        for coord, label in surface_data.items():
            if not include_all_surface and coord[1] != 17:
                continue
            ax.scatter(*coord, s=70, color="#2b6cb0", edgecolor="white", zorder=4)
            ax.annotate(label, coord, xytext=(4, 4), textcoords="offset points", fontsize=8)

    def draw_color_ancillas(ax: plt.Axes) -> None:
        for a, b, name in color_pairs:
            for coord, half in ((a, "A"), (b, "B")):
                ax.scatter(*coord, s=90, marker="D", color="#e0a82e", edgecolor="white", zorder=5)
                ax.annotate(f"{name}-{half}", coord, xytext=(4, 4), textcoords="offset points", fontsize=8)

    # I: the three conceptual merged-boundary parity checks.
    ax = axes[0, 0]
    setup_axis(ax, "Macro-tick I — interface ZZ checks only")
    draw_data(ax)
    left_coord = (0, 16) if round_number == 2 else (2, 16)
    left_name = "Z1a-L (q8)" if round_number == 2 else "Z1a-R (q23)"
    unused_coord = (2, 16) if round_number == 2 else (0, 16)
    unused_name = "Z1a-R q23 idle" if round_number == 2 else "Z1a-L q8 idle"
    interface_checks = [
        (left_coord, [(1, 15), (1, 17)], left_name, "#2f855a"),
        ((4, 16), [(3, 15), (5, 15), (3, 17), (5, 17)], "Z35bc (q38)", "#2f855a"),
        ((8, 16), [(7, 17), (9, 17)], "Zde (q68)", "#2f855a"),
    ]
    for ancilla, support, name, color in interface_checks:
        ax.scatter(*ancilla, marker="P", s=150, color="#7b61a8", edgecolor="white", zorder=6)
        ax.annotate(name, ancilla, xytext=(4, -14), textcoords="offset points", fontsize=9, fontweight="bold")
        for data in support:
            # Surface-Z extraction uses data as CX control and the syndrome
            # ancilla as target. Make this direction visually unmistakable.
            ax.annotate(
                "",
                xy=ancilla,
                xytext=data,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 2.8,
                    "mutation_scale": 24,
                    "shrinkA": 8,
                    "shrinkB": 9,
                },
                zorder=3,
            )
    ax.scatter(*unused_coord, marker="x", s=130, color="#777777", linewidth=2.5, zorder=6)
    ax.annotate(unused_name, unused_coord, xytext=(-5, -25), textcoords="offset points", fontsize=8)
    ax.annotate("active Z1a site", left_coord, xytext=(-10, 15), textcoords="offset points", fontsize=8)

    # S: collapse each compatible surface stabilizer into a single shaded support.
    ax = axes[0, 1]
    setup_axis(ax, "Macro-tick S — compatible surface checks only", surface=True)
    draw_data(ax, include_all_surface=True)
    for coord, measurement in wrapper.surface_syndrome_measurements.items():
        if coord[1] == 16:  # conflicting top X checks are absent during the merge
            continue
        support = surface_support(measurement)
        kind = "X" if measurement.__class__.__name__.startswith("SurfaceX") else "Z"
        color = "#ef767a" if kind == "X" else "#79c267"
        if len(support) >= 3:
            center = np.mean(np.asarray(support), axis=0)
            ordered = sorted(support, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
            ax.add_patch(Polygon(ordered, closed=True, facecolor=color, edgecolor=color, alpha=0.22, linewidth=1.2))
        else:
            ax.plot(
                [support[0][0], support[1][0]],
                [support[0][1], support[1][1]],
                color=color,
                linewidth=8,
                alpha=0.35,
                solid_capstyle="round",
            )
        ax.scatter(*coord, marker="s", s=46, color=color, edgecolor="white", zorder=5)
        ax.text(coord[0], coord[1], kind, fontsize=7, ha="center", va="center", zorder=6)
    ax.text(5, 15.2, "top Xab and Xcd checks removed during merge", ha="center", fontsize=8, color="#a33")

    if round_number == 3:
        # D: destructive X measurement of the Steane/color data.
        ax = axes[0, 2]
        setup_axis(ax, "Macro-tick D — Steane demolition MX only")
        draw_data(ax)
        for coord, label in steane.items():
            ax.annotate(
                "MX",
                coord,
                xytext=(0, -19),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#8b1e3f",
            )
        ax.text(
            5,
            17.7,
            "All Steane data outcomes, including temporary 2′, enter demolition parities",
            ha="center",
            fontsize=9,
        )

        # A: q23 is reused after its Z1a readout as the Xab ancilla.
        ax = axes[1, 0]
        setup_axis(ax, "Macro-tick A — demolition Xab ancilla only")
        draw_data(ax)
        demolition_ancilla = (2, 16)
        ax.scatter(*demolition_ancilla, marker="P", s=170, color="#7b61a8", edgecolor="white", zorder=6)
        ax.annotate("q23: RX … MX", demolition_ancilla, xytext=(-20, 16), textcoords="offset points", fontsize=9, fontweight="bold")
        for data in ((1, 17), (3, 17)):
            ax.annotate(
                "",
                xy=data,
                xytext=demolition_ancilla,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#7b61a8",
                    "lw": 3,
                    "mutation_scale": 25,
                    "shrinkA": 9,
                    "shrinkB": 8,
                },
            )
        ax.text(5, 17.9, "ancilla control → surface targets a,b; readout is Xab", ha="center", fontsize=9)

        # P: exact detector parities reconstructed from demolition outcomes.
        ax = axes[1, 1]
        setup_axis(ax, "Macro-tick P — derive color-X checks only")
        ax.axis("off")
        parity_text = (
            r"$X_{0235}=m_0\oplus m_2\oplus m_{2'}\oplus m_3\oplus m_5$" "\n\n"
            r"$X_{0246}=m_0\oplus m_2\oplus m_{2'}\oplus m_4\oplus m_6$" "\n\n"
            r"$X_{0145ab}=m_0\oplus m_1\oplus m_4\oplus m_5\oplus m_{ab}$" "\n\n"
            r"$X_L=m_1\oplus m_4\oplus m_6$"
        )
        ax.text(0.5, 0.58, parity_text, transform=ax.transAxes, ha="center", va="center", fontsize=15)
        ax.text(0.5, 0.14, "The first three parities are postselected.", transform=ax.transAxes, ha="center", fontsize=10)

        # R: post-merge recovery of the two top surface-X checks.
        ax = axes[1, 2]
        setup_axis(ax, "Macro-tick R — recover top X checks only")
        draw_data(ax)
        for ancilla, support, name in [
            ((2, 16), [(1, 17), (3, 17)], "Xab (q23)"),
            ((6, 16), [(5, 17), (7, 17)], "Xcd (q53)"),
        ]:
            ax.scatter(*ancilla, marker="P", s=160, color="#ef767a", edgecolor="white", zorder=6)
            ax.annotate(name, ancilla, xytext=(-12, 15), textcoords="offset points", fontsize=9, fontweight="bold")
            for data in support:
                ax.annotate(
                    "",
                    xy=data,
                    xytext=ancilla,
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#c2415d",
                        "lw": 3,
                        "mutation_scale": 25,
                        "shrinkA": 9,
                        "shrinkB": 8,
                    },
                )
        ax.text(5, 17.9, "surface-X ancilla controls → data targets", ha="center", fontsize=9)

        fig.suptitle(
            "Hirano interface round 3, barrier-collapsed conceptual macro-ticks\n"
            "This round ends with Steane demolition and recovery of the surface top edge.",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig

    # B: Bell-pair preparation, without data couplings.
    ax = axes[0, 2]
    setup_axis(ax, "Macro-tick B — prepare color Bell pairs only")
    draw_data(ax)
    draw_color_ancillas(ax)
    for a, b, name in color_pairs:
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops={
                "arrowstyle": "-|>",
                "color": check_colors[name],
                "lw": 2.8,
                "mutation_scale": 24,
                "shrinkA": 8,
                "shrinkB": 8,
            },
        )
        midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ax.annotate("CX", midpoint, xytext=(3, -10), textcoords="offset points", fontsize=8)
    ax.text(5, 17.7, "A: RX / later MX     B: RZ / later MZ", ha="center", fontsize=9)

    # Z: collect all data-to-ancilla couplings.
    ax = axes[1, 0]
    setup_axis(ax, "Macro-tick Z — Steane Z-check coupling only")
    draw_data(ax)
    draw_color_ancillas(ax)
    for data, ancilla, name in z_connections:
        ax.annotate(
            "",
            xy=ancilla,
            xytext=data,
            arrowprops={
                "arrowstyle": "-|>",
                "color": check_colors[name],
                "lw": 2.2,
                "alpha": 0.85,
                "mutation_scale": 21,
                "shrinkA": 7,
                "shrinkB": 7,
            },
        )
    ax.text(5, 17.7, "Z0145 (red), Z0235 (blue), Z0246 (purple)", ha="center", fontsize=9)

    # X: collect all ancilla-to-data couplings; X0145 is intentionally absent.
    ax = axes[1, 1]
    setup_axis(ax, "Macro-tick X — Steane X-check coupling only")
    draw_data(ax)
    draw_color_ancillas(ax)
    if round_number == 1:
        for ancilla, data, name in x_connections:
            ax.annotate(
                "",
                xy=data,
                xytext=ancilla,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": check_colors[name],
                    "lw": 2.2,
                    "alpha": 0.85,
                    "mutation_scale": 21,
                    "shrinkA": 7,
                    "shrinkB": 7,
                },
            )
        ax.plot([0.8, 2.3], [12.8, 14.2], color="#555555", linestyle="--", linewidth=1.8)
        ax.text(0.7, 12.2, "X0145 omitted", fontsize=9, color="#555555")
        ax.text(5, 17.7, "X0235 (blue), X0246 (purple)", ha="center", fontsize=9)
    else:
        ax.text(
            5,
            13.8,
            "No direct Steane X stabilizer\nis coupled in color-syndrome round 2",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#555555",
        )
        ax.text(5, 17.7, "The missing X checks are supplied by round-3 demolition", ha="center", fontsize=9)

    # M: show readout bases and the staggered/pipelined nature of the record.
    ax = axes[1, 2]
    setup_axis(ax, "Macro-tick M — disentangle/read out only")
    draw_data(ax)
    draw_color_ancillas(ax)
    for a, b, name in color_pairs:
        ax.annotate("MX", a, xytext=(-8, 13), textcoords="offset points", fontsize=9, fontweight="bold")
        ax.annotate("MZ", b, xytext=(5, -15), textcoords="offset points", fontsize=9, fontweight="bold")
        ax.plot([a[0], b[0]], [a[1], b[1]], color=check_colors[name], linewidth=2, alpha=0.65)
    ax.text(
        5,
        17.35,
        (
            "Color outcomes are postselected. 0145 is read first;\n"
            "0235/0246 finish later because the source pipeline overlaps Hirano round 2."
            if round_number == 1
            else "Second direct color extraction: retain Z0145, Z0235, Z0246.\nThe released source pipelines these readouts into Hirano round 3."
        ),
        ha="center",
        fontsize=8.5,
    )

    fig.suptitle(
        f"Hirano interface round {round_number}, barrier-collapsed conceptual macro-ticks\n"
        "Each panel contains one operation family; panels are not executable hardware TICKs.",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_grouped_interface_repetition_one(
    wrapper: SteanePlusSurfaceCode,
) -> plt.Figure:
    """Backward-compatible name for the round-1 grouped plot."""
    return plot_grouped_hirano_interface_round(wrapper, 1)


def _active_qubits(circuit: stim.Circuit) -> set[int]:
    active: set[int] = set()
    for instruction in circuit.flattened():
        if instruction.name == "QUBIT_COORDS":
            continue
        for target in instruction.targets_copy():
            if target.is_qubit_target or target.is_x_target or target.is_y_target or target.is_z_target:
                active.add(target.qubit_value)
    return active


def _remap_target(target: stim.GateTarget, mapping: dict[int, int]) -> stim.GateTarget:
    if target.is_x_target:
        return stim.target_x(mapping[target.qubit_value])
    if target.is_y_target:
        return stim.target_y(mapping[target.qubit_value])
    if target.is_z_target:
        return stim.target_z(mapping[target.qubit_value])
    if target.is_qubit_target:
        q = mapping[target.qubit_value]
        return stim.target_inv(q) if target.is_inverted_result_target else stim.GateTarget(q)
    return target


def compact_for_diagram(
    circuit: stim.Circuit,
    labels: dict[int, str],
) -> tuple[stim.Circuit, dict[int, str]]:
    """Remove 390+ coordinate-only lines and densely remap active qubits."""
    flat = circuit.flattened()
    active = _active_qubits(flat)
    ordered = sorted(active, key=lambda q: (labels.get(q, "~"), q))
    old_to_new = {old: new for new, old in enumerate(ordered)}
    coordinates = flat.get_final_qubit_coordinates()
    out = stim.Circuit()
    for old in ordered:
        if old in coordinates:
            out.append("QUBIT_COORDS", old_to_new[old], coordinates[old])
    for instruction in flat:
        if instruction.name == "QUBIT_COORDS":
            continue
        targets = [_remap_target(target, old_to_new) for target in instruction.targets_copy()]
        out.append(
            instruction.name,
            targets,
            instruction.gate_args_copy(),
            tag=instruction.tag,
        )
    new_labels = {
        old_to_new[old]: labels.get(old, f"q{old}")
        for old in ordered
    }
    return out, new_labels


def show_timeline(
    circuit: stim.Circuit,
    labels: dict[int, str],
    *,
    ticks: range | Sequence[int] | None = None,
    zoom: float = 1.0,
    max_height: int = 760,
    annotation_font_size: int = 15,
) -> HTML:
    """Render a white-background Stim timeline with semantic qubit labels."""
    compact, compact_labels = compact_for_diagram(circuit, labels)
    diagram = compact.diagram("timeline-svg", tick=ticks) if ticks is not None else compact.diagram("timeline-svg")
    svg = str(diagram)
    svg = svg.replace('font-size="8"', f'font-size="{annotation_font_size}"')
    for q, label in sorted(compact_labels.items(), reverse=True):
        svg = svg.replace(f">q{q}</text>", f">{escape(label)}</text>")
    # Stim reserves enough left margin for labels such as q12, but not for our
    # semantic names. Extend the viewBox into negative x so those names are not
    # clipped while leaving the circuit geometry unchanged.
    match = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    if match:
        width, height = (int(value) for value in match.groups())
        left_padding = max(220, max(map(len, compact_labels.values()), default=0) * 8)
        svg = svg.replace(
            match.group(0),
            f'viewBox="{-left_padding} 0 {width + left_padding} {height}"',
            1,
        )
    return HTML(
        '<div style="background:white;padding:12px;border:1px solid #ddd;'
        f'max-height:{max_height}px;overflow:auto">'
        f'<div style="width:{zoom * 100:.0f}%">{svg}</div></div>'
    )


def show_timeslices(
    circuit: stim.Circuit,
    labels: dict[int, str],
    *,
    ticks: range | Sequence[int],
    zoom: float = 1.0,
    max_height: int = 760,
) -> HTML:
    """Render Stim's tick-by-tick geometry view of the active protocol qubits."""
    compact, _ = compact_for_diagram(circuit, labels)
    svg = str(compact.diagram("timeslice-svg", tick=ticks))
    return HTML(
        '<div style="background:white;padding:12px;border:1px solid #ddd;'
        f'max-height:{max_height}px;overflow:auto">'
        f'<div style="width:{zoom * 100:.0f}%">{svg}</div></div>'
    )


def _measurement_count(instruction: stim.CircuitInstruction) -> int:
    if instruction.name in {"M", "MX", "MY", "MR", "MRX", "MRY"}:
        return sum(target.is_qubit_target for target in instruction.targets_copy())
    if instruction.name == "MPP":
        count = 0
        previous_was_combiner = False
        for target in instruction.targets_copy():
            if target.is_combiner:
                previous_was_combiner = True
            else:
                if not previous_was_combiner:
                    count += 1
                previous_was_combiner = False
        return count
    return 0


def detector_registry(
    wrapper: SteanePlusSurfaceCode,
    detector_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Resolve D numbers into measurement records, ticks, qubits, and roles."""
    circuit = wrapper.circuit.circuit.flattened()
    wanted = set(detector_ids) if detector_ids is not None else None
    postselected = {item.id for item in wrapper.circuit.detectors_for_post_selection}
    coords = {
        q: tuple(int(value) for value in coord)
        for q, coord in circuit.get_final_qubit_coordinates().items()
    }
    labels = semantic_labels(wrapper)
    measurement_registry: dict[int, dict[str, object]] = {}
    measurement_index = 0
    detector_index = 0
    ticks_elapsed = 0
    rows: list[dict[str, object]] = []
    for instruction_offset, instruction in enumerate(circuit):
        if instruction.name == "TICK":
            ticks_elapsed += 1
            continue
        count = _measurement_count(instruction)
        if count:
            qubits = [
                target.qubit_value
                for target in instruction.targets_copy()
                if target.is_qubit_target
            ]
            for local_index in range(count):
                q = qubits[local_index] if local_index < len(qubits) else None
                measurement_registry[measurement_index] = {
                    "measurement": measurement_index,
                    "gate": instruction.name,
                    "q": q,
                    "coordinate": coords.get(q),
                    "role": labels.get(q, f"q{q}") if q is not None else "Pauli product",
                    "tick": ticks_elapsed,
                    "instruction": instruction_offset,
                }
                measurement_index += 1
        if instruction.name != "DETECTOR":
            continue
        if wanted is None or detector_index in wanted:
            record_ids = [
                measurement_index + target.value
                for target in instruction.targets_copy()
                if target.is_measurement_record_target
            ]
            records = [measurement_registry[index] for index in record_ids]
            rows.append(
                {
                    "detector": f"D{detector_index}",
                    "tag": instruction.tag or "untagged",
                    "postselected": detector_index in postselected,
                    "declared_tick": ticks_elapsed,
                    "measurement_ids": record_ids,
                    "measurement_ticks": [record["tick"] for record in records],
                    "measurement_gates": [record["gate"] for record in records],
                    "qubits": [record["q"] for record in records],
                    "coordinates": [record["coordinate"] for record in records],
                    "roles": [record["role"] for record in records],
                }
            )
        detector_index += 1
    return pd.DataFrame(rows)


def _pauli_product_text(location: stim.CircuitErrorLocation, labels: dict[int, str]) -> str:
    pieces = []
    for term in location.flipped_pauli_product:
        target = term.gate_target
        pieces.append(f"{target.pauli_type}{labels.get(target.qubit_value, f'q{target.qubit_value}')}@{tuple(int(x) for x in term.coords)}")
    return " * ".join(pieces) if pieces else str(location.flipped_measurement)


def fault_table(
    wrapper: SteanePlusSurfaceCode,
    effects: list[FaultEffect],
    effect_indices: Iterable[int],
) -> pd.DataFrame:
    labels = semantic_labels(wrapper)
    rows = []
    for effect_index in effect_indices:
        effect = effects[effect_index]
        location = effect.explanation.circuit_error_locations[0]
        rows.append(
            {
                "effect": effect_index,
                "detectors": [f"D{i}" for i in mask_to_indices(effect.detector_mask)],
                "L0": effect.logical_mask,
                "after_TICKs": location.tick_offset,
                "noise_instruction": str(location.instruction_targets),
                "representative_Pauli": _pauli_product_text(location, labels),
                "instruction_offset": location.stack_frames[0].instruction_offset,
            }
        )
    return pd.DataFrame(rows)


NOISE_INSTRUCTIONS = {
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "PAULI_CHANNEL_1",
    "PAULI_CHANNEL_2",
    "CORRELATED_ERROR",
    "ELSE_CORRELATED_ERROR",
    "HERALDED_ERASE",
    "HERALDED_PAULI_CHANNEL_1",
}


def deterministic_fault_circuit(
    circuit: stim.Circuit,
    effects: list[FaultEffect],
    effect_indices: Iterable[int],
) -> stim.Circuit:
    """Strip stochastic noise and insert each Stim representative as Pauli gates."""
    insertions: dict[int, list[stim.GateTargetWithCoords]] = {}
    for effect_index in effect_indices:
        location = effects[effect_index].explanation.circuit_error_locations[0]
        offset = location.stack_frames[0].instruction_offset
        insertions.setdefault(offset, []).extend(location.flipped_pauli_product)

    out = stim.Circuit()
    for instruction_offset, instruction in enumerate(circuit.flattened()):
        if instruction.name not in NOISE_INSTRUCTIONS:
            out.append(instruction)
        for term in insertions.get(instruction_offset, []):
            target = term.gate_target
            out.append(target.pauli_type, target.qubit_value, tag="DETERMINISTIC-WITNESS")
    return out


def verify_deterministic_pair(
    circuit: stim.Circuit,
    effects: list[FaultEffect],
    pair: tuple[int, int],
) -> tuple[list[int], int]:
    injected = deterministic_fault_circuit(circuit, effects, pair)
    noiseless_reference = deterministic_fault_circuit(circuit, effects, [])
    # A detector sampler would choose a new reference sample that already
    # contains the deterministic Pauli gates. Convert the injected measurement
    # record with the *unfaulted* circuit's reference instead.
    measurements = injected.compile_sampler(seed=1).sample(shots=1)
    detection_events, observable_flips = noiseless_reference.compile_m2d_converter().convert(
        measurements=measurements,
        separate_observables=True,
    )
    found_detectors = np.flatnonzero(detection_events[0]).tolist()
    found_logical = int(observable_flips[0, 0])
    expected_detector_mask, expected_logical = combine_pair(effects, pair)
    if found_detectors != mask_to_indices(expected_detector_mask):
        raise AssertionError((found_detectors, mask_to_indices(expected_detector_mask)))
    if found_logical != expected_logical:
        raise AssertionError((found_logical, expected_logical))
    return found_detectors, found_logical


def witness_summary(probe: str) -> tuple[SteanePlusSurfaceCode, list[FaultEffect], pd.DataFrame]:
    """Build, verify, decode, and tabulate the saved exact two-fault witness."""
    wrapper = build_protocol(probe, error_probability=0.001, full_post_selection=False)
    circuit = wrapper.circuit.circuit
    effects = extract_fault_effects(circuit)
    pairs = KNOWN_AMBIGUITIES[probe]
    postselection_mask = sum(1 << item.id for item in wrapper.circuit.detectors_for_post_selection)
    syndrome_mask, logical_difference = verify_ambiguity(effects, postselection_mask, pairs)
    if logical_difference != 1:
        raise AssertionError(logical_difference)
    syndrome = np.zeros(circuit.num_detectors, dtype=np.uint8)
    syndrome[mask_to_indices(syndrome_mask)] = 1
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    decoder_prediction = int(matching.decode(syndrome)[0])
    rows = []
    for name, pair in zip(("A", "B"), pairs):
        detectors, logical = verify_deterministic_pair(circuit, effects, pair)
        rows.append(
            {
                "pair": name,
                "effect_indices": pair,
                "common_accepted_syndrome": [f"D{i}" for i in detectors],
                "actual_L0": logical,
                "PyMatching_prediction": decoder_prediction,
                "decoder_result": "correct" if logical == decoder_prediction else "logical failure",
            }
        )
    return wrapper, effects, pd.DataFrame(rows)


def single_fault_decoder_failures(
    wrapper: SteanePlusSurfaceCode,
    effects: list[FaultEffect],
) -> list[int]:
    """Return accepted elementary effects that PyMatching decodes incorrectly."""
    circuit = wrapper.circuit.circuit
    postselection = sum(1 << item.id for item in wrapper.circuit.detectors_for_post_selection)
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    failures = []
    for index, effect in enumerate(effects):
        if effect.detector_mask & postselection:
            continue
        syndrome = np.zeros(circuit.num_detectors, dtype=np.uint8)
        syndrome[mask_to_indices(effect.detector_mask)] = 1
        prediction = int(matching.decode(syndrome)[0])
        if prediction != effect.logical_mask:
            failures.append(index)
    return failures
