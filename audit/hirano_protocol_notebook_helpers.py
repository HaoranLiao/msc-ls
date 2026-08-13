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
                "only operation family shown": "color-pair preparation and initial q2/q2′ gates",
                "checks / actions": (
                    "prepare the 0145, 0235, and 0246 ancilla pairs; RZ 2′, CX 2→2′, CX 2′→2, RX 2"
                ),
                "relation to executable schedule": "Opposed q2/q2′ CNOTs are shown as separate directional arrows.",
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
                "only operation family shown": "color readout and final q2/q2′ gates",
                "checks / actions": "MX/MZ color readouts; CX 2→2′, CX 2′→2, MX 2′",
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
            (
                "B",
                "color-pair preparation and fresh 2/2′ copy",
                "prepare 0145, 0235, and 0246 pairs; RZ 2′ and CX 2→2′ for the second extraction",
            ),
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
    *,
    detector_overlay: bool = False,
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
    if detector_overlay:
        detector_text = {
            1: "q24: m29\nD16 = m5 ⊕ m29\n(start D46)",
            2: "q24: m61\nD46 = m29 ⊕ m61\n(start D77)",
            3: "q24: m100\nD77 = m61 ⊕ m100",
        }[round_number]
        ax.scatter(2, 18, s=260, facecolor="none", edgecolor="#111111", linewidth=3, zorder=9)
        ax.annotate(
            detector_text,
            (2, 18),
            xytext=(28, 15),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#111111",
            arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1.5},
        )
        if round_number in (1, 2):
            history_label = "A1: X after q24 RZ" if round_number == 1 else "A2: X after q24 RZ"
            ax.scatter(2, 18, s=80, marker="*", color="#dc2626", zorder=10)
            ax.annotate(history_label, (2, 18), xytext=(30, -35), textcoords="offset points", fontsize=8.5, color="#b91c1c", fontweight="bold")

    if detector_overlay and round_number == 3:
        # Competing zero-probe history B has an X component on surface data a
        # after the interface CX a -> q23.
        ax = axes[0, 0]
        ax.scatter(1, 17, s=100, marker="*", color="#dc2626", zorder=10)
        ax.annotate(
            "B2: X on a after\nCX a→q23",
            (1, 17),
            xytext=(18, -38),
            textcoords="offset points",
            fontsize=8.5,
            color="#b91c1c",
            fontweight="bold",
        )

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

        # The sixth grid cell is deliberately empty. Recovery is drawn only
        # in the post-interface atlas, so no recovery operation is duplicated.
        ax = axes[1, 2]
        ax.axis("off")
        ax.set_title("End of interface round 3", fontsize=12, fontweight="bold")
        ax.text(
            0.5,
            0.60,
            "No operation is assigned to this panel.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.36,
            "The next physical operations appear only in\n"
            "post-interface macro-tick R1 below.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.text(
            0.5,
            0.12,
            "Empty by design: prevents duplicating the recovery circuit.",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            color="#555555",
        )

        fig.suptitle(
            "Hirano interface round 3, barrier-collapsed conceptual macro-ticks\n"
            "This round ends with Steane demolition; surface recovery follows afterward.",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig

    # B owns the three color-ancilla Bell-pair preparations and the first two
    # direct q2/q2' CNOTs. The latter are drawn as ordinary directional gates.
    ax = axes[0, 2]
    setup_axis(
        ax,
        "Macro-tick B — pair preparation / q2↔q2′ gates"
        if round_number == 1
        else "Macro-tick B — color pairs + fresh q2/q2′ Z-copy",
    )
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
    if round_number == 1:
        q2 = (6, 12)
        q2_prime = (5, 11)
        for start, end, bend in [
            (q2, q2_prime, 0.25),
            (q2_prime, q2, 0.25),
        ]:
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#00897b",
                    "lw": 3,
                    "mutation_scale": 25,
                    "shrinkA": 8,
                    "shrinkB": 8,
                    "connectionstyle": f"arc3,rad={bend}",
                },
            )
        ax.annotate("RZ", q2_prime, xytext=(-20, 10), textcoords="offset points", fontsize=9, fontweight="bold", color="#00695c")
        ax.annotate("RX", q2, xytext=(8, -15), textcoords="offset points", fontsize=9, fontweight="bold", color="#00695c")
    else:
        ax.annotate(
            "",
            xy=(5, 11),
            xytext=(6, 12),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#00897b",
                "lw": 3,
                "mutation_scale": 25,
                "shrinkA": 8,
                "shrinkB": 8,
            },
        )
        ax.annotate(
            "RZ 2′; CX 2→2′\nfresh copy for extraction 2",
            (5.5, 11.5),
            xytext=(10, -3),
            textcoords="offset points",
            fontsize=8,
            color="#00695c",
            fontweight="bold",
        )
    ax.text(
        5,
        17.7,
        "Color ancillas: A is RX; B is RZ"
        if round_number == 1
        else "Color pairs plus source's fresh RZ 2′; CX 2→2′",
        ha="center",
        fontsize=9,
    )

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
    if round_number == 1:
        ax.annotate(
            "q2 → 0235-B",
            (5.5, 12.6),
            fontsize=8.5,
            color=check_colors["0235"],
            fontweight="bold",
        )
        ax.annotate(
            "q2′ → 0246-B",
            (4.2, 11.35),
            fontsize=8.5,
            color=check_colors["0246"],
            fontweight="bold",
        )

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
        ax.annotate(
            "0235-B → q2",
            (5.35, 12.7),
            fontsize=8.5,
            color=check_colors["0235"],
            fontweight="bold",
        )
        ax.annotate(
            "0246-B → q2′",
            (4.05, 11.25),
            fontsize=8.5,
            color=check_colors["0246"],
            fontweight="bold",
        )
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
    setup_axis(
        ax,
        "Macro-tick M — color readout / q2↔q2′ gates"
        if round_number == 1
        else "Macro-tick M — disentangle/read out only",
    )
    draw_data(ax)
    draw_color_ancillas(ax)
    for a, b, name in color_pairs:
        ax.annotate("MX", a, xytext=(-8, 13), textcoords="offset points", fontsize=9, fontweight="bold")
        ax.annotate("MZ", b, xytext=(5, -15), textcoords="offset points", fontsize=9, fontweight="bold")
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops={
                "arrowstyle": "-|>",
                "color": check_colors[name],
                "lw": 2.8,
                "alpha": 0.85,
                "mutation_scale": 24,
                "shrinkA": 8,
                "shrinkB": 8,
            },
        )
        midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ax.annotate("CX", midpoint, xytext=(3, -10), textcoords="offset points", fontsize=8)
    if round_number == 1:
        q2 = (6, 12)
        q2_prime = (5, 11)
        for start, end, bend in [
            (q2, q2_prime, -0.25),
            (q2_prime, q2, -0.25),
        ]:
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#00897b",
                    "lw": 3,
                    "mutation_scale": 25,
                    "shrinkA": 8,
                    "shrinkB": 8,
                    "connectionstyle": f"arc3,rad={bend}",
                },
            )
        ax.annotate("MX", q2_prime, xytext=(7, 10), textcoords="offset points", fontsize=9, fontweight="bold", color="#00695c")
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
        + (
            "I/S belong to merged-boundary round 1; B/Z/X/M own logical color extraction 1, which is source-pipelined."
            if round_number == 1
            else "Each panel contains one operation family; panels are not executable hardware TICKs."
        ),
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


def q2_prime_lifecycle_table() -> pd.DataFrame:
    """Exact q2/q2' gates in released ZXZ source order."""
    return pd.DataFrame(
        [
            (1, "first Z→X extraction", "RZ 2′", 1380, "initialize the temporary site"),
            (2, "first Z→X extraction", "CX 2→2′", 1395, "first half of moving encoded data to 2′"),
            (3, "first Z→X extraction", "CX 2′→2", 1426, "complete the move to 2′"),
            (4, "first Z→X extraction", "RX 2", 1442, "prepare released site 2 for X-check couplings"),
            (5, "first Z→X extraction", "CX 2→2′", 1478, "first half of moving encoded data back to 2"),
            (6, "first Z→X extraction", "CX 2′→2", 1510, "complete the move back to 2"),
            (7, "first Z→X extraction", "MX 2′ + detector", 1530, "check/postselect the released temporary site"),
            (8, "second Z extraction", "RZ 2′", 1552, "fresh temporary-site initialization"),
            (9, "second Z extraction", "CX 2→2′", 1574, "copy X-error information for the second extraction"),
            (10, "demolition", "MX 2", 1635, "destructive data outcome"),
            (11, "demolition", "MX 2′", 1636, "destructive temporary-copy outcome"),
        ],
        columns=["order", "logical owner", "released-code operation", "source line", "role"],
    )


def q2_prime_emitted_sequence(wrapper: SteanePlusSurfaceCode) -> list[str]:
    """Extract q2/q2' resets, readouts, and mutual CXs from emitted Stim."""
    ids = qubit_id_by_coord(wrapper)
    q2 = ids[(6, 12)]
    q2_prime = ids[(5, 11)]
    name = {q2: "2", q2_prime: "2′"}
    result: list[str] = []
    for instruction in wrapper.circuit.circuit.flattened():
        targets = instruction.targets_copy()
        if instruction.name == "CX":
            for control, target in zip(targets[::2], targets[1::2]):
                if {control.qubit_value, target.qubit_value} == {q2, q2_prime}:
                    result.append(f"CX {name[control.qubit_value]}→{name[target.qubit_value]}")
        elif instruction.name in {"R", "RX", "M", "MX"}:
            for target in targets:
                if target.qubit_value in name:
                    gate = "RZ" if instruction.name == "R" else instruction.name
                    result.append(f"{gate} {name[target.qubit_value]}")
    return result


def emitted_operation_census(wrapper: SteanePlusSurfaceCode) -> pd.DataFrame:
    """Count primitive operations in the noiseless emitted Stim circuit.

    Batched quantum instructions are counted per target (or per target pair
    for two-qubit gates); DETECTOR and OBSERVABLE_INCLUDE are counted per
    instruction. Coordinate declarations and TICK separators are excluded.
    """
    counts: dict[str, int] = {}
    two_qubit = {"CX", "CY", "CZ", "SWAP", "ISWAP", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ"}
    classical = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}
    for instruction in wrapper.circuit.circuit.flattened():
        name = instruction.name
        if name in {"QUBIT_COORDS", "TICK"}:
            continue
        targets = instruction.targets_copy()
        if name in two_qubit:
            amount = len(targets) // 2
        elif name in classical:
            amount = 1
        else:
            amount = sum(
                target.is_qubit_target
                or target.is_x_target
                or target.is_y_target
                or target.is_z_target
                for target in targets
            )
            amount = amount or 1
        counts[name] = counts.get(name, 0) + amount
    family = {
        "R": "quantum reset",
        "RX": "quantum reset",
        "RY": "quantum reset",
        "CX": "quantum two-qubit gate",
        "M": "quantum measurement",
        "MX": "quantum measurement",
        "MY": "quantum measurement",
        "DETECTOR": "classical record annotation",
        "OBSERVABLE_INCLUDE": "classical record annotation",
    }
    return pd.DataFrame(
        [
            {"Stim operation": name, "primitive count": count, "kind": family.get(name, "other")}
            for name, count in sorted(counts.items())
        ]
    )


def plot_q2_prime_exact_lifecycle() -> plt.Figure:
    """Circuit-like strip of every direct q2/q2' operation in source order."""
    operations = [
        (1, "RZ", "2′", None),
        (2, "CX", "2", "2′"),
        (3, "CX", "2′", "2"),
        (4, "RX", "2", None),
        (5, "CX", "2", "2′"),
        (6, "CX", "2′", "2"),
        (7, "MX", "2′", None),
        (8, "RZ", "2′", None),
        (9, "CX", "2", "2′"),
        (10, "MX", "2", None),
        (11, "MX", "2′", None),
    ]
    ypos = {"2": 1.0, "2′": 0.0}
    fig, ax = plt.subplots(figsize=(18, 5.2))
    ax.axvspan(0.5, 7.5, color="#e0f2fe", alpha=0.55)
    ax.axvspan(7.5, 9.5, color="#fef3c7", alpha=0.65)
    ax.axvspan(9.5, 11.5, color="#fce7f3", alpha=0.65)
    for name, y in ypos.items():
        ax.hlines(y, 0.5, 11.5, color="#334155", linewidth=2)
        ax.text(0.25, y, f"q{name}", ha="right", va="center", fontsize=13, fontweight="bold")
    for x, gate, first, second in operations:
        if gate == "CX":
            yc, yt = ypos[first], ypos[second]
            ax.scatter(x, yc, s=70, color="#00695c", zorder=5)
            ax.annotate(
                "",
                xy=(x, yt),
                xytext=(x, yc),
                arrowprops={"arrowstyle": "-|>", "color": "#00897b", "lw": 2.8, "mutation_scale": 22, "shrinkA": 5, "shrinkB": 5},
                zorder=4,
            )
            ax.text(x, 1.31, f"{x}. CX {first}→{second}", rotation=38, ha="left", va="bottom", fontsize=8.5)
        else:
            y = ypos[first]
            marker = "s" if gate.startswith("R") else "D"
            color = "#2563eb" if gate.startswith("R") else "#be123c"
            ax.scatter(x, y, s=105, marker=marker, color=color, edgecolor="white", zorder=5)
            ax.text(x, y + (0.20 if y == 1 else -0.22), gate, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
            ax.text(x, 1.31, f"{x}. {gate} {first}", rotation=38, ha="left", va="bottom", fontsize=8.5)
    ax.axvline(7.5, color="#777777", linestyle="--", linewidth=1.4)
    ax.axvline(9.5, color="#777777", linestyle="--", linewidth=1.4)
    ax.text(4, -0.62, "first Z→X extraction: move out, move back, then check 2′", ha="center", fontsize=11, fontweight="bold")
    ax.text(8.5, -0.62, "second Z extraction:\nfresh copy", ha="center", fontsize=10, fontweight="bold")
    ax.text(10.5, -0.62, "demolition:\nmeasure both", ha="center", fontsize=10, fontweight="bold")
    ax.text(
        6,
        -1.02,
        "Exact released-code q2/q2′ gates. Intervening couplings to color ancillas are shown in the Z/X macro panels.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.2, 2.05)
    ax.axis("off")
    ax.text(
        6,
        1.89,
        "Optimization candidate only: any fusion must preserve the intermediate MX detector and be re-audited for faults.",
        ha="center",
        fontsize=9.5,
        color="#9f1239",
        fontweight="bold",
    )
    ax.set_title("Exact ZXZ-generator q2/q2′ lifecycle (no CNOT cancellation applied)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def pre_interface_macro_table(probe: str = "+") -> pd.DataFrame:
    """Non-overlapping conceptual attribution before interface round 1."""
    if probe not in ("+", "0"):
        raise ValueError("probe must be '+' or '0'")
    steane_state = "perfect |+_L> Steane preparation" if probe == "+" else "perfect |0_L> Steane preparation"
    return pd.DataFrame(
        [
            ("P", "surface-data preparation", "RX all 25 distance-5 surface data qubits"),
            (
                "S0",
                "initial full surface-syndrome round",
                "prepare all 24 check ancillas, couple each support, and read them out",
            ),
            (
                "D0",
                "initial surface detector boundary",
                "emit single-outcome detectors for initially satisfied X checks; retain Z outcomes as baselines",
            ),
            ("C", "Steane preparation", f"{steane_state}; internal code-expansion encoder is collapsed"),
            (
                "T",
                "merge transition",
                "pause top checks Xab and Xcd before starting the merged-boundary measurements",
            ),
        ],
        columns=["macro-tick", "only operation family shown", "checks / actions"],
    )


def post_interface_macro_table(
    *,
    probe: str = "+",
    full_post_selection: bool = False,
) -> pd.DataFrame:
    """Non-overlapping conceptual attribution after interface round 3."""
    if probe not in ("+", "0"):
        raise ValueError("probe must be '+' or '0'")
    final_basis = "MX" if probe == "+" else "MZ"
    closed_family = "surface-X" if probe == "+" else "surface-Z"
    observable = (
        "first surface column in X plus Steane demolition m1,m4,m6"
        if probe == "+"
        else "top surface row in Z plus final interface ZZ outcomes"
    )
    rows = [
        (
            "R1",
            "first post-interface surface round",
            "restore Xab/Xcd and measure one complete 24-check surface round",
        ),
        (
            "D1",
            "recovery detector relations",
            "compare demolition Xab to recovered Xab; seed Xcd; compare all continuing checks in time",
        ),
    ]
    if not full_post_selection:
        rows.append(
            (
                "R2–R5",
                "four decoded surface rounds",
                "repeat the complete 24-check surface round four times, producing temporal detectors",
            )
        )
    rows.extend(
        [
            ("M", "final destructive surface readout", f"{final_basis} all 25 surface data qubits"),
            (
                "F",
                "classical end-cap comparison",
                f"no gates: XOR each stored final {closed_family}-check bit with the matching data parity recorded in M",
            ),
            ("L0", "logical observable parity", observable),
        ]
    )
    return pd.DataFrame(rows, columns=["macro-tick", "only operation family shown", "checks / actions"])


def _macro_surface_data(
    wrapper: SteanePlusSurfaceCode,
) -> dict[tuple[int, int], str]:
    return {
        (wrapper.surface_offset_x + 2 * col, wrapper.surface_offset_y + 2 * row):
            (surface_data_label((wrapper.surface_offset_x + 2 * col, wrapper.surface_offset_y + 2 * row)) or "")
            .removeprefix("Su-")
        for row in range(wrapper.surface_distance)
        for col in range(wrapper.surface_distance)
    }


def _setup_macro_surface_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1, 11)
    ax.set_ylim(27, 14)


def _draw_macro_surface_data(
    ax: plt.Axes,
    wrapper: SteanePlusSurfaceCode,
    *,
    operation: str | None = None,
    highlight: set[tuple[int, int]] | None = None,
) -> None:
    highlight = highlight or set()
    for coord, label in _macro_surface_data(wrapper).items():
        selected = coord in highlight
        ax.scatter(
            *coord,
            s=105 if selected else 72,
            color="#0f766e" if selected else "#2b6cb0",
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(label, coord, xytext=(4, 4), textcoords="offset points", fontsize=7.5)
        if operation is not None:
            ax.annotate(
                operation,
                coord,
                xytext=(0, -16),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color="#173b57",
                fontweight="bold",
            )


def _draw_macro_surface_checks(
    ax: plt.Axes,
    wrapper: SteanePlusSurfaceCode,
    *,
    family: str | None = None,
    highlight_top: bool = False,
) -> None:
    """Draw each complete surface check once as a collapsed support."""
    for coord, measurement in wrapper.surface_syndrome_measurements.items():
        kind = "X" if measurement.__class__.__name__.startswith("SurfaceX") else "Z"
        if family is not None and kind != family:
            continue
        support = surface_support(measurement)
        is_top = coord in {(2, 16), (6, 16)}
        color = "#d1495b" if kind == "X" else "#68a357"
        if highlight_top and is_top:
            color = "#7b2cbf"
        if len(support) >= 3:
            center = np.mean(np.asarray(support), axis=0)
            ordered = sorted(support, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
            ax.add_patch(
                Polygon(
                    ordered,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.24,
                    linewidth=2 if highlight_top and is_top else 1.1,
                )
            )
        else:
            ax.plot(
                [support[0][0], support[1][0]],
                [support[0][1], support[1][1]],
                color=color,
                linewidth=10 if highlight_top and is_top else 7,
                alpha=0.42,
                solid_capstyle="round",
            )
        ax.scatter(*coord, marker="P" if is_top else "s", s=95 if is_top else 44, color=color, edgecolor="white", zorder=6)
        ax.text(coord[0], coord[1], kind, fontsize=7, ha="center", va="center", zorder=7)
        if highlight_top and is_top:
            name = "Xab q23" if coord == (2, 16) else "Xcd q53"
            ax.annotate(name, coord, xytext=(-17, 15), textcoords="offset points", fontsize=8, fontweight="bold")
            for data in support:
                ax.annotate(
                    "",
                    xy=data,
                    xytext=coord,
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#7b2cbf",
                        "lw": 2.5,
                        "mutation_scale": 22,
                        "shrinkA": 8,
                        "shrinkB": 7,
                    },
                    zorder=4,
                )
    ax.text(
        0.5,
        0.025,
        "each shaded support = reset + all CXs + readout\n"
        "X ancilla→data; Z data→ancilla",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )


def plot_grouped_pre_interface(
    wrapper: SteanePlusSurfaceCode,
    *,
    probe: str = "+",
    detector_overlay: bool = False,
) -> plt.Figure:
    """Plot every pre-interface operation family once, collapsing code expansion."""
    if probe not in ("+", "0"):
        raise ValueError("probe must be '+' or '0'")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    _setup_macro_surface_axis(ax, "Macro-tick P — prepare surface data only")
    _draw_macro_surface_data(ax, wrapper, operation="RX")
    ax.text(5, 26.1, r"Prepare the distance-5 surface patch in $|+⟩^{\otimes 25}$", ha="center", fontsize=9)

    ax = axes[0, 1]
    _setup_macro_surface_axis(ax, "Macro-tick S0 — first full surface-check round")
    _draw_macro_surface_data(ax, wrapper)
    _draw_macro_surface_checks(ax, wrapper)
    if detector_overlay:
        ax.scatter(2, 18, s=260, facecolor="none", edgecolor="#111111", linewidth=3, zorder=9)
        ax.scatter(2, 18, s=80, marker="*", color="#dc2626", zorder=10)
        ax.annotate(
            "q24: m5\nstarts D16\nB1: X after initial q24 RZ",
            (2, 18),
            xytext=(30, 12),
            textcoords="offset points",
            fontsize=8.5,
            color="#111111",
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1.5},
        )

    ax = axes[0, 2]
    ax.axis("off")
    ax.set_title("Macro-tick D0 — initial detector boundary only", fontsize=12, fontweight="bold")
    ax.text(
        0.5,
        0.62,
        "Initially satisfied X checks:\n" r"$D_X=m_X^{(0)}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
    )
    ax.text(
        0.5,
        0.32,
        "Each first Z-check outcome is retained as the\nbaseline for its next temporal detector.",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
    )
    ax.text(0.5, 0.12, "DETECTOR instructions do not remeasure qubits.", transform=ax.transAxes, ha="center", fontsize=10, color="#555555")

    ax = axes[1, 0]
    ax.set_title("Macro-tick C — perfect Steane preparation", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(0, 7)
    ax.set_ylim(17, 9)
    ax.set_xticks([])
    ax.set_yticks([])
    reset_x = {"0", "1", "2", "3"} if probe == "+" else {"1", "2", "3"}
    for coord, raw_label in STEANE_COORDS.items():
        label = raw_label.removeprefix("St")
        if label.startswith("2'"):
            continue
        basis = "RX" if label in reset_x else "RZ"
        ax.scatter(*coord, s=105, color="#d1495b", edgecolor="white", zorder=4)
        ax.annotate(label, coord, xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.annotate(basis, coord, xytext=(0, -17), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    ax.text(3.5, 16.7, rf"Prepare perfect Steane $|{probe}_L\rangle$", ha="center", fontsize=11, fontweight="bold")
    ax.text(3.5, 9.4, "The ideal Clifford encoder is one collapsed code-expansion operation.", ha="center", fontsize=9)

    ax = axes[1, 1]
    _setup_macro_surface_axis(ax, "Macro-tick T — enter the merged boundary")
    _draw_macro_surface_data(ax, wrapper)
    for coord, name in [((2, 16), "pause Xab"), ((6, 16), "pause Xcd")]:
        ax.scatter(*coord, marker="x", s=180, color="#b91c1c", linewidth=3, zorder=6)
        ax.annotate(name, coord, xytext=(-18, 17), textcoords="offset points", fontsize=9, fontweight="bold", color="#991b1b")
    ax.text(5, 26.0, "Builder transition only: these two checks stop before interface round 1", ha="center", fontsize=9)

    ax = axes[1, 2]
    ax.axis("off")
    ax.set_title("Pre-interface coverage", fontsize=12, fontweight="bold")
    ax.text(
        0.5,
        0.56,
        "P → S0 → D0 → C → T → interface round 1",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.30,
        "Every pre-interface quantum or detector operation is assigned once.\n"
        "Only the internal perfect Steane encoder is intentionally collapsed.",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
    )

    fig.suptitle(
        "Before Hirano interface round 1 — complete barrier-collapsed operation atlas",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_grouped_post_interface(
    wrapper: SteanePlusSurfaceCode,
    *,
    probe: str = "+",
    full_post_selection: bool = False,
) -> plt.Figure:
    """Plot every post-interface operation family once for one circuit branch."""
    if probe not in ("+", "0"):
        raise ValueError("probe must be '+' or '0'")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    _setup_macro_surface_axis(ax, "Macro-tick R1 — first post-interface surface round")
    _draw_macro_surface_data(ax, wrapper)
    _draw_macro_surface_checks(ax, wrapper, highlight_top=True)
    ax.text(5, 15.0, "q23 performs the next-in-time Xab extraction; q53 starts Xcd", ha="center", fontsize=9)

    ax = axes[0, 1]
    ax.axis("off")
    ax.set_title("Macro-tick D1 — recovery detector relations only", fontsize=12, fontweight="bold")
    ax.text(
        0.5,
        0.67,
        r"$D_{X_{ab}}=m_{ab}^{\rm demolition}\oplus m_{ab}^{\rm recovery}$",
        transform=ax.transAxes,
        ha="center",
        fontsize=16,
    )
    ax.text(
        0.5,
        0.45,
        r"$X_{cd}$: first recovery value seeds its future detector record",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
    )
    ax.text(
        0.5,
        0.25,
        r"Other checks: $D_s=m_s^{\rm interface\ end}\oplus m_s^{\rm recovery}$",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
    )
    ax.text(0.5, 0.09, "This panel contains classical DETECTOR instructions, not extra measurements.", transform=ax.transAxes, ha="center", fontsize=10, color="#555555")

    ax = axes[0, 2]
    _setup_macro_surface_axis(
        ax,
        "Macro-ticks R2–R5 — four decoded surface rounds"
        if not full_post_selection
        else "Macro-ticks R2–R5 — omitted in full-postselection branch",
    )
    if full_post_selection:
        ax.axis("off")
        ax.text(0.5, 0.5, "No additional decoded rounds", transform=ax.transAxes, ha="center", fontsize=15, fontweight="bold")
    else:
        _draw_macro_surface_data(ax, wrapper)
        _draw_macro_surface_checks(ax, wrapper)
        ax.text(5, 15.0, "×4 complete rounds; every check result is compared to the preceding round", ha="center", fontsize=9, fontweight="bold")

    final_basis = "MX" if probe == "+" else "MZ"
    closed_family = "X" if probe == "+" else "Z"
    ax = axes[1, 0]
    _setup_macro_surface_axis(ax, f"Macro-tick M — destructive {final_basis} data readout")
    _draw_macro_surface_data(ax, wrapper, operation=final_basis)
    ax.text(5, 26.0, f"Measure all 25 surface data qubits in the {closed_family} basis", ha="center", fontsize=9)

    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title(
        f"Macro-tick F — compare stored checks with final {closed_family}-data parity",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.94,
        "M HAS ALREADY MEASURED THE DATA • F IS CLASSICAL XOR ONLY",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color="#991b1b",
    )

    def classical_box(x: float, y: float, label: str, *, color: str = "#173b57") -> None:
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#f8fafc",
                "edgecolor": color,
                "linewidth": 1.5,
            },
        )

    def classical_arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "-|>", "lw": 1.7, "color": "#64748b", "mutation_scale": 16},
        )

    if probe == "+":
        first_stored = "stored q23 bit\nlast Xab check"
        first_final = "bits from M\nMX(a) ⊕ MX(b)"
        first_detector = r"$D_{X_{ab}}$"
        first_formula = r"$D_{X_{ab}}=m_{X_{ab}}^{\rm last}\oplus m_a^{\rm final}\oplus m_b^{\rm final}$"
        second_stored = "stored ancilla bit\nlast weight-4 X check"
        second_final = "bits from M\nMX on its 4 data qubits"
    else:
        first_stored = "stored ancilla bit\nlast boundary Z check"
        first_final = "bits from M\nMZ on its 2 data qubits"
        first_detector = r"$D_{Z,\rm boundary}$"
        first_formula = r"$D_Z=m_Z^{\rm last}\oplus m_{d_1}^{\rm final}\oplus m_{d_2}^{\rm final}$"
        second_stored = "stored ancilla bit\nlast weight-4 Z check"
        second_final = "bits from M\nMZ on its 4 data qubits"

    classical_box(0.25, 0.79, first_stored)
    classical_box(0.25, 0.66, first_final)
    classical_box(0.62, 0.725, "XOR", color="#7b2cbf")
    classical_box(0.86, 0.725, first_detector, color="#2f855a")
    classical_arrow(0.37, 0.79, 0.55, 0.735)
    classical_arrow(0.37, 0.66, 0.55, 0.715)
    classical_arrow(0.68, 0.725, 0.80, 0.725)
    ax.text(0.5, 0.55, first_formula, transform=ax.transAxes, ha="center", fontsize=10.5)

    classical_box(0.25, 0.40, second_stored)
    classical_box(0.25, 0.27, second_final)
    classical_box(0.62, 0.335, "XOR", color="#7b2cbf")
    classical_box(0.86, 0.335, r"$D_{\rm check}$", color="#2f855a")
    classical_arrow(0.37, 0.40, 0.55, 0.345)
    classical_arrow(0.37, 0.27, 0.55, 0.325)
    classical_arrow(0.68, 0.335, 0.80, 0.335)
    ax.text(
        0.5,
        0.17,
        r"same rule for weight 4: stored check bit $\oplus$ final parity of its four data bits",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )
    ax.text(
        0.5,
        0.055,
        "Why F comes after M: it consumes M's recorded bits.\nIt never touches the already measured qubits.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        fontweight="bold",
        color="#555555",
    )

    ax = axes[1, 2]
    _setup_macro_surface_axis(ax, "Macro-tick L0 — logical observable parity only")
    logical_surface = (
        {(1, 17 + 2 * row) for row in range(wrapper.surface_distance)}
        if probe == "+"
        else {(1 + 2 * col, 17) for col in range(wrapper.surface_distance)}
    )
    _draw_macro_surface_data(ax, wrapper, highlight=logical_surface)
    if probe == "+":
        formula = r"$L_0=\bigoplus_{q\in\mathrm{first\ column}}m_q^X\oplus m_1\oplus m_4\oplus m_6$"
        source = "green: destructive surface-X support; Steane terms were recorded in demolition"
    else:
        formula = r"$L_0=\bigoplus_{q\in\mathrm{top\ row}}m_q^Z\oplus\bigoplus_j m_{ZZ,j}^{\rm interface}$"
        source = "green: destructive surface-Z support; ZZ terms were recorded at the interface boundary"
    ax.text(5, 25.9, formula, ha="center", fontsize=11)
    ax.text(5, 15.0, source, ha="center", fontsize=8.5)

    branch = "full postselection" if full_post_selection else "decoded-surface"
    fig.suptitle(
        f"After Hirano interface round 3 — complete {branch} operation atlas ({probe} probe)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_d16_d46_d77_spacetime(wrapper: SteanePlusSurfaceCode) -> plt.Figure:
    """Locate the zero-probe temporal-corner witness on macro-S geometry."""
    support = [(1, 17), (3, 17), (1, 19), (3, 19)]
    center = np.mean(np.asarray(support), axis=0)
    ordered = sorted(support, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
    panels = [
        ("before interface: S0", "m5", "starts D16", "B1: X after q24 RZ"),
        ("interface round 1: S", "m29", "closes D16; starts D46", "A1: X after q24 RZ"),
        ("interface round 2: S", "m61", "closes D46; starts D77", "A2: X after q24 RZ"),
        ("interface round 3: S + I", "m100", "closes D77", "B2: X on a after CX a→q23"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for index, (ax, (title, record, edge, fault)) in enumerate(zip(axes.ravel(), panels)):
        _setup_macro_surface_axis(ax, title)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(20.3, 14.5)
        for coord in support:
            label = (surface_data_label(coord) or str(coord)).removeprefix("Su-")
            ax.scatter(*coord, s=85, color="#2b6cb0", edgecolor="white", zorder=5)
            ax.annotate(label, coord, xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.add_patch(
            Polygon(
                ordered,
                closed=True,
                facecolor="#d9f0d3",
                edgecolor="#2f855a",
                alpha=0.58,
                linewidth=2.2,
            )
        )
        ax.scatter(2, 18, marker="s", s=165, color="#79c267", edgecolor="#111111", linewidth=2.5, zorder=8)
        ax.text(2, 18, "Z", ha="center", va="center", fontsize=9, fontweight="bold", zorder=9)
        ax.annotate(
            f"q24@(2,18): {record}\n{edge}",
            (2, 18),
            xytext=(48, -7),
            textcoords="offset points",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )
        fault_coord = (2, 18) if index < 3 else (1, 17)
        ax.scatter(*fault_coord, marker="*", s=140, color="#dc2626", edgecolor="white", linewidth=0.8, zorder=10)
        ax.annotate(
            fault,
            fault_coord,
            xytext=(0, 70) if index < 3 else (-72, 58),
            textcoords="offset points",
            ha="center" if index < 3 else "left",
            fontsize=8.5,
            color="#b91c1c",
            fontweight="bold",
        )
        if index == 3:
            # Exact interface CNOT direction for the competing B2 fault.
            ax.scatter(2, 16, marker="P", s=135, color="#7b61a8", edgecolor="white", zorder=8)
            ax.annotate("q23: Z1a-R", (2, 16), xytext=(9, 7), textcoords="offset points", fontsize=8, fontweight="bold")
            ax.annotate(
                "",
                xy=(2, 16),
                xytext=(1, 17),
                arrowprops={"arrowstyle": "-|>", "color": "#2f855a", "lw": 2.8, "mutation_scale": 23, "shrinkA": 7, "shrinkB": 8},
                zorder=7,
            )
        ax.text(
            0.5,
            0.02,
            "q24 support: a, b, r1c0, r1c1",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#444444",
        )

    fig.suptitle(
        "Where D16, D46, and D77 live: three time edges of one surface-Z check\n"
        "red stars mark the two indistinguishable two-fault histories A and B",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


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
