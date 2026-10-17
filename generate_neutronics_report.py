import csv
import gzip
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "FINAL_NEUTRONICS_PROJECT_REPORT.pdf"
HISTORY_DTYPE = np.dtype(
    [("track_id", "<u4"), ("collision_count", "<u4"), ("energy_ev", "<f4")]
)
REACTIONS = ("elastic", "inelastic", "capture", "fission")
MODERATOR_ISOTOPES = ("H-1", "H-2", "O-16")
PRIMARY_COUNT = 1_000_000
SEED = 20261016
COLORS = {
    "U-238": "#365f91",
    "Natural U": "#2b8a6e",
    "H2O natural": "#d07a24",
    "H2O 2% enriched": "#9b59b6",
    "D2O natural": "#c94c4c",
}


def read_summary(directory):
    values = {}
    for line in (directory / "summary.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(
            r"Collision count, (.+?) (elastic|inelastic|capture|fission): (\d+)$",
            line,
        )
        if match:
            values[f"{match.group(1)}_{match.group(2)}"] = int(match.group(3))
            continue
        match = re.match(
            r"(.+?) fission events in (thermal|resonance|fast) region.*: (\d+)$",
            line,
        )
        if match:
            values[f"{match.group(1)}_fission_{match.group(2)}"] = int(match.group(3))
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def load_case(name, directory, branch, ratio=None):
    summary = read_summary(directory)
    initial = int(summary["Initial neutrons"])
    collisions = {
        isotope: {
            reaction: summary.get(f"{isotope}_{reaction}", 0)
            for reaction in REACTIONS
        }
        for isotope in ("H-1", "H-2", "O-16", "U-235", "U-238")
    }
    captures = sum(item["capture"] for item in collisions.values())
    fissions = sum(item["fission"] for item in collisions.values())
    moderator_captures = sum(
        collisions[isotope]["capture"] for isotope in MODERATOR_ISOTOPES
    )
    fuel_captures = collisions["U-235"]["capture"] + collisions["U-238"]["capture"]
    return {
        "name": name,
        "directory": directory,
        "branch": branch,
        "ratio": ratio,
        "summary": summary,
        "initial": initial,
        "final": int(
            summary[
                "Final set (initial neutrons terminated by capture/fission)"
            ]
        ),
        "tracks": int(summary["Total neutron tracks including fission offspring"]),
        "history_points": int(
            summary["Recorded history points for all neutron tracks"]
        ),
        "generation": int(
            summary["New-generation set size (requested k-infinity count)"]
        ),
        "k_project": int(
            summary["New-generation set size (requested k-infinity count)"]
        )
        / initial,
        "collisions": collisions,
        "captures": captures,
        "fissions": fissions,
        "moderator_captures": moderator_captures,
        "fuel_captures": fuel_captures,
        "moderator_capture_share": (
            moderator_captures / captures if captures else 0.0
        ),
        "fission_capture_ratio": fissions / captures if captures else 0.0,
    }


def load_all_cases():
    u238 = load_case(
        "Pure U-238", ROOT / "simulation_results", "pure_u238"
    )
    natural = load_case(
        "Natural uranium", ROOT / "natural_uranium_results", "natural_uranium"
    )

    h2o_rows = []
    with (ROOT / "h2o_uranium_results" / "index.csv").open(
        newline="", encoding="utf-8"
    ) as index_file:
        for row in csv.DictReader(index_file):
            h2o_rows.append(
                load_case(
                    row["case"],
                    Path(row["output_directory"]),
                    row["uranium_branch"],
                    int(row["water_molecules_per_uranium_atom"]),
                )
            )

    d2o_rows = []
    with (ROOT / "d2o_natural_uranium_results" / "index.csv").open(
        newline="", encoding="utf-8"
    ) as index_file:
        for row in csv.DictReader(index_file):
            d2o_rows.append(
                load_case(
                    row["case"],
                    Path(row["output_directory"]),
                    "natural",
                    int(row["D2O_molecules_per_uranium_atom"]),
                )
            )

    if len(h2o_rows) != 60 or len(d2o_rows) != 30:
        raise RuntimeError(
            f"Expected 60 H2O and 30 D2O cases; found {len(h2o_rows)} and "
            f"{len(d2o_rows)}"
        )
    all_cases = [u238, natural, *h2o_rows, *d2o_rows]
    for case in all_cases:
        if case["initial"] != PRIMARY_COUNT or case["final"] != PRIMARY_COUNT:
            raise RuntimeError(
                f"{case['name']} did not finish {PRIMARY_COUNT} primary neutrons"
            )
        history_path = case["directory"] / "history_points.bin"
        if history_path.stat().st_size != case["history_points"] * 12:
            raise RuntimeError(f"History file size does not match in {case['name']}")
    return u238, natural, h2o_rows, d2o_rows


def indexed_cases(rows, branch=None):
    selected = rows if branch is None else [row for row in rows if row["branch"] == branch]
    return sorted(selected, key=lambda case: case["ratio"])


def read_terminal_collision_counts(directory):
    counts = np.fromiter(
        (
            int(row["collision_count"])
            for row in csv.DictReader(
                (directory / "final_neutrons.csv").open(
                    newline="", encoding="utf-8"
                )
            )
        ),
        dtype=np.uint32,
        count=PRIMARY_COUNT,
    )
    if counts.size != PRIMARY_COUNT:
        raise RuntimeError(f"Invalid terminal-neutron count in {directory}")
    return counts


def read_history_energy_groups(directory, track_stride=200):
    selected_points = defaultdict(list)
    history_path = directory / "history_points.bin"
    with history_path.open("rb") as history_file:
        while True:
            chunk = np.fromfile(history_file, dtype=HISTORY_DTYPE, count=1_000_000)
            if chunk.size == 0:
                break
            mask = (
                (chunk["track_id"] < PRIMARY_COUNT)
                & (chunk["track_id"] % track_stride == 0)
            )
            selected = chunk[mask]
            for collision_count in np.unique(selected["collision_count"]):
                step_mask = selected["collision_count"] == collision_count
                selected_points[int(collision_count)].append(
                    selected["energy_ev"][step_mask]
                )
    return {
        collision_count: np.concatenate(values)
        for collision_count, values in selected_points.items()
    }


def birth_energy_plot(directory, output_path):
    energies_ev = []
    with (directory / "history_points.bin").open("rb") as history_file:
        while True:
            chunk = np.fromfile(history_file, dtype=HISTORY_DTYPE, count=1_000_000)
            if chunk.size == 0:
                break
            mask = (
                (chunk["track_id"] < PRIMARY_COUNT)
                & (chunk["collision_count"] == 0)
            )
            energies_ev.append(chunk["energy_ev"][mask])
    energy_mev = np.concatenate(energies_ev).astype(np.float64) / 1.0e6
    if energy_mev.size != PRIMARY_COUNT:
        raise RuntimeError(
            f"Expected {PRIMARY_COUNT} primary birth energies, found {energy_mev.size}"
        )

    bins = np.logspace(-11, 2, 70)
    fig, ax = plt.subplots(figsize=(9.2, 4.7), constrained_layout=True)
    ax.hist(
        energy_mev,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.4,
        color=COLORS["U-238"],
        label="1,000,000 sampled primary births",
    )
    theory_energy = np.logspace(-11, 2, 500)
    spectrum = 0.771 * np.sqrt(theory_energy) * np.exp(-0.776 * theory_energy)
    ax.plot(
        theory_energy,
        spectrum,
        color="#c94c4c",
        linewidth=1.4,
        label="Specified spectrum",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Birth energy (MeV)")
    ax.set_ylabel("Probability density per MeV")
    ax.set_title("Primary neutron birth spectrum")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return energy_mev


def slowing_plot(representative_cases, output_path):
    fig, ax = plt.subplots(figsize=(9.2, 4.7), constrained_layout=True)
    for case, label, color in representative_cases:
        grouped = read_history_energy_groups(case["directory"])
        steps = sorted(grouped)
        medians = [float(np.median(grouped[step])) for step in steps]
        ax.plot(steps, medians, linewidth=1.35, label=label, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("Collision count")
    ax.set_ylabel("Median primary-neutron energy (eV)")
    ax.set_title("Neutron slowing-down histories")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def history_dynamics_plot(representative_cases, output_path):
    fig, (trajectory_ax, median_ax) = plt.subplots(
        1, 2, figsize=(10.0, 4.8), constrained_layout=True
    )
    for case, label, color in representative_cases:
        trajectories = defaultdict(list)
        with (case["directory"] / "history_points.bin").open("rb") as history_file:
            while True:
                chunk = np.fromfile(
                    history_file, dtype=HISTORY_DTYPE, count=1_000_000
                )
                if chunk.size == 0:
                    break
                mask = (
                    (chunk["track_id"] < PRIMARY_COUNT)
                    & (chunk["track_id"] % 50_000 == 0)
                )
                for point in chunk[mask]:
                    trajectories[int(point["track_id"])].append(
                        (int(point["collision_count"]), float(point["energy_ev"]))
                    )
        for index, points in enumerate(trajectories.values()):
            points.sort()
            values = np.asarray(points)
            trajectory_ax.plot(
                values[:, 0],
                values[:, 1],
                color=color,
                alpha=0.35,
                linewidth=0.65,
                label=label if index == 0 else None,
            )

        grouped = read_history_energy_groups(case["directory"])
        steps = sorted(grouped)
        medians = [float(np.median(grouped[step])) for step in steps]
        median_ax.plot(
            steps,
            medians,
            linewidth=1.25,
            label=label,
            color=color,
        )

    for ax in (trajectory_ax, median_ax):
        ax.set_yscale("log")
        ax.set_xlabel("Collision count")
        ax.grid(True, which="both", alpha=0.2)
    trajectory_ax.set_ylabel("Primary-neutron energy (eV)")
    trajectory_ax.set_title("Sample primary histories")
    trajectory_ax.legend(frameon=False, fontsize=6.2)
    median_ax.set_ylabel("Median primary-neutron energy (eV)")
    median_ax.set_title("Median slowing-down trend")
    median_ax.legend(frameon=False, fontsize=6.2)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def criticality_plot(h2o_rows, d2o_rows, output_path):
    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    series = [
        (indexed_cases(h2o_rows, "natural"), "H2O + natural U", COLORS["H2O natural"]),
        (
            indexed_cases(h2o_rows, "enriched_2pct"),
            "H2O + 2% enriched U",
            COLORS["H2O 2% enriched"],
        ),
        (indexed_cases(d2o_rows), "D2O + natural U", COLORS["D2O natural"]),
    ]
    for cases, label, color in series:
        ax.plot(
            [case["ratio"] for case in cases],
            [case["k_project"] for case in cases],
            marker="o",
            markersize=3.1,
            linewidth=1.3,
            label=label,
            color=color,
        )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.1, label="Critical threshold")
    ax.set_xlim(1, 30)
    ax.set_xlabel("Water molecules per uranium atom")
    ax.set_ylabel("New-generation count / 1,000,000 initial neutrons")
    ax.set_title("Project criticality indicator by moderator ratio")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def absorption_fission_plot(h2o_rows, d2o_rows, output_path):
    fig, (capture_ax, ratio_ax) = plt.subplots(
        2, 1, figsize=(9.2, 7.6), sharex=True, constrained_layout=True
    )
    series = [
        (indexed_cases(h2o_rows, "natural"), "H2O + natural U", COLORS["H2O natural"]),
        (
            indexed_cases(h2o_rows, "enriched_2pct"),
            "H2O + 2% enriched U",
            COLORS["H2O 2% enriched"],
        ),
        (indexed_cases(d2o_rows), "D2O + natural U", COLORS["D2O natural"]),
    ]
    for cases, label, color in series:
        x = [case["ratio"] for case in cases]
        capture_ax.plot(
            x,
            [1.0e6 * case["moderator_captures"] / case["initial"] for case in cases],
            color=color,
            linewidth=1.3,
            linestyle="-",
            label=f"{label}: moderator captures",
        )
        capture_ax.plot(
            x,
            [1.0e6 * case["fuel_captures"] / case["initial"] for case in cases],
            color=color,
            linewidth=1.1,
            linestyle=":",
            label=f"{label}: uranium captures",
        )
        ratio_ax.plot(
            x,
            [case["fission_capture_ratio"] for case in cases],
            color=color,
            linewidth=1.3,
            label=label,
        )
    capture_ax.set_ylabel("Capture events per million primaries")
    capture_ax.set_title("Capture in moderator and uranium")
    capture_ax.grid(True, alpha=0.2)
    capture_ax.legend(frameon=False, fontsize=7, ncol=2)
    ratio_ax.set_xlabel("Water molecules per uranium atom")
    ratio_ax.set_ylabel("Fission events / capture events")
    ratio_ax.set_title("Fission-to-capture event ratio")
    ratio_ax.set_xlim(1, 30)
    ratio_ax.grid(True, alpha=0.2)
    ratio_ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def terminal_collision_plot(representative_cases, output_path):
    fig, ax = plt.subplots(figsize=(9.2, 4.7), constrained_layout=True)
    bins = np.arange(0, 401, 5)
    for case, label, color in representative_cases:
        counts = read_terminal_collision_counts(case["directory"])
        ax.hist(
            counts,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.25,
            label=label,
            color=color,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Collisions before primary termination")
    ax.set_ylabel("Fraction of primary neutrons per bin")
    ax.set_title("Collision-count proxy for neutron residence")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def collision_statistics(case):
    counts = read_terminal_collision_counts(case["directory"])
    return {
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "p90": float(np.percentile(counts, 90)),
    }


def cross_sections_plot(output_path):
    csv_path = ROOT / "neutron_data" / "neutron_cross_sections.csv.gz"
    grouped = defaultdict(lambda: defaultdict(lambda: ([], [])))
    with gzip.open(csv_path, "rt", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            energy = float(row["energy_eV"])
            cross_section = float(row["cross_section_b"])
            if energy <= 0.0 or cross_section <= 0.0:
                continue
            energies, values = grouped[row["isotope"]][row["display_reaction"]]
            energies.append(energy)
            values.append(cross_section)

    isotopes = ("H-1", "H-2", "O-16", "U-235", "U-238")
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 7.5), constrained_layout=True)
    reaction_colors = {
        "elastic": "#365f91",
        "inelastic": "#d07a24",
        "capture (n,g)": "#2b8a6e",
        "total fission": "#c94c4c",
    }
    for ax, isotope in zip(axes.flat, isotopes):
        for reaction, (energies, values) in grouped[isotope].items():
            order = np.argsort(energies)
            ax.plot(
                np.asarray(energies)[order],
                np.asarray(values)[order],
                linewidth=0.8,
                label=reaction,
                color=reaction_colors.get(reaction),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(isotope)
        ax.set_xlabel("Neutron energy (eV)")
        ax.set_ylabel("Cross section (barn)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(frameon=False, fontsize=6.5)
    axes.flat[-1].axis("off")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def final_energy_plot(cases, output_path):
    bins = np.logspace(-5, 8, 66)
    fig, ax = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    for case, label, color in cases:
        energies = []
        with (case["directory"] / "final_neutrons.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            for row in csv.DictReader(source):
                energies.append(float(row["energy_ev"]))
        if len(energies) != PRIMARY_COUNT:
            raise RuntimeError(
                f"Expected {PRIMARY_COUNT} terminal energies in {case['name']}, "
                f"found {len(energies)}"
            )
        ax.hist(
            energies,
            bins=bins,
            weights=np.full(len(energies), 1.0 / PRIMARY_COUNT),
            histtype="step",
            linewidth=1.3,
            color=color,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy immediately before primary termination (eV)")
    ax.set_ylabel("Fraction of initial neutrons per logarithmic bin")
    ax.set_title("Primary terminal-energy distributions")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fmt(value):
    return f"{value:,.0f}"


def fmt_m(value):
    return f"{value / 1e6:.2f}"


def crossing_description(cases):
    ordered = indexed_cases(cases)
    crossings = []
    for left, right in zip(ordered, ordered[1:]):
        a = left["k_project"] - 1.0
        b = right["k_project"] - 1.0
        if a == 0:
            crossings.append(f"at {left['ratio']}")
        elif a * b < 0:
            crossings.append(f"{left['ratio']}-{right['ratio']}")
    return ", ".join(crossings) if crossings else "none in tested range"


def styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            textColor=colors.HexColor("#15324b"),
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            name="Subtitle",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4e6172"),
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    base.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=base["Heading1"],
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#15324b"),
            spaceBefore=8,
            spaceAfter=7,
        )
    )
    base.add(
        ParagraphStyle(
            name="SubHeading",
            parent=base["Heading2"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#23606f"),
            spaceBefore=6,
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="BodySmall",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=11.3,
            spaceAfter=5,
        )
    )
    base.add(
        ParagraphStyle(
            name="TableCell",
            parent=base["BodyText"],
            fontSize=7.4,
            leading=9.3,
        )
    )
    base.add(
        ParagraphStyle(
            name="Callout",
            parent=base["BodyText"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#15324b"),
            backColor=colors.HexColor("#eaf2f6"),
            borderColor=colors.HexColor("#9bb8c8"),
            borderWidth=0.6,
            borderPadding=8,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    return base


def para(text, style):
    return Paragraph(text, style)


def make_table(rows, widths, repeat_rows=1):
    table = Table(rows, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#15324b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bcc8d0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#f3f6f8"),
                ]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def image(path, width=7.25 * inch, height=3.7 * inch):
    picture = Image(str(path))
    picture._restrictSize(width, height)
    return picture


def page_decoration(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cad4dc"))
    canvas.line(0.62 * inch, 0.52 * inch, 7.88 * inch, 0.52 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#526575"))
    canvas.drawString(0.65 * inch, 0.34 * inch, "CUDA neutron Monte Carlo project")
    canvas.drawRightString(7.85 * inch, 0.34 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_report(u238, natural, h2o_rows, d2o_rows, figures, representatives):
    s = styles()
    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.56 * inch,
        bottomMargin=0.68 * inch,
        title="Final Neutronics Monte Carlo Project Report",
        author="",
        subject="Million-neutron CUDA simulations and project criticality criterion",
    )
    story = []
    story.extend(
        [
            Spacer(1, 0.2 * inch),
            para("Neutronics Monte Carlo Project", s["ReportTitle"]),
            para(
                "Final report aligned with CH 5960: The Nuclear Energy Option "
                "assignment",
                s["Subtitle"],
            ),
            para(
                "Simulation date: 17 October 2026 | CUDA 13.3 | GeForce RTX 4060 | "
                "Seed: 20261016",
                s["Subtitle"],
            ),
            para(
                "<b>Headline:</b> The project ran 92 cases with one million initial "
                "neutrons each. Applying the requested project rule "
                "(new-generation count compared with the one-million source), "
                "pure U-238 and natural uranium alone are below threshold; the "
                "H2O and D2O ratio sweeps each cross the threshold.",
                s["Callout"],
            ),
            para("1. Assignment objective", s["SectionHeading"]),
            para(
                "The assignment asks Monte Carlo neutron-transport simulations "
                "to demonstrate four infinite-reactor outcomes: pure U-238 is "
                "subcritical; natural uranium alone is subcritical; 2% enriched "
                "uranium moderated by light water can be critical; and natural "
                "uranium moderated by heavy water can be critical. It also asks "
                "for energy-dependent cross sections, one million and one "
                "billion source-neutron studies, and graphs of birth energies, "
                "slowing down, neutron residence/lifetime, moderator-versus-fuel "
                "absorption, and fission-to-capture ratio.",
                s["BodySmall"],
            ),
            para("Project criticality rule", s["SubHeading"]),
            para(
                "As specified by the project owner for this report, the "
                "criticality indicator is <b>K = new-generation set size / "
                "original neutron count</b>. K = 1 is critical, K &lt; 1 is "
                "subcritical, and K &gt; 1 is supercritical. For the million "
                "source neutrons used here, the threshold is a new-generation "
                "count of 1,000,000.",
                s["BodySmall"],
            ),
        ]
    )

    outcomes = [
        ["Assignment case", "Cases tested", "Observed K range", "Result by project rule"],
        [
            "Pure U-238",
            "1",
            f"{u238['k_project']:.3f}",
            "Subcritical (new-generation count = 0)",
        ],
        [
            "Natural uranium, no moderator",
            "1",
            f"{natural['k_project']:.3f}",
            "Subcritical (new-generation count = 0)",
        ],
    ]
    for label, cases in (
        ("2% enriched U + H2O", indexed_cases(h2o_rows, "enriched_2pct")),
        ("Natural U + H2O", indexed_cases(h2o_rows, "natural")),
        ("Natural U + D2O", indexed_cases(d2o_rows)),
    ):
        k_values = [case["k_project"] for case in cases]
        outcomes.append(
            [
                label,
                str(len(cases)),
                f"{min(k_values):.3f} to {max(k_values):.3f}",
                f"Crosses K=1 between ratios {crossing_description(cases)}",
            ]
        )
    story.append(
        make_table(
            [[para(cell, s["TableCell"]) for cell in row] for row in outcomes],
            [1.55 * inch, 0.68 * inch, 1.1 * inch, 3.2 * inch],
        )
    )
    story.append(Spacer(1, 5))
    story.append(
        para(
            "The tested integer ratios do not land exactly at K = 1; the "
            "crossing intervals locate where the operational count changes "
            "from below to above the threshold.",
            s["BodySmall"],
        )
    )

    story.extend(
        [
            PageBreak(),
            para("2. Model and simulation design", s["SectionHeading"]),
            para(
                "<b>Geometry:</b> A spatially uniform, infinite mixture. The "
                "code tracks neutron energy and collision history, but not "
                "position, flight distance, leakage, or physical time.",
                s["BodySmall"],
            ),
            para(
                "<b>Birth spectrum:</b> S(E) = 0.771 E^0.5 exp(-0.776 E), E in "
                "MeV. The CUDA code samples the equivalent Gamma distribution "
                "using three Gaussian variates, converts MeV to eV, and clamps "
                "energy to 1e-5 through 1e8 eV.",
                s["BodySmall"],
            ),
            para(
                "<b>Reactions:</b> Elastic scattering applies the specified "
                "mass-dependent energy multiplier (and 1e-6 for H-1); inelastic "
                "scattering halves neutron energy; capture terminates a track; "
                "fission terminates the parent and samples 2 or 3 offspring "
                "with the isotope-specific multiplicity rule and birth spectrum.",
                s["BodySmall"],
            ),
            para(
                "<b>Cross sections:</b> The project data archive contains "
                "KAERI NuDat/ENDF/B-VII.1 energy-dependent records for H-1, "
                "H-2, O-16, U-235, and U-238. The CUDA runs interpolate the "
                "provided positive partial cross sections and weight isotope "
                "selection by atom abundance times total available cross "
                "section. H-2 source reaction MT 3 is assigned to the model's "
                "inelastic channel. Missing isotope/reaction channels contribute "
                "zero.",
                s["BodySmall"],
            ),
            para(
                "<b>Mixtures:</b> H2O is 2 H-1 atoms plus 1 O-16 atom per "
                "molecule; D2O is 2 H-2 plus 1 O-16. Part 3 tested natural "
                "uranium (U-235:U-238 = 1:138) and 2%-enriched uranium "
                "(1:49), for water ratios 1-30. Part 4 tested natural uranium "
                "with D2O ratios 1-30.",
                s["BodySmall"],
            ),
            para("Run scale and saved products", s["SubHeading"]),
            para(
                "Each of the four parts used 1,000,000 primary neutrons and "
                "seed 20261016. Parts 1 and 2 are single runs; Part 3 has 60 "
                "runs; Part 4 has 30 runs: 92 runs and 92 million primary "
                "neutrons total. Every case completed with all primary "
                "neutrons terminated. Full binary collision-energy histories, "
                "terminal-neutron CSVs, new-generation CSVs, energy histograms, "
                "plots, and summaries were retained. Compressed duplicate "
                "history CSV exports were skipped; trajectory images display "
                "up to 25,000 sampled primary tracks.",
                s["BodySmall"],
            ),
            para("Single-composition results", s["SectionHeading"]),
        ]
    )
    single_rows = [
        [
            "Case",
            "Tracks",
            "Elastic",
            "Inelastic",
            "Capture",
            "Fission",
            "Fission/capture",
            "K",
        ]
    ]
    for case in (u238, natural):
        total = {
            reaction: sum(item[reaction] for item in case["collisions"].values())
            for reaction in REACTIONS
        }
        single_rows.append(
            [
                case["name"],
                fmt(case["tracks"]),
                fmt(total["elastic"]),
                fmt(total["inelastic"]),
                fmt(total["capture"]),
                fmt(total["fission"]),
                f"{case['fission_capture_ratio']:.3f}",
                f"{case['k_project']:.3f}",
            ]
        )
    story.append(
        make_table(
            [[para(str(cell), s["TableCell"]) for cell in row] for row in single_rows],
            [0.92 * inch, 0.82 * inch, 0.83 * inch, 0.82 * inch,
             0.79 * inch, 0.74 * inch, 0.90 * inch, 0.52 * inch],
        )
    )
    story.append(Spacer(1, 5))
    story.append(
        para(
            "In the natural-uranium-only run, U-235 contributed 66,267 fissions "
            "and U-238 171,183. The recorded generation set is zero for both "
            "single-composition runs, so both are below the stated project "
            "threshold.",
            s["BodySmall"],
        )
    )

    story.extend(
        [
            PageBreak(),
            para("3. Neutron energy behavior", s["SectionHeading"]),
            image(figures["birth"], height=3.45 * inch),
            para(
                "Figure 1. Empirical distribution from all one million primary "
                "birth records in the pure-U-238 run compared with the specified "
                "spectrum. The initial source sample is shared across cases by "
                "using the same seed.",
                s["BodySmall"],
            ),
            Spacer(1, 3),
            image(figures["slowing"], height=3.45 * inch),
            para(
                "Figure 2. Median energy versus collision count for deterministic "
                "samples of primary histories from the four simulation families. "
                "Daughter tracks are excluded from this plot; full daughter "
                "histories remain in the binary records.",
                s["BodySmall"],
            ),
            PageBreak(),
            para("4. Criticality and neutron balance", s["SectionHeading"]),
            image(figures["criticality"], height=4.05 * inch),
            para(
                "Figure 3. Project criticality indicator K across moderator "
                "ratios. The dashed line is K = 1 under the requested "
                "new-generation-count criterion. The series show the threshold "
                "crossings summarized in Section 1.",
                s["BodySmall"],
            ),
            image(figures["absorption"], height=4.3 * inch),
            para(
                "Figure 4. Moderator and uranium capture-event counts per "
                "million primaries, followed by fission-to-capture event ratio. "
                "These are reaction tallies, not normalized microscopic "
                "probabilities or time-dependent rates.",
                s["BodySmall"],
            ),
            PageBreak(),
            para("5. Collision residence and detailed mixture results", s["SectionHeading"]),
            image(figures["terminal"], height=3.65 * inch),
            para(
                "Figure 5. Distribution of terminal collision counts for primary "
                "neutrons. This is the available residence proxy: the current "
                "model does not calculate physical elapsed time or free-flight "
                "distance, so it cannot report lifetimes in seconds.",
                s["BodySmall"],
            ),
            para("Representative terminal collision-count statistics", s["SubHeading"]),
        ]
    )
    lifetime_rows = [
        ["Representative case", "Mean collisions", "Median", "90th percentile"]
    ]
    for case, label, _ in representatives:
        stats = collision_statistics(case)
        lifetime_rows.append(
            [
                label,
                f"{stats['mean']:.1f}",
                f"{stats['median']:.0f}",
                f"{stats['p90']:.0f}",
            ]
        )
    story.append(
        make_table(
            [
                [para(str(cell), s["TableCell"]) for cell in row]
                for row in lifetime_rows
            ],
            [3.35 * inch, 1.15 * inch, 0.82 * inch, 1.25 * inch],
        )
    )
    story.extend(
        [
            para("Mixture sweep summary", s["SubHeading"]),
        ]
    )

    mixture_rows = [
        [
            "Mixture",
            "K range",
            "Moderator capture share at ratio 1",
            "at ratio 30",
            "Fission/capture at 1 -> 30",
        ]
    ]
    for label, cases in (
        ("Natural U + H2O", indexed_cases(h2o_rows, "natural")),
        ("2% U + H2O", indexed_cases(h2o_rows, "enriched_2pct")),
        ("Natural U + D2O", indexed_cases(d2o_rows)),
    ):
        first = next(case for case in cases if case["ratio"] == 1)
        last = next(case for case in cases if case["ratio"] == 30)
        mixture_rows.append(
            [
                label,
                f"{min(c['k_project'] for c in cases):.3f}-{max(c['k_project'] for c in cases):.3f}",
                f"{100 * first['moderator_capture_share']:.1f}%",
                f"{100 * last['moderator_capture_share']:.1f}%",
                f"{first['fission_capture_ratio']:.3f} -> {last['fission_capture_ratio']:.3f}",
            ]
        )
    story.append(
        make_table(
            [[para(str(cell), s["TableCell"]) for cell in row] for row in mixture_rows],
            [1.15 * inch, 0.85 * inch, 1.26 * inch, 0.72 * inch, 1.52 * inch],
        )
    )
    story.append(
        para(
            "Across the tested range, the H2O cases show moderator captures "
            "increasing with water ratio. D2O moderator captures remain a small "
            "fraction of total captures in these modeled runs. The fission/"
            "capture ratio is the aggregate count of fission events divided by "
            "capture events for all isotopes in a case.",
            s["BodySmall"],
        )
    )

    story.extend(
        [
            PageBreak(),
            para("6. Interpretation, scope, and next steps", s["SectionHeading"]),
            para("Assignment goals addressed", s["SubHeading"]),
            para(
                "The simulations implement the four requested material "
                "families, one-million-primary runs, energy-dependent reaction "
                "selection, fission offspring, collision histories, final "
                "energy distributions, and parameter sweeps. Under the stated "
                "project rule, the measured K curves show the two unmoderated "
                "uranium cases below threshold and identify H2O and D2O ratio "
                "ranges that cross K = 1.",
                s["BodySmall"],
            ),
            para("Important interpretation", s["SubHeading"]),
            para(
                "The code's recorded new-generation set contains fission "
                "offspring from parents whose fission energy is below 1 eV; "
                "offspring from resonance/fast fission are transported rather "
                "than immediately placed in that set. The transport queue does "
                "not record a generation number. Consequently, the report "
                "applies the user's prescribed count comparison exactly, while "
                "the count remains specifically the code's recorded thermal "
                "daughter-set statistic. A future generation-resolved "
                "calculation should explicitly define whether all fission-born "
                "neutrons or only the thermal source are counted at each "
                "generation.",
                s["BodySmall"],
            ),
            para("Graph and scale limitations", s["SubHeading"]),
            para(
                "Birth-energy and slowing-down graphs are generated from the "
                "recorded histories. The requested lifetime-by-energy concept "
                "is represented only by collision counts: no distances, "
                "velocities along a path, time-of-flight, or physical lifetime "
                "are modeled. The absorption and fission/capture figures show "
                "event tallies and ratios derived from those tallies. The "
                "assignment's one-billion-neutron scale was not run; the "
                "completed scale is one million primaries per case.",
                s["BodySmall"],
            ),
            para("Reproducibility and files", s["SubHeading"]),
            para(
                "All runs used seed 20261016. Single-run results are in "
                "simulation_results/ and natural_uranium_results/. The 60 H2O "
                "cases and 30 D2O cases are indexed by "
                "h2o_uranium_results/index.csv and "
                "d2o_natural_uranium_results/index.csv. Each case directory "
                "contains summary.txt, final_neutrons.csv, new_generation.csv, "
                "history_points.bin, a final-energy histogram CSV, and two "
                "plots. Binary history records are 12 bytes each: 4-byte track "
                "ID, 4-byte collision count, and 4-byte energy in eV.",
                s["BodySmall"],
            ),
        ]
    )
    doc.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)


def main():
    from generate_neutronics_paper import main as generate_paper

    generate_paper()


if __name__ == "__main__":
    main()
