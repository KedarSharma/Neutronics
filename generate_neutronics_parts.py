import csv
import gzip
from io import BytesIO
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import generate_neutronics_report as data


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "FINAL_NEUTRONICS_PROJECT_REPORT.pdf"
PALETTE = {
    "U-238": "#365f91",
    "Natural U": "#2b8a6e",
    "H2O natural": "#d07a24",
    "H2O 2% enriched": "#9b59b6",
    "D2O natural": "#c94c4c",
    "moderator": "#d07a24",
    "fuel": "#365f91",
    "fission": "#c94c4c",
    "capture": "#2b8a6e",
}


def styles():
    sheet = getSampleStyleSheet()
    sheet.add(
        ParagraphStyle(
            name="PartTitle",
            parent=sheet["Title"],
            fontName="Times-Bold",
            fontSize=19,
            leading=23,
            textColor=colors.HexColor("#192f3d"),
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    sheet.add(
        ParagraphStyle(
            name="PartHeading",
            parent=sheet["Heading1"],
            fontName="Times-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#192f3d"),
            spaceBefore=6,
            spaceAfter=4,
            keepWithNext=True,
        )
    )
    sheet.add(
        ParagraphStyle(
            name="PartBody",
            parent=sheet["BodyText"],
            fontName="Times-Roman",
            fontSize=9.5,
            leading=12.2,
            spaceAfter=6,
        )
    )
    sheet.add(
        ParagraphStyle(
            name="PartCaption",
            parent=sheet["BodyText"],
            fontName="Times-Roman",
            fontSize=8,
            leading=9.8,
            textColor=colors.HexColor("#394955"),
            spaceBefore=2,
            spaceAfter=6,
        )
    )
    sheet.add(
        ParagraphStyle(
            name="PartCell",
            parent=sheet["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=9.2,
        )
    )
    return sheet


def make_table(rows, widths):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#344d5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bcc8d0")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f3f5f6")],
                ),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def table(rows, widths, style):
    return make_table(
        [[Paragraph(str(value), style) for value in row] for row in rows],
        widths,
    )


def image(path, max_width=7.0 * inch, max_height=4.0 * inch):
    picture = Image(str(path))
    picture._restrictSize(max_width, max_height)
    return picture


def page_footer(label):
    def draw(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd3d8"))
        canvas.line(0.67 * inch, 0.52 * inch, 7.83 * inch, 0.52 * inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#53616b"))
        canvas.drawString(0.69 * inch, 0.34 * inch, label.upper())
        canvas.restoreState()

    return draw


def draw_cross_sections(isotopes, output_path):
    rows_by_isotope = {isotope: {} for isotope in isotopes}
    with gzip.open(
        data.ROOT / "neutron_data" / "neutron_cross_sections.csv.gz",
        "rt",
        newline="",
        encoding="utf-8",
    ) as source:
        for row in csv.DictReader(source):
            isotope = row["isotope"]
            if isotope not in rows_by_isotope:
                continue
            energy = float(row["energy_eV"])
            value = float(row["cross_section_b"])
            if energy <= 0.0 or value <= 0.0:
                continue
            energies, cross_sections = rows_by_isotope[isotope].setdefault(
                row["display_reaction"], ([], [])
            )
            energies.append(energy)
            cross_sections.append(value)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10.0, 9.3),
        squeeze=False,
        constrained_layout=True,
    )
    reaction_colors = {
        "elastic": "#365f91",
        "inelastic": "#d07a24",
        "capture (n,g)": "#2b8a6e",
        "total fission": "#c94c4c",
    }
    for ax, isotope in zip(axes.flat, isotopes):
        for reaction, (energies, values) in rows_by_isotope[isotope].items():
            order = np.argsort(energies)
            ax.plot(
                np.asarray(energies)[order],
                np.asarray(values)[order],
                linewidth=0.85,
                label=reaction,
                color=reaction_colors.get(reaction),
            )
        ax.set_title(isotope)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Neutron energy (eV)")
        ax.set_ylabel("Cross section (barn)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
    axes.flat[-1].axis("off")
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def draw_birth_energy(output_path):
    energy_mev = data.birth_energy_plot(
        data.ROOT / "simulation_results", output_path
    )
    return energy_mev


def draw_case_energy_histogram(cases, output_path):
    bins = np.logspace(-5, 8, 66)
    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    for case, label, color in cases:
        energies = []
        with (case["directory"] / "final_neutrons.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            energies = [
                float(row["energy_ev"]) for row in csv.DictReader(source)
            ]
        if len(energies) != data.PRIMARY_COUNT:
            raise RuntimeError(
                f"Expected {data.PRIMARY_COUNT} primary terminal energies "
                f"for {case['name']}; found {len(energies)}"
            )
        ax.hist(
            energies,
            bins=bins,
            weights=np.full(len(energies), 1.0 / data.PRIMARY_COUNT),
            histtype="step",
            linewidth=1.25,
            label=label,
            color=color,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy before capture or fission (eV)")
    ax.set_ylabel("Fraction of primary neutrons per bin")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def draw_case_dynamics(cases, output_path):
    data.history_dynamics_plot(cases, output_path)


def draw_collision_counts(cases, output_path):
    data.terminal_collision_plot(cases, output_path)


def draw_moderator_results(series, output_path):
    fig, (k_ax, capture_ax, ratio_ax) = plt.subplots(
        3,
        1,
        figsize=(9.0, 8.4),
        sharex=True,
        constrained_layout=True,
    )
    for cases, label, color in series:
        ordered = data.indexed_cases(cases)
        x = [case["ratio"] for case in ordered]
        k_ax.plot(
            x,
            [case["k_project"] for case in ordered],
            marker="o",
            markersize=2.7,
            linewidth=1.25,
            label=label,
            color=color,
        )
        capture_ax.plot(
            x,
            [
                1.0e6 * case["moderator_captures"] / case["initial"]
                for case in ordered
            ],
            linewidth=1.25,
            label=f"{label}: moderator",
            color=color,
        )
        capture_ax.plot(
            x,
            [
                1.0e6 * case["fuel_captures"] / case["initial"]
                for case in ordered
            ],
            linewidth=1.1,
            linestyle=":",
            label=f"{label}: uranium",
            color=color,
        )
        ratio_ax.plot(
            x,
            [case["fission_capture_ratio"] for case in ordered],
            linewidth=1.25,
            marker="o",
            markersize=2.5,
            label=label,
            color=color,
        )
    k_ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    k_ax.set_ylabel(r"$K_{\infty}$")
    k_ax.set_title("Project-defined neutron reproduction index")
    k_ax.legend(frameon=False, fontsize=7, ncol=2)
    capture_ax.set_ylabel("Capture events per million primaries")
    capture_ax.legend(frameon=False, fontsize=6.8, ncol=2)
    ratio_ax.set_xlabel("Moderator molecules per uranium atom")
    ratio_ax.set_ylabel("Fission / capture events")
    ratio_ax.legend(frameon=False, fontsize=7, ncol=2)
    for ax in (k_ax, capture_ax, ratio_ax):
        ax.grid(True, alpha=0.2)
        ax.set_xlim(1, 30)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def draw_isotope_fission_trends(cases, output_path):
    ordered = data.indexed_cases(cases)
    ratios = [case["ratio"] for case in ordered]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 2.7),
        sharex=True,
        constrained_layout=True,
    )
    for ax, isotope, color in zip(
        axes,
        ("U-235", "U-238"),
        (PALETTE["fission"], PALETTE["U-238"]),
    ):
        ax.plot(
            ratios,
            [
                1_000_000
                * case["collisions"][isotope]["fission"]
                / case["initial"]
                for case in ordered
            ],
            color=color,
            marker="o",
            markersize=2.5,
            linewidth=1.25,
        )
        ax.set_title(f"{isotope} fissions")
        ax.set_ylabel("Events per million primaries")
        ax.set_xlabel("Moderator molecules per uranium atom")
        ax.set_xlim(1, 30)
        ax.set_xticks((1, 5, 10, 15, 20, 25, 30))
        ax.grid(True, alpha=0.2)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def part_summary_table(cases, style):
    rows = [
        [
            "Composition",
            "Tracks",
            "Capture events",
            "Fission events",
            "Fission/capture",
            "K∞",
        ]
    ]
    for case in cases:
        rows.append(
            [
                case["name"],
                data.fmt(case["tracks"]),
                data.fmt(case["captures"]),
                data.fmt(case["fissions"]),
                f"{case['fission_capture_ratio']:.3f}",
                f"{case['k_project']:.3f}",
            ]
        )
    return table(
        rows,
        [
            1.45 * inch,
            0.8 * inch,
            1.2 * inch,
            1.1 * inch,
            1.1 * inch,
            0.55 * inch,
        ],
        style,
    )


def create_part_pdf(path, heading, paragraphs, story_items, label):
    sheet = styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.67 * inch,
        rightMargin=0.67 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.68 * inch,
        title=heading,
        author="Neutronics Simulation Study",
    )
    story = [
        Paragraph(heading, sheet["PartTitle"]),
    ]
    for text in paragraphs:
        story.append(Paragraph(text, sheet["PartBody"]))
    for item in story_items(sheet):
        story.append(item)
    footer = page_footer(label)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def create_part_a(path, figure_path):
    def content(sheet):
        yield Paragraph("Energy-dependent cross sections", sheet["PartHeading"])
        yield Paragraph(
            "These isotope-specific cross sections determine the relative "
            "probabilities of elastic scattering, inelastic scattering, "
            "capture, and fission throughout the four simulations.",
            sheet["PartBody"],
        )
        yield image(figure_path, max_height=7.0 * inch)
        yield Paragraph(
            "Figure 1. Cross sections for all modeled isotopes: H-1, H-2, "
            "O-16, U-235, and U-238. Both axes are logarithmic. Only reaction "
            "channels with supplied data are shown.",
            sheet["PartCaption"],
        )

    create_part_pdf(
        path,
        "A. Cross-Section Plots",
        [
            "The five isotope plots below provide the nuclear-data context "
            "for Parts B-E."
        ],
        content,
        "A | CROSS-SECTION PLOTS",
    )


def create_paper_front_matter(path):
    sheet = styles()
    centered = ParagraphStyle(
        name="PaperSubtitle",
        parent=sheet["PartBody"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#53616b"),
        spaceAfter=18,
    )
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.75 * inch,
        title="Neutron Transport in Uranium Systems: A Four-Part Study",
        author="Neutronics Simulation Study",
        subject="Neutron moderation, energy histories, and reaction balance",
    )
    story = [
        Spacer(1, 0.7 * inch),
        Paragraph("Neutron Transport in Uranium Systems", sheet["PartTitle"]),
        Paragraph(
            "A four-part study of uranium and water-moderated mixtures",
            centered,
        ),
        Paragraph("Abstract", sheet["PartHeading"]),
        Paragraph(
            "Monte Carlo neutron histories were simulated for pure U-238, "
            "unmoderated natural uranium, H2O with natural and 2%-enriched "
            "uranium, and D2O with natural uranium. The modeled medium is "
            "uniform and infinite. One million primary neutrons were tracked "
            "per case, with isotope and reaction selection weighted by atomic "
            "abundance and energy-dependent cross sections. For this study, "
            "K<sub>∞</sub> is the recorded thermal-fission daughter count "
            "divided by the initial neutron count. The unmoderated cases had "
            "K<sub>∞</sub> = 0. In the moderated cases, K<sub>∞</sub> reached "
            "unity between ratios 4-5 for natural U-H2O, 20-21 for "
            "2%-enriched U-H2O, and 3-4 for natural U-D2O. H2O moderator "
            "captures increased strongly with water ratio, while D2O "
            "moderator captures remained below 1%.",
            sheet["PartBody"],
        ),
        Paragraph("Introduction", sheet["PartHeading"]),
        Paragraph(
            "Neutron reaction balance depends on neutron energy, isotope "
            "composition, and moderator. This study compares unmoderated "
            "U-238 and natural uranium with H2O-moderated natural and "
            "2%-enriched uranium and D2O-moderated natural uranium. The "
            "analysis focuses on collision-history trends, isotope-resolved "
            "fission, capture partitioning, and the study-defined "
            "K<sub>∞</sub> threshold across moderator ratios. Established "
            "Monte Carlo transport publications provide precedent for "
            "reporting model scope and nuclear-data choices [1, 2].",
            sheet["PartBody"],
        ),
        Paragraph("Computational method and scope", sheet["PartHeading"]),
        Paragraph(
            "Collision histories are followed in a homogeneous, infinite "
            "mixture without spatial coordinates, leakage, or flight "
            "distances. At each collision, isotope and reaction probabilities "
            "are sampled from atom abundances and the available partial "
            "cross sections. The model uses pointwise KAERI NuChart "
            "ENDF/B-VII.1 data [3], with log-log interpolation between tabulated "
            "energies and endpoint-slope extrapolation above the tabulated "
            "range. Results are accumulated by isotope and reaction and "
            "reported for the specified 1–30 moderator-to-uranium ratios. "
            "Established transport-code publications emphasize documenting "
            "physics options and nuclear data [1, 2]; this study reports its "
            "model scope and tally definitions explicitly.",
            sheet["PartBody"],
        ),
        Paragraph(
            "The analysis is descriptive: no replicate-based statistical "
            "uncertainties, convergence study, or benchmark comparison are "
            "reported. Conclusions therefore refer to trends within this "
            "implemented model. The benchmark literature provides a "
            "framework for a future matched validation study [4].",
            sheet["PartBody"],
        ),
        Paragraph("Notation", sheet["PartHeading"]),
        Paragraph(
            "K<sub>∞</sub> = N<sub>g</sub>/N<sub>0</sub>, where "
            "N<sub>g</sub> is the recorded set of daughters from fissions "
            "below 1 eV and N<sub>0</sub> is the initial neutron count. "
            "Unity is the criticality threshold used in this study.",
            sheet["PartBody"],
        ),
    ]
    footer = page_footer("NEUTRON TRANSPORT IN URANIUM SYSTEMS")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def create_part_one(path, u238, figures):
    def content(sheet):
        yield Paragraph("1. Material and reaction results", sheet["PartHeading"])
        yield Paragraph(
            "The medium contains only U-238. "
            f"{data.fmt(u238['fissions'])} fission and "
            f"{data.fmt(u238['captures'])} capture events were recorded across "
            "primary and daughter tracks. Fission occurred in the fast and "
            "resonance regions; no thermal fission occurred, giving "
            "K<sub>∞</sub> = 0. The energy histories show a sharp initial "
            "energy decrease followed by continued slowing over later "
            "collisions.",
            sheet["PartBody"],
        )
        yield part_summary_table([u238], sheet["PartCell"])
        yield Spacer(1, 6)
        yield Paragraph("2. Neutron source and energy history", sheet["PartHeading"])
        yield image(figures["u238_birth"], max_height=2.5 * inch)
        yield Paragraph(
            "Figure 2. Sampled primary birth spectrum and its specified "
            "distribution.",
            sheet["PartCaption"],
        )
        yield PageBreak()
        yield Paragraph("3. Energy histories and terminal state", sheet["PartHeading"])
        yield image(figures["u238_dynamics"], max_height=3.35 * inch)
        yield Paragraph(
            "Figure 3. Sampled primary-neutron collision histories and median "
            "energy trend. Strong energy reduction occurs early in the track.",
            sheet["PartCaption"],
        )
        yield image(figures["u238_terminal"], max_height=2.7 * inch)
        yield Paragraph(
            "Figure 4. Primary-neutron energies immediately before capture "
            "or fission. All values belong to the initial population.",
            sheet["PartCaption"],
        )
        yield image(figures["u238_collisions"], max_height=2.25 * inch)
        yield Paragraph(
            "Figure 5. Number of collisions before primary-neutron termination; "
            "this is not a physical lifetime.",
            sheet["PartCaption"],
        )

    create_part_pdf(
        path,
        "B. U-238 Only",
        [
            "The unmoderated U-238 case provides a reference for neutron "
            "slowing and fast-region fission in the absence of hydrogenous "
            "moderation."
        ],
        content,
        "B | U-238 ONLY",
    )


def create_part_two(path, natural, figures):
    isotope_rows = [[
        "Uranium isotope",
        "Elastic",
        "Inelastic",
        "Capture",
        "Fission",
        "Fission region",
    ]]
    for isotope in ("U-235", "U-238"):
        counts = natural["collisions"][isotope]
        thermal = natural["summary"].get(f"{isotope}_fission_thermal", 0)
        resonance = natural["summary"].get(f"{isotope}_fission_resonance", 0)
        fast = natural["summary"].get(f"{isotope}_fission_fast", 0)
        dominant = max(
            (("thermal", thermal), ("resonance", resonance), ("fast", fast)),
            key=lambda item: item[1],
        )[0]
        isotope_rows.append(
            [
                isotope,
                data.fmt(counts["elastic"]),
                data.fmt(counts["inelastic"]),
                data.fmt(counts["capture"]),
                data.fmt(counts["fission"]),
                dominant,
            ]
        )

    def content(sheet):
        yield Paragraph("1. Material and model", sheet["PartHeading"])
        yield Paragraph(
            "Natural uranium is represented by U-235:U-238 = 1:138. The "
            "recorded thermal-fission daughter count is zero "
            "(K<sub>∞</sub> = 0). U-235 accounts for "
            f"{data.fmt(natural['collisions']['U-235']['fission'])} fissions, "
            "predominantly in the resonance range, while U-238 fission is "
            "predominantly fast in this run. Thus the minority U-235 isotope "
            "contributes a substantial resonance-fission component, whereas "
            "U-238 fission remains mainly fast.",
            sheet["PartBody"],
        )
        yield Paragraph("2. Isotope-specific reaction balance", sheet["PartHeading"])
        yield table(
            isotope_rows,
            [1.15 * inch, 0.82 * inch, 0.85 * inch, 0.82 * inch, 0.82 * inch, 0.9 * inch],
            sheet["PartCell"],
        )
        yield Spacer(1, 6)
        yield part_summary_table([natural], sheet["PartCell"])
        yield Spacer(1, 6)
        yield Paragraph("3. Energy histories", sheet["PartHeading"])
        yield image(figures["natural_dynamics"], max_height=3.45 * inch)
        yield Paragraph(
            "Figure 6. Sampled primary-neutron energy histories and median "
            "energy versus collision count in natural uranium.",
            sheet["PartCaption"],
        )
        yield PageBreak()
        yield Paragraph("4. Terminal distribution", sheet["PartHeading"])
        yield image(figures["natural_terminal"], max_height=4.0 * inch)
        yield Paragraph(
            "Figure 7. Terminal primary-neutron energies before capture or "
            "fission. The distributions retain substantial high-energy "
            "population because the modeled medium contains no moderator.",
            sheet["PartCaption"],
        )
        yield image(figures["natural_collisions"], max_height=3.0 * inch)
        yield Paragraph(
            "Figure 8. Primary-neutron collision counts before termination.",
            sheet["PartCaption"],
        )

    create_part_pdf(
        path,
        "C. Natural Uranium",
        [
            "Unmoderated natural uranium provides an isotope-resolved "
            "comparison with the pure U-238 case."
        ],
        content,
        "C | NATURAL URANIUM",
    )


def create_part_three(path, h2o_rows, figures):
    def content(sheet):
        yield Paragraph("1. Material and model", sheet["PartHeading"])
        yield Paragraph(
            "Ordinary water was represented by two H-1 atoms and one O-16 "
            "atom per molecule. Natural uranium (U-235:U-238 = 1:138) and "
            "nominally 2%-enriched uranium (1:49) were each examined from "
            "1 to 30 water molecules per uranium atom. K<sub>∞</sub> is the "
            "recorded thermal-fission daughter count divided by the initial "
            "neutron count; the unity line marks the study's criticality "
            "threshold.",
            sheet["PartBody"],
        )
        yield Paragraph("2. Moderator-ratio response", sheet["PartHeading"])
        yield image(figures["h2o_trends"], max_height=4.45 * inch)
        yield Paragraph(
            "Figure 9. Project-defined K∞, moderator and uranium capture "
            "events per million primaries, and total fission-to-capture "
            "event ratio across both H2O branches.",
            sheet["PartCaption"],
        )
        yield Paragraph("3. Observed trends", sheet["PartHeading"])
        yield Paragraph(
            "Both K<sub>∞</sub> curves decline as the H2O ratio increases, "
            "but the enriched branch remains higher throughout the sweep. "
            "Natural uranium falls from 1.3480 at ratio 1 to 0.4017 at "
            "ratio 30 and passes unity between ratios 4 and 5 (0.9994 at "
            "ratio 5). The 2%-enriched branch falls from 1.8027 to 0.8322 "
            "and passes unity between ratios 20 and 21 (0.9911 at ratio 21). "
            "The two branches therefore show substantially different "
            "moderator ranges around the unity threshold.",
            sheet["PartBody"],
        )
        yield Paragraph(
            "The capture curves show the reaction balance moving away from "
            "uranium and toward water as the moderator fraction increases. "
            "Moderator capture rises from 15.5% to 84.8% of all captures "
            "for natural uranium and from 11.4% to 79.7% for the enriched "
            "mixture. At the same time, the fission-to-capture event ratio "
            "decreases from 1.111 to 0.193 and from 2.181 to 0.500, "
            "respectively. Thus the changing K<sub>∞</sub> response occurs "
            "alongside a strong shift in where neutron captures are recorded.",
            sheet["PartBody"],
        )
        yield Paragraph(
            "The isotope plots resolve the fission response behind these "
            "curves. From ratio 1 to 30, U-235 fissions per million primary "
            "neutrons decrease from 542,794 to 160,814 in natural uranium "
            "and from 731,542 to 333,296 in the enriched mixture. U-238 "
            "fissions decrease more sharply: 41,719 to 1,907 and 41,651 to "
            "1,883, respectively. The higher U-235 fission counts in the "
            "enriched branch persist across the full range, while the "
            "reduction in U-238 fission is similar in both compositions.",
            sheet["PartBody"],
        )
        yield PageBreak()
        yield Paragraph("4. Isotope-specific fission: natural uranium", sheet["PartHeading"])
        yield image(figures["h2o_natural_fissions"], max_height=2.15 * inch)
        yield Paragraph(
            "Figure 11. U-235 and U-238 fission events per million primary "
            "neutrons across the H2O ratio sweep. Both isotope-specific "
            "curves decrease, with the steeper decline in U-238.",
            sheet["PartCaption"],
        )
        yield Paragraph(
            "In the natural-uranium series, U-235 fissions fall by about "
            "70% between the lowest and highest water ratios. U-238 fissions "
            "fall by about 95%, from 41,719 to 1,907 events per million "
            "primaries. The curves separate the isotope contributions and "
            "show that the reduction is not uniform across the two uranium "
            "isotopes. U-235's share of the two isotope-specific fission "
            "counts rises from about 92.9% to 98.8% over the sweep. This "
            "change accompanies the decline in K<sub>∞</sub> and the rising "
            "share of captures in the moderator.",
            sheet["PartBody"],
        )
        yield PageBreak()
        yield Paragraph("5. Isotope-specific fission: 2%-enriched uranium", sheet["PartHeading"])
        yield image(figures["h2o_enriched_fissions"], max_height=2.15 * inch)
        yield Paragraph(
            "Figure 12. U-235 and U-238 fission events per million primary "
            "neutrons across the H2O ratio sweep. The enriched composition "
            "has higher U-235 fission counts throughout.",
            sheet["PartCaption"],
        )
        yield Paragraph(
            "U-235 fissions decrease from 731,542 to 333,296 events per "
            "million primaries, a smaller proportional reduction than in "
            "the natural-uranium series. U-238 fissions fall from 41,651 to "
            "1,883, closely matching the high-ratio endpoint of the natural "
            "uranium case. The persistent separation in U-235 counts reflects "
            "the distinct uranium compositions in these two sweeps, while "
            "both exhibit declining isotope-specific fission with increasing "
            "H2O ratio. U-235 accounts for about 94.6% of the two isotope "
            "fission counts at ratio 1 and 99.4% at ratio 30; the U-238 "
            "contribution becomes progressively smaller.",
            sheet["PartBody"],
        )

    create_part_pdf(
        path,
        "D. H2O-Moderated Uranium",
        [
            "The two light-water branches compare natural uranium "
            "(U-235:U-238 = 1:138) with nominally 2%-enriched uranium (1:49)."
        ],
        content,
        "D | H2O-MODERATED URANIUM",
    )


def create_part_four(path, d2o_rows, figures):
    def content(sheet):
        yield Paragraph("1. Material and model", sheet["PartHeading"])
        yield Paragraph(
            "Heavy water was represented by two H-2 atoms and one O-16 atom "
            "per molecule, mixed with natural uranium at U-235:U-238 = 1:138. "
            "The moderator ratio spans 1-30 D2O molecules per uranium atom. "
            "K<sub>∞</sub> is the recorded thermal-fission daughter count "
            "per initial neutron; unity is the criticality threshold used "
            "in this study.",
            sheet["PartBody"],
        )
        yield Paragraph("2. Moderator-ratio response", sheet["PartHeading"])
        yield image(figures["d2o_trends"], max_height=4.6 * inch)
        yield Paragraph(
            "Figure 10. Project-defined K∞, moderator and uranium capture "
            "events per million primaries, and total fission-to-capture "
            "event ratio for natural uranium moderated by D2O.",
            sheet["PartCaption"],
        )
        yield Paragraph("3. Observed trends", sheet["PartHeading"])
        yield Paragraph(
            "Unlike the declining H2O curves, D2O K<sub>∞</sub> increases "
            "throughout this sweep, rising from 0.6746 at ratio 1 to 1.2759 "
            "at ratio 30. It passes unity between ratios 3 and 4; the closest "
            "tested sub-unity value is 0.9944 at ratio 3. This steady increase "
            "is accompanied by a rising fission-to-capture event ratio, from "
            "0.431 to 1.036. The curves therefore show a progressive shift "
            "in the modeled reaction balance as D2O content increases.",
            sheet["PartBody"],
        )
        yield Paragraph(
            "The capture plot distinguishes D2O from H2O: moderator captures "
            "increase from 143 to 4,171 events per million primary neutrons, "
            "but remain a small share of total captures, rising from about "
            "0.02% to 0.84%. Uranium captures decrease from 907,215 to "
            "493,689 per million over the same interval. Most recorded "
            "captures therefore remain in uranium even at the largest D2O "
            "ratio, unlike the water-dominated capture balance at high H2O "
            "ratios.",
            sheet["PartBody"],
        )
        yield PageBreak()
        yield Paragraph("4. Isotope-specific fission response", sheet["PartHeading"])
        yield image(figures["d2o_fissions"], max_height=2.15 * inch)
        yield Paragraph(
            "Figure 13. U-235 and U-238 fission events per million primary "
            "neutrons across the D2O ratio sweep.",
            sheet["PartCaption"],
        )
        yield Paragraph(
            "U-235 fissions increase from 339,347 to 513,555 events per "
            "million primaries, while U-238 fissions decline from 51,298 to "
            "2,070. U-235's share of the two isotope-specific fission counts "
            "therefore grows from about 86.9% to 99.6%. The opposing isotope "
            "trends accompany the rise in K<sub>∞</sub> and in the overall "
            "fission-to-capture ratio, while uranium continues to account "
            "for most capture events. Together, the capture and fission "
            "plots show that increased D2O content changes both the balance "
            "of capture locations and the isotope distribution of fission.",
            sheet["PartBody"],
        )

    create_part_pdf(
        path,
        "E. D2O-Moderated Natural Uranium",
        [
            "This part examines how the heavy-water moderator changes the "
            "reaction balance in natural uranium."
        ],
        content,
        "E | D2O-MODERATED NATURAL URANIUM",
    )


def create_conclusion(path):
    def content(sheet):
        yield Paragraph("Conclusion", sheet["PartHeading"])
        yield Paragraph(
            "The simulations show distinct moderation trends for the "
            "compositions examined. In the two H2O series, increasing the "
            "water-to-uranium ratio lowers the recorded K<sub>∞</sub> and "
            "shifts an increasing share of captures from uranium to the "
            "moderator. The unity crossing occurs at ratios 4–5 for natural "
            "uranium and 20–21 for the 2%-enriched mixture. The isotope "
            "tallies show decreasing U-235 and U-238 fissions in both "
            "series, with U-238 fission decreasing most sharply.",
            sheet["PartBody"],
        )
        yield Paragraph(
            "The D2O natural-uranium sweep follows a different pattern: "
            "K<sub>∞</sub> rises from 0.6746 to 1.2759 and crosses unity "
            "between ratios 3 and 4, while moderator capture remains below "
            "1% of total captures. Across that sweep, U-235 fissions increase "
            "and U-238 fissions decrease. Taken together, these results "
            "identify moderator type and uranium enrichment as important "
            "variables in the reaction balances produced by this model.",
            sheet["PartBody"],
        )
        yield Paragraph(
            "These findings are bounded by the homogeneous, infinite "
            "collision-history model and its supplied reaction channels. "
            "The next steps for a stronger neutronics study are to quantify "
            "sampling uncertainty and convergence and to compare a suitably "
            "matched model with evaluated criticality benchmarks.",
            sheet["PartBody"],
        )
        yield Paragraph("AI-use note", sheet["PartHeading"])
        yield Paragraph(
            "The code and report were generated with AI assistance and "
            "reviewed by a human at each stage.",
            sheet["PartBody"],
        )

    create_part_pdf(
        path,
        "F. Conclusion",
        [],
        content,
        "F | CONCLUSION",
    )


def main():
    u238, natural, h2o_rows, d2o_rows = data.load_all_cases()
    with tempfile.TemporaryDirectory(prefix="neutronics-part-reports-") as temp:
        temp_path = Path(temp)
        figure_dir = temp_path / "figures"
        figure_dir.mkdir()
        figures = {
            name: figure_dir / f"{name}.png"
            for name in (
                "all_xs",
                "u238_birth",
                "u238_dynamics",
                "u238_terminal",
                "u238_collisions",
                "natural_dynamics",
                "natural_terminal",
                "natural_collisions",
                "h2o_trends",
                "d2o_trends",
                "h2o_natural_fissions",
                "h2o_enriched_fissions",
                "d2o_fissions",
            )
        }
        draw_cross_sections(
            ("H-1", "H-2", "O-16", "U-235", "U-238"),
            figures["all_xs"],
        )
        data.birth_energy_plot(
            data.ROOT / "simulation_results", figures["u238_birth"]
        )
        draw_case_dynamics(
            [(u238, "Pure U-238", PALETTE["U-238"])],
            figures["u238_dynamics"],
        )
        draw_case_energy_histogram(
            [(u238, "Pure U-238", PALETTE["U-238"])],
            figures["u238_terminal"],
        )
        draw_collision_counts(
            [(u238, "Pure U-238", PALETTE["U-238"])],
            figures["u238_collisions"],
        )
        draw_case_dynamics(
            [(natural, "Natural uranium", PALETTE["Natural U"])],
            figures["natural_dynamics"],
        )
        draw_case_energy_histogram(
            [(natural, "Natural uranium", PALETTE["Natural U"])],
            figures["natural_terminal"],
        )
        draw_collision_counts(
            [(natural, "Natural uranium", PALETTE["Natural U"])],
            figures["natural_collisions"],
        )
        draw_moderator_results(
            [
                (
                    data.indexed_cases(h2o_rows, "natural"),
                    "Natural U + H2O",
                    PALETTE["H2O natural"],
                ),
                (
                    data.indexed_cases(h2o_rows, "enriched_2pct"),
                    "2% enriched U + H2O",
                    PALETTE["H2O 2% enriched"],
                ),
            ],
            figures["h2o_trends"],
        )
        draw_isotope_fission_trends(
            data.indexed_cases(h2o_rows, "natural"),
            figures["h2o_natural_fissions"],
        )
        draw_isotope_fission_trends(
            data.indexed_cases(h2o_rows, "enriched_2pct"),
            figures["h2o_enriched_fissions"],
        )
        draw_moderator_results(
            [
                (
                    data.indexed_cases(d2o_rows),
                    "Natural U + D2O",
                    PALETTE["D2O natural"],
                )
            ],
            figures["d2o_trends"],
        )
        draw_isotope_fission_trends(
            data.indexed_cases(d2o_rows),
            figures["d2o_fissions"],
        )
        front_matter = temp_path / "front_matter.pdf"
        create_paper_front_matter(front_matter)
        part_files = [temp_path / f"section_{letter}.pdf" for letter in "abcdef"]
        create_part_a(part_files[0], figures["all_xs"])
        create_part_one(part_files[1], u238, figures)
        create_part_two(part_files[2], natural, figures)
        create_part_three(part_files[3], h2o_rows, figures)
        create_part_four(part_files[4], d2o_rows, figures)
        create_conclusion(part_files[5])

        writer = PdfWriter()
        for part_file in [front_matter, *part_files]:
            reader = PdfReader(str(part_file))
            for page in reader.pages:
                writer.add_page(page)
        for page_number, page in enumerate(writer.pages, start=1):
            number_overlay = BytesIO()
            number_canvas = pdf_canvas.Canvas(number_overlay, pagesize=letter)
            number_canvas.setFont("Helvetica", 7)
            number_canvas.setFillColor(colors.HexColor("#53616b"))
            number_canvas.drawRightString(
                7.81 * inch, 0.34 * inch, str(page_number)
            )
            number_canvas.save()
            number_overlay.seek(0)
            page.merge_page(PdfReader(number_overlay).pages[0])
        writer.add_metadata(
            {
                "/Title": "Neutron Transport in Uranium Systems: A Four-Part Study",
                "/Author": "Neutronics Simulation Study",
                "/Subject": "Part-by-part results for uranium and water mixtures",
            }
        )
        with REPORT_PATH.open("wb") as output:
            writer.write(output)

    print(f"Combined four-part report: {REPORT_PATH}")
    print(f"Report size: {REPORT_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
