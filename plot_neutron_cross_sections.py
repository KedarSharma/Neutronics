import csv
import argparse
import bisect
import gzip
import math
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import matplotlib.pyplot as plt


DATASET = "ENDF/B-VII.1"
DATABASE_ID = "endfb7.1"
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "neutron_data"
PLOT_DIR = OUTPUT_DIR / "neutron_plots"
ENERGY_MIN_EV = 1e-5
ENERGY_MAX_EV = 1e8
NEAR_ZERO_XS_B = 1e-10
H2_INELASTIC_FADE_ENERGY_EV = 3e6
H2_INELASTIC_DATA_START_EV = 3.4e6

REACTIONS = {
    2: ("elastic", "el"),
    3: ("inelastic", "non"),
    4: ("inelastic", "inl"),
    18: ("total fission", "total fission"),
    102: ("capture (n,g)", "g"),
}

ISOTOPES = {
    "H-1": (1, 1, 125, (2, 102)),
    "H-2": (1, 2, 128, (2, 3, 102)),
    "O-16": (8, 16, 825, (2, 4, 102)),
    "U-235": (92, 235, 9228, (2, 4, 18, 102)),
    "U-238": (92, 238, 9237, (2, 4, 18, 102)),
}

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
DATA_ROW = re.compile(rf"^\s*({NUMBER})\s+({NUMBER})\s*$")
TITLE = re.compile(r"([A-Z][a-z]?-\d+)\(n,([^)]+)\)\s*(ENDF/B-[\w.]+)")


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"br", "p", "div"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def fetch_record(isotope, z, mass, index, mt):
    target = f"{DATABASE_ID},{z},{mass},{index},3,{mt}"
    url = f"https://atom.kaeri.re.kr/nuchart/getData.jsp?target={target}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = _TextExtractor()
    parser.feed(html)
    text = "\n".join(parser.parts)
    title_match = TITLE.search(text)
    if not title_match:
        raise ValueError(f"Could not identify the isotope/reaction in KAERI response: {url}")

    response_isotope, response_reaction, response_dataset = title_match.groups()
    if response_isotope != isotope or response_dataset != DATASET:
        raise ValueError(
            f"Unexpected KAERI record for {isotope} MT {mt}: "
            f"{response_isotope} {response_dataset} ({url})"
        )

    expected_reaction = REACTIONS[mt][1]
    if response_reaction.lower() != expected_reaction:
        raise ValueError(
            f"Expected reaction {expected_reaction!r}, got {response_reaction!r} "
            f"for {isotope} MT {mt} ({url})"
        )

    points = []
    for line in text.splitlines():
        match = DATA_ROW.match(line)
        if not match:
            continue
        energy_ev, cross_section_b = map(float, match.groups())
        if not math.isfinite(energy_ev) or not math.isfinite(cross_section_b):
            raise ValueError(f"Non-finite data value in {url}")
        if energy_ev <= 0:
            raise ValueError(f"Non-positive energy cannot be plotted on a log axis: {url}")
        points.append((energy_ev, cross_section_b))

    if not points:
        raise ValueError(f"No energy/cross-section rows found in KAERI response: {url}")
    points.sort(key=lambda point: point[0])

    return {
        "isotope": isotope,
        "mt": mt,
        "reaction": REACTIONS[mt][0],
        "source_reaction": response_reaction,
        "url": url,
        "points": points,
    }


def plot_points_for_record(record):
    positive_points = [
        (energy, cross_section)
        for energy, cross_section in record["points"]
        if cross_section > 0
    ]
    if not positive_points:
        return []

    if record["isotope"] == "H-2" and record["mt"] == 3:
        positive_points = [
            point for point in positive_points
            if point[0] >= H2_INELASTIC_DATA_START_EV
        ]
        if not positive_points:
            return []
        positive_points.insert(0, (H2_INELASTIC_FADE_ENERGY_EV, NEAR_ZERO_XS_B))
    elif record["mt"] == 4 and positive_points[0][0] > ENERGY_MIN_EV:
        fade_energy = max(ENERGY_MIN_EV, positive_points[0][0] * 0.9)
        positive_points.insert(0, (fade_energy, NEAR_ZERO_XS_B))

    if positive_points[-1][0] < ENERGY_MAX_EV:
        previous_energy, previous_xs = positive_points[-2] if len(positive_points) > 1 else positive_points[-1]
        last_energy, last_xs = positive_points[-1]
        if previous_energy < last_energy:
            slope = math.log(last_xs / previous_xs) / math.log(last_energy / previous_energy)
            extrapolated_xs = last_xs * (ENERGY_MAX_EV / last_energy) ** slope
            if math.isfinite(extrapolated_xs) and extrapolated_xs > 0:
                positive_points.append((ENERGY_MAX_EV, extrapolated_xs))

    if positive_points[0][0] < ENERGY_MIN_EV:
        boundary_xs = interpolate_log_log(positive_points, ENERGY_MIN_EV)
        positive_points = [(ENERGY_MIN_EV, boundary_xs)] + [
            point for point in positive_points if point[0] > ENERGY_MIN_EV
        ]

    if positive_points[-1][0] > ENERGY_MAX_EV:
        boundary_xs = interpolate_log_log(positive_points, ENERGY_MAX_EV)
        positive_points = [
            point for point in positive_points if point[0] < ENERGY_MAX_EV
        ] + [(ENERGY_MAX_EV, boundary_xs)]

    return positive_points


def interpolate_log_log(points, energy):
    upper = bisect.bisect_left(points, (energy, -math.inf))
    if upper < len(points) and points[upper][0] == energy:
        return points[upper][1]
    if upper == 0 or upper == len(points):
        raise ValueError(f"Cannot interpolate outside the plotted data range at {energy:g} eV")

    low_energy, low_xs = points[upper - 1]
    high_energy, high_xs = points[upper]
    fraction = math.log(energy / low_energy) / math.log(high_energy / low_energy)
    return math.exp(math.log(low_xs) + fraction * math.log(high_xs / low_xs))


def total_cross_section_points(curves):
    energy_grid = sorted({energy for points in curves for energy, _ in points})
    total_points = []
    for energy in energy_grid:
        total_xs = 0.0
        for points in curves:
            if energy < points[0][0] or energy > points[-1][0]:
                continue
            total_xs += interpolate_log_log(points, energy)
        if total_xs > 0:
            total_points.append((energy, total_xs))
    return total_points


def classify_plot_point(record, energy, cross_section, source_points):
    if (record["isotope"] == "H-2" and record["mt"] == 3
            and energy == H2_INELASTIC_FADE_ENERGY_EV
            and cross_section == NEAR_ZERO_XS_B):
        return "near_zero_floor"
    if (record["mt"] == 4 and energy < record["points"][0][0]
            and cross_section == NEAR_ZERO_XS_B):
        return "near_zero_floor"
    if (energy, cross_section) in source_points:
        return "source"
    return "high_energy_extrapolation"


def load_cached_records(csv_path):
    cached = {isotope: {mt: [] for mt in config[3]} for isotope, config in ISOTOPES.items()}
    metadata = {}
    with gzip.open(csv_path, "rt", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            isotope = row["isotope"]
            mt = int(row["mt"])
            if isotope not in cached or mt not in cached[isotope]:
                continue
            key = (isotope, mt)
            metadata[key] = {
                "isotope": isotope,
                "mt": mt,
                "reaction": REACTIONS[mt][0],
                "source_reaction": row.get("source_reaction", ""),
                "url": row["source_url"],
            }
            cached[isotope][mt].append(
                (float(row["energy_eV"]), float(row["cross_section_b"]))
            )

    records_by_isotope = {}
    for isotope, reactions in cached.items():
        records = []
        for mt, points in reactions.items():
            if not points:
                raise ValueError(f"Cached raw data is missing {isotope} MT {mt}")
            record = metadata[(isotope, mt)]
            record["points"] = points
            records.append(record)
        records_by_isotope[isotope] = records
    return records_by_isotope


def main(refresh=False):
    DATA_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    csv_path = DATA_DIR / "neutron_cross_sections.csv.gz"
    if csv_path.exists() and not refresh:
        print(f"Using saved source data: {csv_path}")
        records_by_isotope = load_cached_records(csv_path)
    else:
        records_by_isotope = {}
        for isotope, (z, mass, index, mts) in ISOTOPES.items():
            records = []
            for mt in mts:
                record = fetch_record(isotope, z, mass, index, mt)
                records.append(record)
            records_by_isotope[isotope] = records

    with gzip.open(csv_path, "wt", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "isotope",
                "display_reaction",
                "source_reaction",
                "mt",
                "energy_eV",
                "cross_section_b",
                "source_url",
            ]
        )
        for records in records_by_isotope.values():
            for record in records:
                energies = [point[0] for point in record["points"]]
                print(
                    f"{record['isotope']} {record['reaction']}: {len(energies)} points "
                    f"({min(energies):.6g} to {max(energies):.6g} eV)"
                )
                for energy_ev, cross_section_b in record["points"]:
                    writer.writerow(
                        [
                            record["isotope"],
                            record["reaction"],
                            record["source_reaction"],
                            record["mt"],
                            f"{energy_ev:.12g}",
                            f"{cross_section_b:.12g}",
                            record["url"],
                        ]
                    )

    plot_rows = []
    for isotope, records in records_by_isotope.items():
        fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
        positive_series_count = 0
        plotted_curves = []
        for record in records:
            positive_points = plot_points_for_record(record)
            if not positive_points:
                print(f"Warning: {isotope} {record['reaction']} has no positive XS values; skipped")
                continue
            energies, cross_sections = zip(*positive_points)
            ax.loglog(energies, cross_sections, linewidth=1.5,
                      label=f"{record['reaction']} (MT {record['mt']})")
            plotted_curves.append((record, positive_points))
            source_points = set(record["points"])
            for energy, cross_section in positive_points:
                plot_rows.append(
                    [
                        isotope,
                        record["reaction"],
                        record["source_reaction"],
                        record["mt"],
                        f"{energy:.12g}",
                        f"{cross_section:.12g}",
                        classify_plot_point(record, energy, cross_section, source_points),
                    ]
                )
            positive_series_count += 1

        if positive_series_count == 0:
            raise ValueError(f"No positive cross sections available to plot for {isotope}")

        total_points = total_cross_section_points(
            [points for _, points in plotted_curves]
        )
        total_energies, total_cross_sections = zip(*total_points)
        ax.loglog(
            total_energies,
            total_cross_sections,
            color="black",
            linewidth=2,
            label="total (sum of plotted channels)",
        )
        plot_rows.extend(
            [
                [
                    isotope,
                    "total (sum of plotted channels)",
                    "derived",
                    "",
                    f"{energy:.12g}",
                    f"{cross_section:.12g}",
                    "derived_total",
                ]
                for energy, cross_section in total_points
            ]
        )

        ax.set_title(f"{isotope} neutron cross sections — {DATASET}")
        ax.set_xlabel("Neutron energy (eV)")
        ax.set_ylabel("Cross section (barns)")
        ax.set_xlim(ENERGY_MIN_EV, ENERGY_MAX_EV)
        ax.set_ylim(bottom=NEAR_ZERO_XS_B)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.savefig(PLOT_DIR / f"{isotope.lower()}_cross_sections.png", dpi=220)
        plt.close(fig)

    plot_data_path = DATA_DIR / "neutron_plot_data.csv.gz"
    with gzip.open(plot_data_path, "wt", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "isotope",
                "display_reaction",
                "source_reaction",
                "mt",
                "energy_eV",
                "cross_section_b",
                "point_kind",
            ]
        )
        writer.writerows(plot_rows)

    print(f"Data saved to: {csv_path}")
    print(f"Plotted coordinates saved to: {plot_data_path}")
    print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download the KAERI source records again instead of using saved raw data",
    )
    main(refresh=parser.parse_args().refresh)
