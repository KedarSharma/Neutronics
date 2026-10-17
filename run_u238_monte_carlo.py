import argparse
import bisect
import csv
import gzip
import struct
import subprocess
import time
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "neutron_data" / "neutron_cross_sections.csv.gz"
XS_BINARY = ROOT / "neutron_data" / "u238_cross_sections.bin"
OUTPUT_DIR = ROOT / "simulation_results"
EXECUTABLE = ROOT / "u238_monte_carlo.exe"
TABLE_ORDER = (2, 4, 102, 18)
CHANNEL_NAMES = {2: "elastic", 4: "inelastic", 102: "capture", 18: "fission"}
HISTORY_POINT = struct.Struct("<IIf")
XS_POINT = struct.Struct("<dd")


def prepare_cross_sections(isotopes=(238,), binary_path=XS_BINARY):
    if not RAW_DATA.is_file():
        raise FileNotFoundError(
            f"Raw KAERI dataset not found: {RAW_DATA}. "
            "Run plot_neutron_cross_sections.py first."
        )
    tables = {(mass_number, mt): [] for mass_number in isotopes for mt in TABLE_ORDER}
    with gzip.open(RAW_DATA, "rt", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            isotope = row["isotope"]
            mass_number = int(isotope.split("-")[1])
            if mass_number not in isotopes:
                continue
            mt = int(row["mt"])
            if mass_number == 2 and mt == 3:
                mt = 4
            key = (mass_number, mt)
            if key not in tables:
                continue
            energy = float(row["energy_eV"])
            cross_section = float(row["cross_section_b"])
            if cross_section > 0:
                tables[key].append((energy, cross_section))

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    required_mts = {
        1: {2, 102},
        2: {2, 4, 102},
        16: {2, 4, 102},
        235: set(TABLE_ORDER),
        238: set(TABLE_ORDER),
    }
    with binary_path.open("wb") as output:
        for (mass_number, mt), points in tables.items():
            points.sort()
            if len(points) < 2 and (
                mt in required_mts.get(mass_number, set()) or points
            ):
                raise ValueError(
                    f"Insufficient positive isotope-{mass_number} MT {mt} source points"
                )
            output.write(struct.Struct("<III").pack(mass_number, mt, len(points)))
            for energy, cross_section in points:
                output.write(XS_POINT.pack(energy, cross_section))
    return {key: len(points) for key, points in tables.items()}


def read_final_set(output_dir=OUTPUT_DIR):
    finals = {}
    with (output_dir / "final_neutrons.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        for row in csv.DictReader(source):
            finals[int(row["track_id"])] = {
                "collision_count": int(row["collision_count"]),
                "energy_ev": float(row["energy_ev"]),
                "terminal_event": row["terminal_event"],
            }
    return finals


def convert_and_plot(
    finals,
    new_generation_count,
    output_dir=OUTPUT_DIR,
    name_prefix="u238",
    material_name="U-238",
    plot_label=None,
    export_history_csv=True,
    max_plotted_histories=None,
):
    timings = {}
    history_path = output_dir / "history_points.bin"
    history_csv_path = output_dir / f"{name_prefix}_history_points.csv.gz"
    final_histories = defaultdict(list)
    if max_plotted_histories is None or max_plotted_histories >= len(finals):
        plotted_track_ids = set(finals)
    elif max_plotted_histories < 1:
        raise ValueError("max_plotted_histories must be positive")
    else:
        final_track_ids = sorted(finals)
        stride = (len(final_track_ids) + max_plotted_histories - 1) // max_plotted_histories
        plotted_track_ids = set(final_track_ids[::stride][:max_plotted_histories])
    plotted_track_ids_array = np.fromiter(
        plotted_track_ids, dtype=np.uint32, count=len(plotted_track_ids)
    )

    history_processing_start = time.perf_counter()
    if export_history_csv:
        csv_output = gzip.open(history_csv_path, "wt", newline="", encoding="utf-8")
    else:
        csv_output = nullcontext(None)
    with history_path.open("rb") as binary, csv_output as compressed:
        writer = csv.writer(compressed) if compressed is not None else None
        if writer is not None:
            writer.writerow(
                ["track_id", "collision_count", "energy_ev", "is_final_primary"]
            )
        if writer is None:
            history_dtype = np.dtype(
                [("track_id", "<u4"), ("collision_count", "<u4"), ("energy_ev", "<f4")]
            )
            chunk_size = 1_000_000
            while True:
                chunk = np.fromfile(binary, dtype=history_dtype, count=chunk_size)
                if chunk.size == 0:
                    break
                selected = chunk[
                    np.isin(chunk["track_id"], plotted_track_ids_array)
                ]
                for row in selected:
                    track_id = int(row["track_id"])
                    final_histories[track_id].append(
                        (int(row["collision_count"]), float(row["energy_ev"]))
                    )
        else:
            while True:
                record = binary.read(HISTORY_POINT.size)
                if not record:
                    break
                if len(record) != HISTORY_POINT.size:
                    raise RuntimeError("History binary ended with an incomplete record")
                track_id, collision_count, energy = HISTORY_POINT.unpack(record)
                writer.writerow(
                    [
                        track_id,
                        collision_count,
                        f"{energy:.9g}",
                        int(track_id in finals),
                    ]
                )
                if track_id in plotted_track_ids:
                    final_histories[track_id].append((collision_count, energy))
    timings["history_processing_seconds"] = time.perf_counter() - history_processing_start

    if len(final_histories) != len(plotted_track_ids):
        missing = sorted(plotted_track_ids - set(final_histories))
        raise RuntimeError(
            f"History output missing {len(missing)} selected final primary track(s)"
        )

    plot_start = time.perf_counter()
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    segments = []
    for points in final_histories.values():
        points.sort()
        if len(points) > 1:
            segments.append(points)
    line_collection = LineCollection(
        segments,
        colors="#2878b5",
        alpha=0.18,
        linewidths=0.65,
    )
    ax.add_collection(line_collection)
    ax.autoscale_view()
    ax.set_yscale("log")
    ax.set_xlabel("Collision count")
    ax.set_ylabel("Neutron energy (eV)")
    title_label = plot_label or material_name
    ax.set_title(f"{title_label}: Neutron Histories")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(output_dir / f"{name_prefix}_final_neutron_histories.png", dpi=200)
    plt.close(fig)

    bin_edges = [10.0**exponent for exponent in range(-5, 9)]
    final_energies = [item["energy_ev"] for item in finals.values()]
    counts = [0] * (len(bin_edges) - 1)
    for energy in final_energies:
        if energy < bin_edges[0] or energy > bin_edges[-1]:
            continue
        if energy == bin_edges[-1]:
            counts[-1] += 1
            continue
        bin_index = bisect.bisect_right(bin_edges, energy) - 1
        bin_index = max(0, min(len(counts) - 1, bin_index))
        counts[bin_index] += 1

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.stairs(counts, bin_edges, fill=True, alpha=0.65)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Final energy before capture/fission (eV; decade bins)")
    ax.set_ylabel("Primary neutron count")
    ax.set_title(f"{title_label}: Final Energy")
    ax.set_xticks(bin_edges)
    ax.set_xticklabels(
        [f"10^{exponent}" for exponent in range(-5, 9)], rotation=45, ha="right"
    )
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(output_dir / f"{name_prefix}_final_energy_histogram.png", dpi=200)
    plt.close(fig)
    timings["plot_generation_seconds"] = time.perf_counter() - plot_start

    histogram_path = output_dir / f"{name_prefix}_final_energy_histogram.csv"
    histogram_csv_start = time.perf_counter()
    with histogram_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["energy_low_ev", "energy_high_ev", "primary_count"])
        for index, count in enumerate(counts):
            writer.writerow([bin_edges[index], bin_edges[index + 1], count])
    timings["histogram_csv_seconds"] = time.perf_counter() - histogram_csv_start

    summary_path = output_dir / "summary.txt"
    history_csv_status = (
        f"Full trajectory CSV: {name_prefix}_history_points.csv.gz\n"
        if export_history_csv
        else "Full trajectory CSV: skipped for this run "
             "(binary history_points.bin retained; any existing CSV was not updated)\n"
    )
    with summary_path.open("a", encoding="utf-8") as summary:
        summary.write(
            f"\nFinal energy histogram: {name_prefix}_final_energy_histogram.csv\n"
            f"{history_csv_status}"
            f"Primary-history plot: {name_prefix}_final_neutron_histories.png "
            f"({len(final_histories)} sampled primary histories)\n"
            f"Final-energy plot: {name_prefix}_final_energy_histogram.png\n"
        )
    return timings


def main():
    overall_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Run the 10,000-neutron pure U-238 CUDA Monte Carlo simulation."
    )
    parser.add_argument("--initial-neutrons", type=int, default=10_000)
    parser.add_argument("--max-tracks", type=int, default=1_000_000)
    parser.add_argument("--max-history-points", type=int, default=20_000_000)
    parser.add_argument("--max-collisions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument(
        "--skip-history-csv",
        action="store_true",
        help="skip writing the large compressed trajectory CSV; keep binary histories",
    )
    parser.add_argument(
        "--max-plotted-histories",
        type=int,
        help="deterministically sample at most this many primary histories for the plot",
    )
    args = parser.parse_args()
    if (
        args.initial_neutrons < 1
        or args.max_tracks < args.initial_neutrons
        or args.max_tracks > 0xFFFFFFFF
        or args.max_history_points < args.initial_neutrons
        or args.max_collisions < 1
        or not 0 <= args.seed <= 0xFFFFFFFFFFFFFFFF
        or (
            args.max_plotted_histories is not None
            and args.max_plotted_histories < 1
        )
    ):
        parser.error(
            "require 1 <= initial-neutrons <= max-tracks <= 2^32-1, "
            "max-history-points >= initial-neutrons, max-collisions >= 1, "
            "and a seed in [0, 2^64-1]"
        )

    preparation_start = time.perf_counter()
    table_sizes = prepare_cross_sections()
    preparation_seconds = time.perf_counter() - preparation_start
    print("Prepared positive uranium cross-section source points:", table_sizes)
    if not EXECUTABLE.is_file():
        raise FileNotFoundError(
            f"CUDA executable not found: {EXECUTABLE}. Build it from a "
            "Visual Studio Developer shell; see README.md."
        )
    OUTPUT_DIR.mkdir(exist_ok=True)
    command = [
        str(EXECUTABLE),
        str(XS_BINARY),
        str(OUTPUT_DIR),
        str(args.initial_neutrons),
        str(args.max_tracks),
        str(args.max_history_points),
        str(args.seed),
        str(args.max_collisions),
    ]
    native_start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    native_seconds = time.perf_counter() - native_start
    output_read_start = time.perf_counter()
    finals = read_final_set()
    with (OUTPUT_DIR / "summary.txt").open("a", encoding="utf-8") as summary:
        summary.write(f"Run seed: {args.seed}\n")
        summary.write(
            "k-infinity note: this run reports the requested new-generation "
            "set size; it is not a multiplication-factor estimate.\n"
        )
    with (OUTPUT_DIR / "new_generation.csv").open(
        newline="", encoding="utf-8"
    ) as generation_file:
        new_generation_count = sum(1 for _ in csv.DictReader(generation_file))
    output_read_seconds = time.perf_counter() - output_read_start
    postprocess_timings = convert_and_plot(
        finals,
        new_generation_count,
        export_history_csv=not args.skip_history_csv,
        max_plotted_histories=args.max_plotted_histories,
    )
    total_seconds = time.perf_counter() - overall_start
    timings = {
        "Cross-section preparation (seconds)": preparation_seconds,
        "Native CUDA executable (seconds)": native_seconds,
        "Read simulation summaries (seconds)": output_read_seconds,
        "History read and primary-history preparation (seconds)":
            postprocess_timings["history_processing_seconds"],
        "Plot generation (seconds)": postprocess_timings["plot_generation_seconds"],
        "Histogram CSV export (seconds)": postprocess_timings["histogram_csv_seconds"],
        "End-to-end pipeline (seconds)": total_seconds,
    }
    timing_lines = ["\nRun timings (wall-clock):"]
    timing_lines.extend(f"{label}: {duration:.3f}" for label, duration in timings.items())
    with (OUTPUT_DIR / "summary.txt").open("a", encoding="utf-8") as summary:
        summary.write("\n".join(timing_lines) + "\n")
    print("\n".join(timing_lines))
    print(f"Primary final set size: {len(finals)}")
    print(f"Requested k-infinity set size: see {OUTPUT_DIR / 'new_generation.csv'}")
    print(f"Plots and trajectories saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
