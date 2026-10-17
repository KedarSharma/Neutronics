import argparse
import csv
import subprocess
import time
from pathlib import Path

from run_u238_monte_carlo import (
    ROOT,
    convert_and_plot,
    prepare_cross_sections,
    read_final_set,
)


OUTPUT_DIR = ROOT / "natural_uranium_results"
XS_BINARY = ROOT / "neutron_data" / "natural_uranium_cross_sections.bin"
EXECUTABLE = ROOT / "natural_uranium_monte_carlo.exe"


def main():
    overall_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Run a 100,000-neutron natural-uranium CUDA Monte Carlo simulation."
    )
    parser.add_argument("--initial-neutrons", type=int, default=100_000)
    parser.add_argument("--max-tracks", type=int, default=2_000_000)
    parser.add_argument("--max-history-points", type=int, default=40_000_000)
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
    if not EXECUTABLE.is_file():
        raise FileNotFoundError(
            f"CUDA executable not found: {EXECUTABLE}. Build it from a "
            "Visual Studio Developer shell; see README.md."
        )

    preparation_start = time.perf_counter()
    table_sizes = prepare_cross_sections((235, 238), XS_BINARY)
    preparation_seconds = time.perf_counter() - preparation_start
    print("Prepared natural-uranium positive cross-section points:", table_sizes)

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
    finals = read_final_set(OUTPUT_DIR)
    with (OUTPUT_DIR / "new_generation.csv").open(
        newline="", encoding="utf-8"
    ) as generation_file:
        new_generation_count = sum(1 for _ in csv.DictReader(generation_file))
    with (OUTPUT_DIR / "summary.txt").open("a", encoding="utf-8") as summary:
        summary.write(
            "\nComposition: natural uranium, U-235:U-238 atom ratio 1:138\n"
            f"Run seed: {args.seed}\n"
            "k-infinity note: this run reports the requested new-generation "
            "set size; it is not a physics multiplication-factor estimate.\n"
        )
    output_read_seconds = time.perf_counter() - output_read_start

    postprocess = convert_and_plot(
        finals,
        new_generation_count,
        output_dir=OUTPUT_DIR,
        name_prefix="natural_uranium",
        material_name="Natural uranium (U-235:U-238 = 1:138)",
        plot_label="Natural U",
        export_history_csv=not args.skip_history_csv,
        max_plotted_histories=args.max_plotted_histories,
    )
    total_seconds = time.perf_counter() - overall_start
    timings = {
        "Cross-section preparation (seconds)": preparation_seconds,
        "Native CUDA executable (seconds)": native_seconds,
        "Read simulation summaries (seconds)": output_read_seconds,
        "History read and primary-history preparation (seconds)":
            postprocess["history_processing_seconds"],
        "Plot generation (seconds)": postprocess["plot_generation_seconds"],
        "Histogram CSV export (seconds)": postprocess["histogram_csv_seconds"],
        "End-to-end pipeline (seconds)": total_seconds,
    }
    timing_lines = ["\nRun timings (wall-clock):"]
    timing_lines.extend(f"{label}: {duration:.3f}" for label, duration in timings.items())
    with (OUTPUT_DIR / "summary.txt").open("a", encoding="utf-8") as summary:
        summary.write("\n".join(timing_lines) + "\n")

    print(f"Primary final set size: {len(finals)}")
    print(f"New-generation set size: {new_generation_count}")
    print("\n".join(timing_lines))
    print(f"Plots and trajectories saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
