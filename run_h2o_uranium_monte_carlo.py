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


OUTPUT_ROOT = ROOT / "h2o_uranium_results"
XS_BINARY = ROOT / "neutron_data" / "h2o_uranium_cross_sections.bin"
EXECUTABLE = ROOT / "h2o_uranium_monte_carlo.exe"
ISOTOPES = (1, 16, 235, 238)
WATER_MOLECULE_COUNTS = tuple(range(1, 31))
URANIUM_BRANCHES = {
    "natural": {"u238_to_u235": 138, "label": "natural uranium (U-235:U-238 = 1:138)"},
    "enriched_2pct": {
        "u238_to_u235": 49,
        "label": "2% enriched uranium (U-235:U-238 = 1:49)",
    },
}


def build_run_cases(branches, water_molecule_counts):
    cases = []
    for branch in branches:
        branch_config = URANIUM_BRANCHES[branch]
        for water_molecules in water_molecule_counts:
            case_name = f"{branch}_water_{water_molecules}"
            cases.append(
                {
                    "branch": branch,
                    "uranium_ratio": branch_config["u238_to_u235"],
                    "uranium_label": branch_config["label"],
                    "water_molecules": water_molecules,
                    "name": case_name,
                    "output_dir": OUTPUT_ROOT / case_name,
                }
            )
    return cases


def write_index(rows):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "index.csv").open("w", newline="", encoding="utf-8") as index_file:
        writer = csv.DictWriter(
            index_file,
            fieldnames=(
                "case",
                "uranium_branch",
                "U235_to_U238_ratio",
                "water_molecules_per_uranium_atom",
                "initial_neutrons",
                "final_primary_neutrons",
                "new_generation_size",
                "total_neutron_tracks",
                "end_to_end_seconds",
                "output_directory",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the 60 CUDA H2O/uranium Monte Carlo cases "
            "(100,000 initial neutrons per case by default)."
        )
    )
    parser.add_argument("--initial-neutrons", type=int, default=100_000)
    parser.add_argument("--max-tracks", type=int, default=2_000_000)
    parser.add_argument("--max-history-points", type=int, default=40_000_000)
    parser.add_argument("--max-collisions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument(
        "--water-molecules",
        type=int,
        nargs="+",
        choices=WATER_MOLECULE_COUNTS,
        default=list(WATER_MOLECULE_COUNTS),
        help="water molecules per uranium atom (1 through 30 by default)",
    )
    parser.add_argument(
        "--uranium-branches",
        nargs="+",
        choices=tuple(URANIUM_BRANCHES),
        default=list(URANIUM_BRANCHES),
        help="natural (1:138) and/or 2%% enriched (1:49) uranium",
    )
    parser.add_argument(
        "--skip-history-csv",
        action="store_true",
        help="skip compressed trajectory CSVs while retaining binary histories",
    )
    parser.add_argument(
        "--max-plotted-histories",
        type=int,
        help="deterministically sample at most this many primary tracks per trajectory plot",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="list the requested simulation cases without building or running them",
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
            "a seed in [0, 2^64-1], and a positive plotted-history limit"
        )

    branches = list(dict.fromkeys(args.uranium_branches))
    water_molecule_counts = list(dict.fromkeys(args.water_molecules))
    cases = build_run_cases(branches, water_molecule_counts)
    if args.plan_only:
        print(f"Planned simulations: {len(cases)}")
        for case in cases:
            print(
                f"{case['name']}: {case['uranium_label']}, "
                f"{case['water_molecules']} H2O molecules per uranium atom "
                f"-> {case['output_dir']}"
            )
        return

    if not EXECUTABLE.is_file():
        raise FileNotFoundError(
            f"CUDA executable not found: {EXECUTABLE}. Build it from a "
            "Visual Studio Developer shell; see README.md."
        )

    preparation_start = time.perf_counter()
    table_sizes = prepare_cross_sections(ISOTOPES, XS_BINARY)
    preparation_seconds = time.perf_counter() - preparation_start
    print("Prepared H-1/O-16/U-235/U-238 positive cross-section points:", table_sizes)

    index_rows = []
    for case in cases:
        case_start = time.perf_counter()
        case_output = case["output_dir"]
        case_output.mkdir(parents=True, exist_ok=True)
        command = [
            str(EXECUTABLE),
            str(XS_BINARY),
            str(case_output),
            str(args.initial_neutrons),
            str(args.max_tracks),
            str(args.max_history_points),
            str(args.seed),
            str(args.max_collisions),
            str(case["water_molecules"]),
            str(case["uranium_ratio"]),
        ]
        print(f"\nRunning {case['name']} ...", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

        finals = read_final_set(case_output)
        with (case_output / "new_generation.csv").open(
            newline="", encoding="utf-8"
        ) as generation_file:
            new_generation_count = sum(1 for _ in csv.DictReader(generation_file))
        summary_path = case_output / "summary.txt"
        with summary_path.open("a", encoding="utf-8") as summary:
            summary.write(
                "\nComposition: H2O (H-1:O-16 atom ratio 2:1) plus "
                f"{case['uranium_label']}\n"
                "Water molecules per total uranium atom: "
                f"{case['water_molecules']}\n"
                f"Run seed: {args.seed}\n"
                "All scenarios use the same seed so their primary birth-energy "
                "samples are paired.\n"
            )
        material_name = (
            f"H2O + {case['uranium_label']}; "
            f"{case['water_molecules']} water molecules/U atom"
        )
        timings = convert_and_plot(
            finals,
            new_generation_count,
            output_dir=case_output,
            name_prefix="h2o_uranium",
            material_name=material_name,
            plot_label=(
                "H2O + Nat. U"
                if case["branch"] == "natural"
                else "H2O + 2% U"
            ) + f", ratio {case['water_molecules']}:1",
            export_history_csv=not args.skip_history_csv,
            max_plotted_histories=args.max_plotted_histories,
        )
        elapsed = time.perf_counter() - case_start
        with summary_path.open("a", encoding="utf-8") as summary:
            summary.write(
                "\nRun timings (wall-clock):\n"
                f"Cross-section preparation (seconds): {preparation_seconds:.3f}\n"
                f"History read and primary-history preparation (seconds): "
                f"{timings['history_processing_seconds']:.3f}\n"
                f"Plot generation (seconds): {timings['plot_generation_seconds']:.3f}\n"
                f"Histogram CSV export (seconds): {timings['histogram_csv_seconds']:.3f}\n"
                f"Case end-to-end pipeline (seconds): {elapsed:.3f}\n"
            )
        with summary_path.open(encoding="utf-8") as summary:
            summary_text = summary.read()
        total_tracks = next(
            int(line.split(":", 1)[1])
            for line in summary_text.splitlines()
            if line.startswith("Total neutron tracks including fission offspring:")
        )
        index_rows.append(
            {
                "case": case["name"],
                "uranium_branch": case["branch"],
                "U235_to_U238_ratio": f"1:{case['uranium_ratio']}",
                "water_molecules_per_uranium_atom": case["water_molecules"],
                "initial_neutrons": args.initial_neutrons,
                "final_primary_neutrons": len(finals),
                "new_generation_size": new_generation_count,
                "total_neutron_tracks": total_tracks,
                "end_to_end_seconds": f"{elapsed:.3f}",
                "output_directory": str(case_output),
            }
        )
        write_index(index_rows)
        print(f"Completed {case['name']} in {elapsed:.2f} s")

    print(f"\nCompleted all {len(cases)} simulations.")
    print(f"Cross-section preparation: {preparation_seconds:.3f} s")
    print(f"Case index: {OUTPUT_ROOT / 'index.csv'}")


if __name__ == "__main__":
    main()
