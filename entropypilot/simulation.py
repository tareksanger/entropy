"""
Simulation to measure constraint violation rates between negative and affirmative prompts.

Runs the color generation multiple times and tracks:
1. Negative constraint violations: How often red/orange appears despite "must NOT contain"
2. Affirmative constraint violations: How often non-blue/aqua/teal appears despite "ONLY contain"
"""

import random
from concurrent.futures import ThreadPoolExecutor

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from entropypilot.utils import (
    Colors,
    get_colors_from_llm_sync,
    hex_to_rgb,
    is_cool_blue_aqua_teal,
    is_red_or_orange,
)

BATCH_SIZE = 10  # Number of concurrent requests


def get_colors_from_llm(prompt: str, seed: int | None = None) -> Colors:
    """Sync wrapper for LLM calls - returns Colors model."""
    return get_colors_from_llm_sync(prompt, model="gpt-4o-mini", temperature=0, seed=seed)


def process_constraint_batch(
    prompt: str,
    constraint_type: str,
    violation_check,
    num_runs: int,
    batch_size: int,
    stats: dict,
    use_unique_seeds: bool = True,
):
    """
    Process a constraint type in batches using ThreadPoolExecutor.

    Args:
        prompt: The LLM prompt to use
        constraint_type: Either "negative" or "affirmative"
        violation_check: Function that returns True if a color violates the constraint
        num_runs: Total number of runs
        batch_size: Number of concurrent requests per batch
        stats: Statistics dictionary to update
        use_unique_seeds: If True, uses unique seeds to bypass OpenAI caching (default: True)
    """
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for batch_start in range(0, num_runs, batch_size):
            batch_end = min(batch_start + batch_size, num_runs)
            current_batch_size = batch_end - batch_start
            print(f"  Batch {batch_start + 1}-{batch_end}/{num_runs}...")

            # Submit all tasks for this batch with unique seeds to bypass caching
            futures = [
                executor.submit(
                    get_colors_from_llm,
                    prompt,
                    seed=random.randint(0, 1_000_000) if use_unique_seeds else None
                )
                for _ in range(current_batch_size)
            ]

            # Collect results in order (this maintains batch ordering)
            results = [future.result() for future in futures]

            # Process results in order
            for i, colors in enumerate(results):
                run_num = batch_start + i + 1
                if colors:
                    stats[constraint_type]["total_runs"] += 1
                    stats[constraint_type]["total_colors"] += len(colors.palette)
                    stats[constraint_type]["all_palettes"].append(colors.palette)
                    violation_mask = [violation_check(c) for c in colors.palette]
                    stats[constraint_type]["all_violations"].append(violation_mask)
                    violations = [c for c, v in zip(colors.palette, violation_mask) if v]
                    if violations:
                        stats[constraint_type]["runs_with_violations"] += 1
                        stats[constraint_type]["violation_colors"] += len(violations)
                        if len(stats[constraint_type]["violation_examples"]) < 10:
                            stats[constraint_type]["violation_examples"].append(
                                {"run": run_num, "palette": colors.palette, "violations": violations}
                            )


def run_simulation(num_runs: int = 100, batch_size: int = BATCH_SIZE):
    """
    Run the simulation and collect statistics using batched concurrent requests.
    """
    neg_prompt = "Generate a palette of 6 distinct hex codes. CONSTRAINT: The palette must NOT contain any shade of red or orange."
    aff_prompt = "Generate a palette of 6 distinct hex codes. CONSTRAINT: The palette must ONLY contain shades of cool blues, aquas, and teals."

    # Statistics tracking
    stats = {
        "negative": {
            "total_runs": 0,
            "runs_with_violations": 0,
            "total_colors": 0,
            "violation_colors": 0,
            "violation_examples": [],  # Store some examples of violations
            "all_palettes": [],  # Store all generated palettes
            "all_violations": [],  # Store violation status per color
        },
        "affirmative": {
            "total_runs": 0,
            "runs_with_violations": 0,
            "total_colors": 0,
            "violation_colors": 0,
            "violation_examples": [],
            "all_palettes": [],
            "all_violations": [],
        },
    }

    # Process negative constraints
    print("Processing NEGATIVE constraints (NOT red/orange)...")
    process_constraint_batch(
        prompt=neg_prompt,
        constraint_type="negative",
        violation_check=is_red_or_orange,
        num_runs=num_runs,
        batch_size=batch_size,
        stats=stats,
    )

    # Process affirmative constraints
    print("\nProcessing AFFIRMATIVE constraints (ONLY blues/aquas/teals)...")
    process_constraint_batch(
        prompt=aff_prompt,
        constraint_type="affirmative",
        violation_check=lambda c: not is_cool_blue_aqua_teal(c),
        num_runs=num_runs,
        batch_size=batch_size,
        stats=stats,
    )

    return stats


def plot_color_palettes(stats: dict, max_palettes: int = 20):
    """Display all generated color palettes as visual swatches."""
    fig, axes = plt.subplots(1, 2, figsize=(14, max(8, max_palettes * 0.4)))

    for ax, (constraint_type, label) in zip(
        axes, [("negative", "Negative Constraint\n(must NOT contain red/orange)"),
               ("affirmative", "Affirmative Constraint\n(ONLY blues/aquas/teals)")]
    ):
        palettes = stats[constraint_type]["all_palettes"][:max_palettes]
        violations = stats[constraint_type]["all_violations"][:max_palettes]

        if not palettes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
            ax.set_title(label)
            ax.axis("off")
            continue

        num_palettes = len(palettes)
        num_colors = len(palettes[0]) if palettes else 6

        for row, (palette, violation_mask) in enumerate(zip(palettes, violations)):
            for col, (color, is_violation) in enumerate(zip(palette, violation_mask)):
                rect = mpatches.FancyBboxPatch(
                    (col, num_palettes - row - 1), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=hex_to_rgb(color),
                    edgecolor="red" if is_violation else "none",
                    linewidth=3 if is_violation else 0
                )
                ax.add_patch(rect)
                # Add hex label
                rgb = hex_to_rgb(color)
                text_color = "white" if (rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114) < 0.5 else "black"
                ax.text(col + 0.45, num_palettes - row - 0.55, color,
                       ha="center", va="center", fontsize=6, color=text_color,
                       fontweight="bold" if is_violation else "normal")

        ax.set_xlim(-0.1, num_colors + 0.1)
        ax.set_ylim(-0.1, num_palettes + 0.1)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Run #")
        ax.set_yticks(np.arange(num_palettes) + 0.45)
        ax.set_yticklabels([str(i + 1) for i in range(num_palettes - 1, -1, -1)])
        ax.set_xticks([])
        ax.set_aspect("equal")

    plt.suptitle("Generated Color Palettes (red border = violation)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_results_graph(stats: dict):
    """Create bar charts comparing violation rates."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate rates
    neg = stats["negative"]
    aff = stats["affirmative"]

    neg_run_rate = (neg["runs_with_violations"] / neg["total_runs"] * 100) if neg["total_runs"] > 0 else 0
    aff_run_rate = (aff["runs_with_violations"] / aff["total_runs"] * 100) if aff["total_runs"] > 0 else 0
    neg_color_rate = (neg["violation_colors"] / neg["total_colors"] * 100) if neg["total_colors"] > 0 else 0
    aff_color_rate = (aff["violation_colors"] / aff["total_colors"] * 100) if aff["total_colors"] > 0 else 0

    # Plot 1: Run violation rate
    ax1 = axes[0]
    bars1 = ax1.bar(["Negative\n(NOT red/orange)", "Affirmative\n(ONLY blue/aqua/teal)"],
                    [neg_run_rate, aff_run_rate],
                    color=["#e74c3c", "#3498db"], edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("Violation Rate (%)", fontsize=11)
    ax1.set_title("Runs with at Least One Violation", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(neg_run_rate, aff_run_rate, 10) * 1.2)
    for bar, rate in zip(bars1, [neg_run_rate, aff_run_rate]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Plot 2: Individual color violation rate
    ax2 = axes[1]
    bars2 = ax2.bar(["Negative\n(NOT red/orange)", "Affirmative\n(ONLY blue/aqua/teal)"],
                    [neg_color_rate, aff_color_rate],
                    color=["#e74c3c", "#3498db"], edgecolor="black", linewidth=1.5)
    ax2.set_ylabel("Violation Rate (%)", fontsize=11)
    ax2.set_title("Individual Colors that Violated Constraint", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(neg_color_rate, aff_color_rate, 5) * 1.2)
    for bar, rate in zip(bars2, [neg_color_rate, aff_color_rate]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{rate:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add summary stats as text
    fig.text(0.5, 0.02,
             f"Negative: {neg['runs_with_violations']}/{neg['total_runs']} runs, {neg['violation_colors']}/{neg['total_colors']} colors  |  "
             f"Affirmative: {aff['runs_with_violations']}/{aff['total_runs']} runs, {aff['violation_colors']}/{aff['total_colors']} colors",
             ha="center", fontsize=10, style="italic")

    plt.suptitle("Constraint Violation Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # type: ignore
    plt.show()


def print_results(stats: dict):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    # Negative constraint results
    neg = stats["negative"]
    print("\n📛 NEGATIVE CONSTRAINT (must NOT contain red/orange)")
    print("-" * 50)
    print(f"  Total runs: {neg['total_runs']}")
    print(f"  Runs with violations: {neg['runs_with_violations']}")
    if neg["total_runs"] > 0:
        violation_rate = (neg["runs_with_violations"] / neg["total_runs"]) * 100
        print(f"  Violation rate: {violation_rate:.1f}%")
    print(f"  Total colors generated: {neg['total_colors']}")
    print(f"  Violation colors (red/orange): {neg['violation_colors']}")
    if neg["total_colors"] > 0:
        color_violation_rate = (neg["violation_colors"] / neg["total_colors"]) * 100
        print(f"  Color violation rate: {color_violation_rate:.2f}%")

    if neg["violation_examples"]:
        print("\n  Example violations:")
        for ex in neg["violation_examples"][:5]:
            print(f"    Run {ex['run']}: {ex['violations']} in palette {ex['palette']}")

    # Affirmative constraint results
    aff = stats["affirmative"]
    print("\n✅ AFFIRMATIVE CONSTRAINT (ONLY cool blues/aquas/teals)")
    print("-" * 50)
    print(f"  Total runs: {aff['total_runs']}")
    print(f"  Runs with violations: {aff['runs_with_violations']}")
    if aff["total_runs"] > 0:
        violation_rate = (aff["runs_with_violations"] / aff["total_runs"]) * 100
        print(f"  Violation rate: {violation_rate:.1f}%")
    print(f"  Total colors generated: {aff['total_colors']}")
    print(f"  Violation colors (non-blue/aqua/teal): {aff['violation_colors']}")
    if aff["total_colors"] > 0:
        color_violation_rate = (aff["violation_colors"] / aff["total_colors"]) * 100
        print(f"  Color violation rate: {color_violation_rate:.2f}%")

    if aff["violation_examples"]:
        print("\n  Example violations:")
        for ex in aff["violation_examples"][:5]:
            print(f"    Run {ex['run']}: {ex['violations']} in palette {ex['palette']}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    if neg["total_runs"] > 0 and aff["total_runs"] > 0:
        neg_rate = (neg["runs_with_violations"] / neg["total_runs"]) * 100
        aff_rate = (aff["runs_with_violations"] / aff["total_runs"]) * 100
        print(f"  Negative constraint violation rate: {neg_rate:.1f}%")
        print(f"  Affirmative constraint violation rate: {aff_rate:.1f}%")
        if neg_rate > aff_rate:
            print(
                f"\n  → Negative constraints failed {neg_rate/aff_rate:.1f}x more often!"
                if aff_rate > 0
                else f"\n  → Negative constraints failed while affirmative had 0 violations!"
            )
        elif aff_rate > neg_rate:
            print(f"\n  → Affirmative constraints failed more often (unexpected!)")
        else:
            print(f"\n  → Both constraints had similar violation rates")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run color constraint violation simulation"
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=100,
        help="Number of simulation runs (default: 100, max recommended: 400)",
    )
    parser.add_argument(
        "--max-palettes",
        type=int,
        default=20,
        help="Maximum number of palettes to display in visualization (default: 20)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots (text results only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of concurrent API requests per batch (default: 10)",
    )
    args = parser.parse_args()

    num_runs = min(max(args.num_runs, 1), 400)  # Clamp between 1-400
    BATCH_SIZE = max(1, args.batch_size)  # Override global batch size

    stats = run_simulation(num_runs, batch_size=BATCH_SIZE)
    print_results(stats)

    if not args.no_plot:
        plot_color_palettes(stats, max_palettes=args.max_palettes)
        plot_results_graph(stats)
