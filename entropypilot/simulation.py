"""
Simulation to measure constraint violation rates between negative and affirmative prompts.

Runs the color generation multiple times and tracks:
1. Negative constraint violations: How often red/orange appears despite "must NOT contain"
2. Affirmative constraint violations: How often non-blue/aqua/teal appears despite "ONLY contain"
"""

import os
import json
import colorsys
import asyncio
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from openai import AsyncOpenAI
from entropypilot.config import config

os.environ["OPENAI_API_KEY"] = config.openai_api_key

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
BATCH_SIZE = 10  # Number of concurrent requests

client = AsyncOpenAI()


def hex_to_hsl(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color to HSL (Hue, Saturation, Lightness)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (h * 360, s, l)  # Hue in degrees, S and L as 0-1


def is_red_or_orange(hex_color: str) -> bool:
    """
    Detect if a color is a shade of red or orange.
    Red: Hue roughly 0-30 or 330-360
    Orange: Hue roughly 15-45
    Combined: 0-45 or 330-360
    """
    try:
        h, s, l = hex_to_hsl(hex_color)
        # Need some saturation to be considered a color (not gray)
        if s < 0.15:
            return False
        # Red/orange hue ranges
        return (0 <= h <= 45) or (330 <= h <= 360)
    except (ValueError, IndexError):
        return False


def is_cool_blue_aqua_teal(hex_color: str) -> bool:
    """
    Detect if a color is a shade of cool blue, aqua, or teal.
    Cool blues: Hue roughly 180-260
    Aquas/Teals: Hue roughly 160-200
    Combined range: 160-260
    """
    try:
        h, s, l = hex_to_hsl(hex_color)
        # Very low saturation = gray (acceptable as neutral)
        # Very high/low lightness = white/black (acceptable as neutral)
        if s < 0.1 and (l < 0.15 or l > 0.85):
            return True  # Allow near-black, near-white, and grays
        # Must have some saturation to be a "color"
        if s < 0.1:
            return True  # Grays are acceptable
        # Check if in the cool blue/aqua/teal hue range
        return 160 <= h <= 260
    except (ValueError, IndexError):
        return False


async def get_colors_from_llm(prompt: str) -> list[str]:
    """Calls the LLM and returns a list of hex codes."""
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a color palette generator. Output only raw JSON lists of 6 hex codes under the key 'colors'.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        return json.loads(content)["colors"] # type: ignore
    except Exception as e:
        print(f"Error: {e}")
        return []


async def run_simulation(num_runs: int = 100, batch_size: int = BATCH_SIZE):
    """
    Run the simulation and collect statistics using batched async requests.
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

    print(f"Starting simulation with {num_runs} runs (batch size: {batch_size})...")
    print("=" * 60)

    # Process in batches
    for batch_start in range(0, num_runs, batch_size):
        batch_end = min(batch_start + batch_size, num_runs)
        current_batch_size = batch_end - batch_start
        print(f"Processing batch {batch_start + 1}-{batch_end}/{num_runs}...")

        # Create tasks for both negative and affirmative prompts for this batch
        neg_tasks = [get_colors_from_llm(neg_prompt) for _ in range(current_batch_size)]
        aff_tasks = [get_colors_from_llm(aff_prompt) for _ in range(current_batch_size)]

        # Run all tasks concurrently
        all_results = await asyncio.gather(*neg_tasks, *aff_tasks)

        # Split results back into negative and affirmative
        neg_results = all_results[:current_batch_size]
        aff_results = all_results[current_batch_size:]

        # Process negative constraint results
        for i, neg_colors in enumerate(neg_results):
            run_num = batch_start + i + 1
            if neg_colors:
                stats["negative"]["total_runs"] += 1
                stats["negative"]["total_colors"] += len(neg_colors)
                stats["negative"]["all_palettes"].append(neg_colors)
                violation_mask = [is_red_or_orange(c) for c in neg_colors]
                stats["negative"]["all_violations"].append(violation_mask)
                violations = [c for c, v in zip(neg_colors, violation_mask) if v]
                if violations:
                    stats["negative"]["runs_with_violations"] += 1
                    stats["negative"]["violation_colors"] += len(violations)
                    if len(stats["negative"]["violation_examples"]) < 10:
                        stats["negative"]["violation_examples"].append(
                            {"run": run_num, "palette": neg_colors, "violations": violations}
                        )

        # Process affirmative constraint results
        for i, aff_colors in enumerate(aff_results):
            run_num = batch_start + i + 1
            if aff_colors:
                stats["affirmative"]["total_runs"] += 1
                stats["affirmative"]["total_colors"] += len(aff_colors)
                stats["affirmative"]["all_palettes"].append(aff_colors)
                violation_mask = [not is_cool_blue_aqua_teal(c) for c in aff_colors]
                stats["affirmative"]["all_violations"].append(violation_mask)
                violations = [c for c, v in zip(aff_colors, violation_mask) if v]
                if violations:
                    stats["affirmative"]["runs_with_violations"] += 1
                    stats["affirmative"]["violation_colors"] += len(violations)
                    if len(stats["affirmative"]["violation_examples"]) < 10:
                        stats["affirmative"]["violation_examples"].append(
                            {"run": run_num, "palette": aff_colors, "violations": violations}
                        )

    return stats


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color to RGB tuple (0-1 range) for matplotlib."""
    hex_color = hex_color.lstrip("#")
    try:
        r, g, b = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        return (r, g, b)
    except (ValueError, IndexError):
        return (0.5, 0.5, 0.5)  # Gray fallback


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
    plt.savefig("color_palettes.png", dpi=150, bbox_inches="tight")
    print("\nSaved color palettes to: color_palettes.png")
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
    plt.savefig("violation_rates.png", dpi=150, bbox_inches="tight")
    print("Saved violation rates graph to: violation_rates.png")
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

    stats = asyncio.run(run_simulation(num_runs, batch_size=BATCH_SIZE))
    print_results(stats)

    if not args.no_plot:
        plot_color_palettes(stats, max_palettes=args.max_palettes)
        plot_results_graph(stats)
