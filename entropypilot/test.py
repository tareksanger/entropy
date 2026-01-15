import json
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from openai import OpenAI

from entropypilot.config import config

os.environ["OPENAI_API_KEY"] = config.openai_api_key

# 1. SETUP CLIENT
# Using a slightly older/smaller model (like gpt-3.5 or gpt-4-turbo)
# often highlights these architectural issues better than the newest flagship.
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.9

client = OpenAI()


def get_colors_from_llm(prompt):
    """
    Calls the LLM and demands a raw JSON list of hex codes.
    """
    print(f"Asking LLM: '{prompt}'...")
    try:
        response = client.chat.completions.create(
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
        return json.loads(content)["colors"]
    except Exception as e:
        print(f"Error generating colors: {e}")
        # Return a fallback gray palette if LLM fails entirely
        return ["#cccccc"] * 6


def draw_palette_on_axis(ax, colors, title, subtitle):
    """
    Draws color swatches onto a specific matplotlib Axis (ax).
    """
    # Main Title
    ax.text(0, 1.3, title, fontsize=12, fontweight="bold", transform=ax.transAxes)
    # Subtitle explaining the mechanism
    ax.text(
        0,
        1.1,
        subtitle,
        fontsize=10,
        fontstyle="italic",
        color="#555555",
        transform=ax.transAxes,
    )

    # Draw the swatches
    for i, color in enumerate(colors):
        # Draw the colored rectangle
        # Coordinates are (x, y), width, height
        rect = patches.Rectangle(
            (i, 0), 1, 1, linewidth=1, edgecolor="#e0e0e0", facecolor=color
        )
        ax.add_patch(rect)

        # Add the hex code text below
        # Use try/except in case the LLM generates invalid hex colors that break matplotlib
        try:
            ax.text(
                i + 0.5,
                -0.2,
                color,
                ha="center",
                va="center",
                fontsize=9,
                family="monospace",
            )
        except:
            ax.text(
                i + 0.5,
                -0.2,
                "INVALID",
                ha="center",
                va="center",
                fontsize=8,
                color="red",
            )

    # Clean up the axis view
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis("off")  # Hide X/Y axes and ticks


# =========================================
# MAIN EXECUTION
# =========================================
if __name__ == "__main__":
    # Define the two opposing prompts

    # TEST A: High Entropy (Negative Constraint)
    # The model has to consider every color in existence and try to filter out red.
    neg_prompt = "Generate a palette of 6 distinct hex codes. CONSTRAINT: The palette must NOT contain any shade of red or orange."

    # TEST B: Low Entropy (Affirmative Constraint)
    # The model only has to look in one specific corner of color space.
    aff_prompt = "Generate a palette of 6 distinct hex codes. CONSTRAINT: The palette must ONLY contain shades of cool blues, aquas, and teals."

    # Fetch data
    print("--- Fetching palettes from LLM (this might take a few seconds) ---")
    neg_colors = get_colors_from_llm(neg_prompt)
    aff_colors = get_colors_from_llm(aff_prompt)
    print("--- Data fetched. Rendering GUI. ---")

    # Setup the GUI Window (Figure and Axes)
    # nrows=2, ncols=1 means stacked vertically
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 7))
    fig.canvas.manager.set_window_title("Architecture vs. Prompting Demo") # type: ignore

    # 1. Draw Negative Results on Top Axis (ax1)
    draw_palette_on_axis(
        ax=ax1,
        colors=neg_colors,
        title="TEST A: Negative Constraint (High Entropy)",
        subtitle=f'Prompt: "{neg_prompt}"\nResult: Often muddy or destructed. The model spends its energy trying to *exclude* data.',
    )

    # 2. Draw Affirmative Results on Bottom Axis (ax2)
    draw_palette_on_axis(
        ax=ax2,
        colors=aff_colors,
        title="TEST B: Affirmative Constraint (Low Entropy)",
        subtitle=f'Prompt: "{aff_prompt}"\nResult: Vibrant and cohesive. The model\'s search space was architecturally narrowed.',
    )

    # Adjust layout to prevent overlaps and show the window
    plt.subplots_adjust(hspace=0.6, top=0.9, bottom=0.1)
    print("Displaying results...")
    plt.show()
