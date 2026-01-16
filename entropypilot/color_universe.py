import matplotlib.pyplot as plt
import numpy as np

# Plotly for interactive 3D visualization
import plotly.graph_objects as go

# This import is necessary for 3D plotting even if not explicitly used
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots

# Color utilities for HSL-based filtering
from entropypilot.utils.color import is_red_or_orange, is_cool_blue_aqua_teal, rgb_to_hex

# ==========================================
# CONFIGURATION
# ==========================================
# The density of the "universe". Higher = denser cloud but slower render.
# 15,000 is a good balance for most machines.
NUM_POINTS = 15000

# ==========================================
# 1. GENERATE THE TOTAL UNIVERSE
# ==========================================
# Generate random RGB triplets between 0.0 and 1.0
print(f"Generating universe of {NUM_POINTS} random colors...")
# Set seed for reproducibility during your demo
np.random.seed(42)
total_universe = np.random.rand(NUM_POINTS, 3)

# ==========================================
# 2. DEFINE THE MATHEMATICAL FILTERS
# Use precise HSL hue-based filtering instead of RGB approximations
# ==========================================

# ---- Filter A: Affirmative ("ONLY cool blues, aquas, teals") ----
# Use precise HSL hue-based filtering (160-260°)
print("Filtering affirmative universe (cool blues/aquas/teals only)...")
aff_mask = np.array([
    is_cool_blue_aqua_teal(rgb_to_hex(r, g, b))
    for r, g, b in total_universe
])
aff_universe = total_universe[aff_mask]

# ---- Filter B: Negative ("NOT red or orange") ----
# Use precise HSL hue-based filtering (exclude 330-45°)
# This should preserve ~85% of the color space:
# - Yellows (50-70°): ✓ Included
# - Greens (85-150°): ✓ Included
# - Cyans (150-200°): ✓ Included
# - Blues (200-260°): ✓ Included
# - Purples (260-300°): ✓ Included
# - Magentas (300-330°): ✓ Included
# - Reds/Oranges/Pinks (330-45°): ✗ Excluded (~15% of hues)
print("Filtering negative universe (excluding red/orange/pink)...")
neg_mask = np.array([
    not is_red_or_orange(rgb_to_hex(r, g, b))
    for r, g, b in total_universe
])
neg_universe = total_universe[neg_mask]


# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================
def plot_universe(ax, data, title, subtitle):
    """Helper function to plot a 3D scatter of colors onto an axis."""
    # R, G, B correspond to X, Y, Z axes
    r = data[:, 0]
    g = data[:, 1]
    b = data[:, 2]

    # Plot the points. Crucially, set the color 'c' to the data itself
    # so the points look like their actual color value.
    ax.scatter(r, g, b, c=data, marker="o", s=5, alpha=0.6)

    # Formatting the 3D Cube
    ax.set_xlabel("Red Amount (X)")
    ax.set_ylabel("Green Amount (Y)")
    ax.set_zlabel("Blue Amount (Z)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    # Set a consistent camera angle for easier comparison
    ax.view_init(elev=20, azim=-60)

    # Titles/Stats
    ax.set_title(title, fontsize=12, fontweight="bold")
    entropy_pct = (len(data) / NUM_POINTS) * 100
    stats_text = f"{subtitle}\nSpace Volume: ~{entropy_pct:.1f}% of total universe ({len(data)} valid options)"
    ax.text2D(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
    )


def plot_universe_interactive(aff_data, neg_data):
    """Create an interactive Plotly visualization of both universes."""
    # Create subplots side by side
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(
            'UNIVERSE A: Affirmative Constraint (Low Entropy)',
            'UNIVERSE B: Negative Constraint (High Entropy)'
        ),
        horizontal_spacing=0.15
    )

    # Convert RGB values to Plotly color strings
    def rgb_array_to_plotly_colors(rgb_array):
        """Convert array of RGB values (0-1) to Plotly RGB color strings."""
        return [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                for r, g, b in rgb_array]

    # --- Plot 1: Affirmative Universe ---
    aff_colors = rgb_array_to_plotly_colors(aff_data)
    entropy_pct_aff = (len(aff_data) / NUM_POINTS) * 100

    fig.add_trace(
        go.Scatter3d(
            x=aff_data[:, 0],
            y=aff_data[:, 1],
            z=aff_data[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=aff_colors,
                opacity=0.6,
                line=dict(width=0)
            ),
            name='Affirmative',
            hovertemplate='<b>RGB Color</b><br>' +
                         'R: %{x:.3f}<br>' +
                         'G: %{y:.3f}<br>' +
                         'B: %{z:.3f}<br>' +
                         '<extra></extra>',
            text=[f'Prompt: "Use ONLY cool blues and teals."<br>' +
                  f'Space Volume: ~{entropy_pct_aff:.1f}% ({len(aff_data)} valid options)'] * len(aff_data)
        ),
        row=1, col=1
    )

    # --- Plot 2: Negative Universe ---
    neg_colors = rgb_array_to_plotly_colors(neg_data)
    entropy_pct_neg = (len(neg_data) / NUM_POINTS) * 100

    fig.add_trace(
        go.Scatter3d(
            x=neg_data[:, 0],
            y=neg_data[:, 1],
            z=neg_data[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=neg_colors,
                opacity=0.6,
                line=dict(width=0)
            ),
            name='Negative',
            hovertemplate='<b>RGB Color</b><br>' +
                         'R: %{x:.3f}<br>' +
                         'G: %{y:.3f}<br>' +
                         'B: %{z:.3f}<br>' +
                         '<extra></extra>',
            text=[f'Prompt: "Do NOT use red or orange."<br>' +
                  f'Space Volume: ~{entropy_pct_neg:.1f}% ({len(neg_data)} valid options)'] * len(neg_data)
        ),
        row=1, col=2
    )

    # Update axes for both subplots
    for i in [1, 2]:
        fig.update_scenes(
            xaxis=dict(title='Red Amount', range=[0, 1], showgrid=True),
            yaxis=dict(title='Green Amount', range=[0, 1], showgrid=True),
            zaxis=dict(title='Blue Amount', range=[0, 1], showgrid=True),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            aspectmode='cube',
            row=1, col=i
        )

    # Update subplot titles to include stats
    fig.layout.annotations[0].update( # type: ignore
        text=(
            f'<b>UNIVERSE A: Affirmative Constraint</b><br>'
            f'<sub>Low Entropy - "Use ONLY cool blues and teals"<br>'
            f'Space Volume: {entropy_pct_aff:.1f}% ({len(aff_data):,} valid options)</sub>'
        ),
        font=dict(size=12)
    )
    fig.layout.annotations[1].update( # type: ignore
        text=(
            f'<b>UNIVERSE B: Negative Constraint</b><br>'
            f'<sub>High Entropy - "Do NOT use red or orange"<br>'
            f'Space Volume: {entropy_pct_neg:.1f}% ({len(neg_data):,} valid options)</sub>'
        ),
        font=dict(size=12)
    )

    # Update layout
    fig.update_layout(
        title_text="Visualizing Probability Spaces: Affirmative vs Negative (Interactive 3D)",
        title_x=0.5,
        title_font_size=16,
        showlegend=False,
        width=1600,
        height=700,
        hovermode='closest',
        margin=dict(t=120, l=50, r=50, b=50)
    )

    return fig


# ==========================================
# MAIN EXECUTION
# ==========================================
def render_matplotlib():
    """Render using matplotlib (static with basic rotation)."""
    print("Rendering 3D Visualizations with Matplotlib...")
    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title( # type: ignore
        "Visualizing Probability Spaces: Affirmative vs Negative"
    )

    # --- Plot 1: Affirmative ---
    ax1 = fig.add_subplot(121, projection="3d")
    plot_universe(
        ax1,
        aff_universe,
        title="UNIVERSE A: Affirmative Constraint\n(Low Entropy)",
        subtitle='Prompt: "Use ONLY cool blues and teals."',
    )

    # --- Plot 2: Negative ---
    ax2 = fig.add_subplot(122, projection="3d")
    plot_universe(
        ax2,
        neg_universe,
        title="UNIVERSE B: Negative Constraint\n(High Entropy)",
        subtitle='Prompt: "Do NOT use red or orange."',
    )

    plt.tight_layout()
    print("Displaying results.")
    plt.show()


def render_interactive():
    """Render using Plotly (fully interactive, spinnable in 3D)."""
    print("Rendering Interactive 3D Visualizations with Plotly...")
    fig = plot_universe_interactive(aff_universe, neg_universe)

    # VSCode Jupyter and JupyterLab compatibility
    # Returns the figure object so notebooks can display it inline
    return fig


# Choose which visualization to use:
# For Jupyter notebooks or interactive exploration, use render_interactive()
# For quick static plots, use render_matplotlib()

if __name__ == "__main__":
    # Default to interactive if running as a script
    # Comment out one of these based on your preference:
    fig = render_interactive()  # Interactive 3D (recommended)
    # render_matplotlib()  # Static matplotlib
    
    fig.show()  # For interactive rendering in scripts
