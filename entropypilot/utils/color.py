"""Color conversion and validation utilities for EntropyPilot.

HSL Hue Degree Reference (0-360):
    Red:           355-10°  (wraps around 0°)
    Red-Orange:    10-20°
    Orange:        20-40°
    Orange-Yellow: 40-50°
    Yellow:        50-70°
    Yellow-Green:  70-85°
    Green:         85-150°
    Cyan/Aqua:     150-200°
    Blue:          200-260°
    Purple/Violet: 260-300°
    Magenta:       300-325°
    Pink/Rose:     325-355°

Saturation & Lightness considerations:
    - Saturation < 0.15: Gray/Desaturated (color hue is less relevant)
    - Lightness < 0.15: Near-black
    - Lightness > 0.85: Near-white

Red/Orange/Pink Detection Range:
    - Combined: 325-360° or 0-45° (~22% of hue spectrum)
    - This catches all reds, oranges, and pinks (including deep pinks at ~327°)
"""

import colorsys


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """
    Convert hex color to RGB tuple (0-1 range) for matplotlib.

    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")

    Returns:
        Tuple of (r, g, b) values in 0-1 range

    Example:
        >>> hex_to_rgb("#FF0000")
        (1.0, 0.0, 0.0)
    """
    hex_color = hex_color.lstrip("#")
    try:
        r, g, b = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        return (r, g, b)
    except (ValueError, IndexError):
        return (0.5, 0.5, 0.5)  # Gray fallback


def hex_to_hsl(hex_color: str) -> tuple[float, float, float]:
    """
    Convert hex color to HSL (Hue, Saturation, Lightness).

    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")

    Returns:
        Tuple of (h, s, l) where:
        - h: Hue in degrees (0-360)
        - s: Saturation (0-1)
        - l: Lightness (0-1)

    Example:
        >>> hex_to_hsl("#FF0000")
        (0.0, 1.0, 0.5)
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (h * 360, s, l)  # Hue in degrees, S and L as 0-1


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """
    Convert RGB values (0-1 range) to hex color string.

    Args:
        r: Red component (0-1)
        g: Green component (0-1)
        b: Blue component (0-1)

    Returns:
        Hex color string with # prefix

    Example:
        >>> rgb_to_hex(1.0, 0.0, 0.0)
        '#ff0000'
    """
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def is_red_or_orange(hex_color: str) -> bool:
    """
    Detect if a color is a shade of red or orange (including pink).

    Uses HSL color space with precise hue ranges:
    - Pink/Rose: 325-355° (tints of red, catches deep pinks at ~327°)
    - Red: 355-10° (wraps around 0°)
    - Red-Orange: 10-20°
    - Orange: 20-45°
    - Combined range: 325-360° or 0-45°

    Explicitly EXCLUDES:
    - Yellow: 50-70° (distinct from orange)
    - Magenta: 300-325° (purple-leaning, not red-leaning)

    Args:
        hex_color: Hex color string to validate

    Returns:
        True if color is red, orange, or pink, False otherwise

    Example:
        >>> is_red_or_orange("#FF0000")  # Pure red
        True
        >>> is_red_or_orange("#FF8800")  # Orange
        True
        >>> is_red_or_orange("#FF69B4")  # Hot pink (330°)
        True
        >>> is_red_or_orange("#FF1493")  # Deep pink (327.6°)
        True
        >>> is_red_or_orange("#FFFF00")  # Yellow
        False
        >>> is_red_or_orange("#FF00FF")  # Magenta (300°)
        False
    """
    try:
        h, s, l = hex_to_hsl(hex_color)

        # Need some saturation to be considered a color (not gray)
        if s < 0.15:
            return False

        # Red/orange/pink hue ranges: 325-360° or 0-45°
        # Lower bound at 325° catches deep pinks (327°) without over-blocking magentas
        # Upper bound at 45° excludes yellows (which start around 50°)
        return h >= 325 or h <= 45
    except (ValueError, IndexError):
        return False


def is_cool_blue_aqua_teal(hex_color: str) -> bool:
    """
    Detect if a color is a shade of cool blue, aqua, or teal.

    Uses HSL color space to check hue ranges:
    - Cool blues: Hue roughly 180-260
    - Aquas/Teals: Hue roughly 160-200
    - Combined range: 160-260

    Args:
        hex_color: Hex color string to validate

    Returns:
        True if color is cool blue/aqua/teal, False otherwise

    Example:
        >>> is_cool_blue_aqua_teal("#00FFFF")
        True
        >>> is_cool_blue_aqua_teal("#FF0000")
        False
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
