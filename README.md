# EntropyPilot

An experimental project exploring how entropy control affects LLM reliability through concrete demonstrations. This repository investigates how reducing the "universe" of possible outputs—whether through prompt design, data structure, or architectural choices—improves the consistency and predictability of LLM behavior.

## Overview

EntropyPilot demonstrates a fundamental principle: LLM reliability improves when we shrink the probability space the model operates in. This isn't just about clever prompting—it's about systematically eliminating ambiguity at every level:

- **Linguistic**: How we phrase instructions and constraints
- **Structural**: How we design data schemas and representations
- **Architectural**: How we compose systems and interfaces

The current implementation focuses on one concrete example: affirmative vs. negative constraints in color palette generation. This serves as a visual, measurable demonstration of how different approaches to the same constraint affect entropy and, consequently, reliability. The project is ongoing and will expand to include additional demonstrations of entropy control across different domains.

## Requirements

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

## Installation

### 1. Install uv

uv is a fast Python package installer and resolver. Choose your installation method:

**macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**With pip:**
```bash
pip install uv
```

**With Homebrew (macOS):**
```bash
brew install uv
```

For more installation options, see the [official uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Clone the Repository

```bash
git clone <repository-url>
cd EntropyPilot
```

### 3. Install Dependencies

uv will automatically create a virtual environment and install all dependencies:

```bash
uv sync
```

This command:
- Creates a `.venv` directory with a Python 3.12+ virtual environment
- Installs all dependencies from `pyproject.toml`
- Locks the dependency versions in `uv.lock`

### 4. Configure Environment Variables

Create a `.env` file in the `entropypilot` directory:

```bash
cp entropypilot/.env.sample entropypilot/.env
```

Edit `entropypilot/.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```

## Running the Examples

The project includes two Jupyter notebook examples that demonstrate entropy control principles.

### Starting Jupyter

Activate the virtual environment and start Jupyter:

```bash
# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Start Jupyter
jupyter notebook
```

Alternatively, use uv to run Jupyter directly:

```bash
uv run jupyter notebook
```

### Example 1: Full Demo (`demo.ipynb`)

Location: `entropypilot/demo.ipynb`

This comprehensive notebook includes:

1. **Theory Section**: Mathematical explanation of compounding risk in LLM agents
2. **Attention Mechanism Analysis**: Visualizes how transformers process negative vs. affirmative constraints
3. **Color Universe Visualization**: 3D plots showing the probability space of affirmative vs. negative prompts
4. **Practical Experiment**: Generates 100+ color palettes to measure violation rates
5. **Statistical Analysis**: Compares success rates between constraint types

**Key experiments:**
- Attention weight visualization using GPT-2
- Color palette generation with GPT-4o-mini
- Statistical comparison of constraint violations

**Expected outcomes:**
- Negative constraints show higher violation rates
- Affirmative constraints collapse the search space
- Visual proof that entropy affects reliability

### Example 2: Simplified Demo (`demo_sync.ipynb`)

Location: `entropypilot/demo_sync.ipynb`

A streamlined version focusing on the core experiment:

1. Quick setup and configuration
2. Color palette generation (negative vs. affirmative)
3. 100-run simulation with statistical tracking
4. Visualization of results

**Use this notebook if you want:**
- A faster demonstration
- To focus on the practical results
- To modify and run your own experiments

## Project Structure

```
EntropyPilot/
├── entropypilot/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── main.py                # Main entry point
│   ├── simulation.py          # Simulation logic and statistics
│   ├── color_universe.py      # 3D visualization of probability spaces
│   ├── demo.ipynb            # Full demonstration notebook
│   ├── demo_sync.ipynb       # Simplified demonstration
│   ├── .env.sample           # Environment template
│   └── utils/
│       ├── __init__.py
│       ├── color.py          # Color validation utilities
│       ├── llm.py            # LLM interaction helpers
│       ├── models.py         # Pydantic models
│       └── visualization.py  # Plotting utilities
├── pyproject.toml            # Project dependencies
├── uv.lock                   # Locked dependency versions
└── README.md                 # This file
```

## Key Concepts

### Entropy as a Reliability Framework

Reliability isn't about making models "smarter"—it's about reducing the number of valid worlds they can inhabit at each decision point. Ambiguity introduced at any level expands the search space and distributes probability mass across multiple interpretations.

### Current Example: Affirmative vs. Negative Constraints

This demonstration uses color palette generation to visualize entropy differences:

**Negative Constraint (High Entropy):**
```
"Generate a palette. DO NOT use red or orange."
```
- Model must consider entire color space
- Must apply negation logic
- Leaves probability distribution wide open

**Affirmative Constraint (Low Entropy):**
```
"Generate a palette using ONLY cool blues, aquas, and teals."
```
- Collapses probability space immediately
- Direct attention to target tokens
- Eliminates ambiguity

### Why This Matters

In multi-step LLM agents:
- Small amounts of entropy compound across steps
- A 95% success rate per step → 59% success over 10 steps
- Every ambiguous decision point—whether in prompts, data structures, or system architecture—contributes to this compounding risk
- Reliability emerges from systematically collapsing the probability space before the model ever has to choose

## Troubleshooting

### "No module named 'openai'" or similar import errors

Make sure you've installed dependencies and activated the virtual environment:
```bash
uv sync
source .venv/bin/activate
```

### "Missing API key" error

Verify that:
1. `.env` file exists in the `entropypilot/` directory
2. It contains a valid OpenAI API key
3. The key is in the format: `OPENAI_API_KEY=sk-...`

### Jupyter kernel not found

Install the IPython kernel in your virtual environment:
```bash
uv pip install ipykernel
python -m ipykernel install --user --name=entropypilot
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Citation

If you use this work in your research, please cite:

```
[Add citation information here]
```
