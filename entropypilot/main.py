"""
Controlling LLM Entropy: Why Negation Fails (Attention Mechanism Demo)

KEY INSIGHT:
Building reliable LLMs requires controlling the probability distribution (entropy) of outputs.
Negation language INCREASES entropy on unwanted tokens by placing them in the attention context.

What this demonstrates:
1. When you say "DON'T run", you're putting "run" into the model's attention
2. The attention mechanism latches onto these tokens, increasing their probability
3. This is counterintuitive but critical for prompt engineering

The Experiment (Completely Neutral):
- Task: "After hearing the lunch bell ring, the students began to..."
- No bias towards fast or slow actions in the base task
- Both fast and slow actions are value-neutral descriptors

Two approaches:
A) AFFIRMATIVE: "walk" - suggests SLOW actions, does NOT mention FAST actions at all
B) NEGATION: "DON'T run, sprint, rush" - mentions FAST actions (the opposite of what we want!)

Expected Result:
- NEGATION approach will show HIGHER probability for "run", "sprint", "rush" tokens
- AFFIRMATIVE approach will show LOWER probability for fast actions
- This proves negation primes attention on exactly what you're trying to avoid
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt

# --- OpenAI client (adjust to your environment) ---
# pip install openai
from openai import OpenAI

from entropypilot.config import config

os.environ["OPENAI_API_KEY"] = config.openai_api_key

client = OpenAI()


SYSTEM_NEGATION = """You are a creative writing assistant helping an author write realistic fiction about everyday school life.

CRITICAL WRITING GUIDELINES:
- Do NOT use violent or aggressive language like: attack, fight, assault, hit, punch, strike, combat, battle
- Do NOT describe any physical confrontations or conflicts between characters
- Avoid all aggressive verbs: charge, rush, storm, barge, shove, push, force
- Never depict students engaging in hostile or combative behavior
- Focus on peaceful, everyday school interactions

Your writing should be appropriate for middle-grade readers and depict positive social dynamics."""

SYSTEM_AFFIRMATIVE = """You are a creative writing assistant helping an author write realistic fiction about everyday school life.

WRITING GUIDELINES:
- Use calm, peaceful language that depicts cooperative social interactions
- Describe characters who communicate respectfully and resolve differences through dialogue
- Emphasize positive behaviors: listening, discussing, sharing, helping, collaborating
- Show characters working together and supporting each other
- Depict everyday moments of friendship, teamwork, and mutual respect

Your writing should be appropriate for middle-grade readers and showcase positive social dynamics."""

USER_QUERY = """Write a short scene about two students who had a disagreement yesterday meeting each other in the hallway before class."""

AVOIDED_TOKENS = ["fight", "fighting", "fought", "attack", "attacking", "attacked",
                  "punch", "punching", "hit", "hitting", "strike", "striking",
                  "shove", "shoving", "push", "pushing", "aggressive", "aggression",
                  "hostile", "hostility", "anger", "angry", "mad", "furious"]

DESIRED_TOKENS = ["talk", "talking", "discuss", "discussing", "listen", "listening",
                  "calm", "calmly", "peaceful", "peacefully", "apologize", "apologizing",
                  "sorry", "understand", "understanding", "friend", "friends", "friendly",
                  "respect", "respectful", "kind", "kindly", "gentle", "gently"]


# -----------------------------
# 2) Helpers
# -----------------------------
def _display_token(tok: str) -> str:
    if tok == " ":
        return "␣"
    if tok == "\n":
        return "↵"
    if tok == "\t":
        return "⇥"
    return tok


def _logprob_to_prob(lp: float) -> float:
    return math.exp(lp)


@dataclass
class TokenProbabilities:
    position: int
    generated_token: str
    top_candidates: Dict[str, float]


@dataclass
class GenerationResult:
    system_prompt: str
    user_prompt: str
    full_text: str
    tokens: List[TokenProbabilities]


def generate_with_logprobs(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 100,
    top_n: int = 20,
) -> GenerationResult:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.8,
        logprobs=True,
        top_logprobs=top_n,
    )

    choice = resp.choices[0]
    generated_text = choice.message.content or ""

    tokens: List[TokenProbabilities] = []

    if choice.logprobs and choice.logprobs.content:
        for idx, token_data in enumerate(choice.logprobs.content):
            top_candidates: Dict[str, float] = {}

            if token_data.top_logprobs:
                for candidate in token_data.top_logprobs:
                    top_candidates[candidate.token] = _logprob_to_prob(candidate.logprob)

            tokens.append(TokenProbabilities(
                position=idx,
                generated_token=token_data.token,
                top_candidates=top_candidates,
            ))

    return GenerationResult(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        full_text=generated_text,
        tokens=tokens,
    )


def plot_token_evolution(
    result: GenerationResult,
    title: str,
    highlight_tokens: List[str] | None = None,
    max_tokens_display: int = 40,
) -> None:
    highlight_tokens = highlight_tokens or []
    highlight_lower = [t.lower() for t in highlight_tokens]

    tokens_to_show = result.tokens[:max_tokens_display]

    if not tokens_to_show:
        print(f"No tokens to display for: {title}")
        return

    tracked_tokens = []
    for token_prob in tokens_to_show:
        for token, prob in token_prob.top_candidates.items():
            if any(hl in token.lower() for hl in highlight_lower):
                tracked_tokens.append(token)

    tracked_unique = sorted(set(tracked_tokens))

    if not tracked_unique:
        print(f"No tracked tokens found in: {title}")
        return

    matrix = []
    for tracked in tracked_unique:
        row = []
        for token_prob in tokens_to_show:
            prob = token_prob.top_candidates.get(tracked, 0.0)
            row.append(prob)
        matrix.append(row)

    num_positions = len(tokens_to_show)
    fig, ax = plt.subplots(figsize=(max(20, num_positions * 0.5), max(8, len(tracked_unique) * 0.4)))

    im = ax.imshow(matrix, aspect="auto", cmap="Reds", interpolation="nearest", vmin=0, vmax=0.5)

    ax.set_xticks(range(0, num_positions, 5))
    ax.set_xticklabels([f"{i}" for i in range(0, num_positions, 5)], fontsize=9)

    ax.set_yticks(range(len(tracked_unique)))
    ax.set_yticklabels([_display_token(t) for t in tracked_unique], fontsize=10)

    ax.set_xlabel("Token Position in Generation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tracked Tokens (Avoided)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Probability", fontsize=12, fontweight="bold")

    for i in range(len(tracked_unique)):
        for j in range(num_positions):
            val = matrix[i][j]
            if val >= 0.05:
                ax.text(j, i, f"{val:.2f}",
                       ha="center", va="center",
                       fontsize=7, color="white" if val > 0.2 else "black", fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_entropy_comparison(
    result_a: GenerationResult,
    result_b: GenerationResult,
    tokens_a: List[str],
    tokens_b: List[str],
) -> None:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    def calc_token_mass(token_probs: TokenProbabilities, tokens: List[str]) -> float:
        tokens_lower = [t.lower() for t in tokens]
        total = 0.0
        for token, prob in token_probs.top_candidates.items():
            if any(t in token.lower() for t in tokens_lower):
                total += prob
        return total

    max_positions = min(len(result_a.tokens), len(result_b.tokens))

    a_group_a = [calc_token_mass(tp, tokens_a) for tp in result_a.tokens[:max_positions]]
    b_group_a = [calc_token_mass(tp, tokens_a) for tp in result_b.tokens[:max_positions]]

    a_group_b = [calc_token_mass(tp, tokens_b) for tp in result_a.tokens[:max_positions]]
    b_group_b = [calc_token_mass(tp, tokens_b) for tp in result_b.tokens[:max_positions]]

    positions = list(range(max_positions))

    ax1.plot(positions, a_group_a, 'r-', linewidth=3, label='Prompt A', marker='o', markersize=4)
    ax1.plot(positions, b_group_a, 'b-', linewidth=3, label='Prompt B', marker='s', markersize=4)
    ax1.set_xlabel("Token Position", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Probability Mass", fontsize=11, fontweight="bold")
    ax1.set_title("Avoided Token Probability", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.plot(positions, a_group_b, 'r-', linewidth=3, label='Prompt A', marker='o', markersize=4)
    ax2.plot(positions, b_group_b, 'b-', linewidth=3, label='Prompt B', marker='s', markersize=4)
    ax2.set_xlabel("Token Position", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability Mass", fontsize=11, fontweight="bold")
    ax2.set_title("Desired Token Probability", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    total_a_group_a = sum(a_group_a)
    total_b_group_a = sum(b_group_a)
    ax3.bar(['Prompt A', 'Prompt B'],
            [total_a_group_a, total_b_group_a],
            color=['#ff4444', '#4444ff'], alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel("Total Probability Mass", fontsize=11, fontweight="bold")
    ax3.set_title("Total Avoided Token Probability", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis='y')

    ax4.axis('off')
    text_content = f"""PROMPT A (System):
{result_a.system_prompt[:150]}...

Generated:
{result_a.full_text[:150]}...

{'─' * 60}

PROMPT B (System):
{result_b.system_prompt[:150]}...

Generated:
{result_b.full_text[:150]}...
"""
    ax4.text(0.05, 0.95, text_content,
            transform=ax4.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.suptitle("Entropy Control: Prompt Comparison",
                 fontsize=15, fontweight="bold", y=0.998)
    plt.tight_layout()
    plt.show()


def main() -> None:
    print("=" * 80)
    print("ENTROPY CONTROL DEMO")
    print("=" * 80)
    print()
    print("EXPERIMENT SETUP:")
    print(f"User Query: {USER_QUERY}")
    print(f"Avoided tokens: {', '.join(AVOIDED_TOKENS[:8])}")
    print(f"Desired tokens: {', '.join(DESIRED_TOKENS[:8])}")
    print()
    print("=" * 80)
    print()

    print("Generating with System Prompt A...")
    result_a = generate_with_logprobs(
        SYSTEM_NEGATION,
        USER_QUERY,
        max_tokens=100,
        top_n=20,
    )
    print(f"Generated: {result_a.full_text}")
    print()

    print("Generating with System Prompt B...")
    result_b = generate_with_logprobs(
        SYSTEM_AFFIRMATIVE,
        USER_QUERY,
        max_tokens=100,
        top_n=20,
    )
    print(f"Generated: {result_b.full_text}")
    print()

    def contains_tokens(text: str, tokens: List[str]) -> List[str]:
        found = []
        for token in tokens:
            if token.lower() in text.lower():
                found.append(token)
        return found

    a_has_avoided = contains_tokens(result_a.full_text, AVOIDED_TOKENS)
    b_has_avoided = contains_tokens(result_b.full_text, AVOIDED_TOKENS)

    a_has_desired = contains_tokens(result_a.full_text, DESIRED_TOKENS)
    b_has_desired = contains_tokens(result_b.full_text, DESIRED_TOKENS)

    print("=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print(f"Prompt A output contains avoided tokens: {a_has_avoided if a_has_avoided else 'None'}")
    print(f"Prompt B output contains avoided tokens: {b_has_avoided if b_has_avoided else 'None'}")
    print(f"Prompt A output contains desired tokens: {a_has_desired if a_has_desired else 'None'}")
    print(f"Prompt B output contains desired tokens: {b_has_desired if b_has_desired else 'None'}")
    print()

    print("Creating visualizations...")
    print()

    print("Visualization 1: Prompt A entropy distribution...")
    plot_token_evolution(
        result_a,
        title="Prompt A: Token Probability Heatmap\n(🔴 = Avoided Tokens)",
        highlight_tokens=AVOIDED_TOKENS,
        max_tokens_display=80,
    )

    print("Visualization 2: Prompt B entropy distribution...")
    plot_token_evolution(
        result_b,
        title="Prompt B: Token Probability Heatmap\n(🔴 = Avoided Tokens)",
        highlight_tokens=AVOIDED_TOKENS,
        max_tokens_display=80,
    )

    print("Visualization 3: Comparison...")
    plot_entropy_comparison(
        result_a,
        result_b,
        AVOIDED_TOKENS,
        DESIRED_TOKENS,
    )

    print()
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print()
    print("Negation in prompts can increase probability of unwanted tokens")
    print("by placing them in the attention context.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
