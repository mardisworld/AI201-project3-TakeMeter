"""
prompts.py — the prompts used in this project, built from a single LABEL_MAP.

Three prompts live here:
  1. classification_system_prompt()  -> tells the LLM how to classify, from LABEL_MAP
  2. classification_user_prompt(text) -> wraps one comment for the zero-shot baseline
  3. STRESS_TEST_PROMPT                -> generates boundary cases to stress-test labels

Keeping LABEL_MAP here as the single source of truth means editing a definition
updates every prompt automatically — the prompts never drift from the labels the
model is actually trained on.
"""

from labels import LABEL_MAP  # single source of truth for the label taxonomy

# Explicit tie-breakers for the boundaries that are genuinely ambiguous (see planning.md §3).
# These are appended to the system prompt so the LLM resolves edge cases the same way a
# human annotator would.
DECISION_RULES = [
    "analysis vs op_response: if removing the opinion framing leaves evidence that still "
    "supports the claim on its own, choose `analysis`. A lone or cherry-picked statistic "
    "used only to score a point is `op_response`.",
    "op_response vs response: if the comment gives ANY reason, mechanism, or 'because', "
    "choose `op_response`. If it only signals agreement/disagreement or emotion with no "
    "reason attached, choose `response`.",
    "question vs response: choose `question` only when the comment genuinely seeks "
    "information or an answer. A rhetorical question used to attack or react is `response`.",
]


def _label_block():
    """Render the label definitions as a numbered list for a prompt."""
    lines = []
    for name, (idx, definition) in LABEL_MAP.items():
        lines.append(f"- {name} (id {idx}): {definition}")
    return "\n".join(lines)


def classification_system_prompt():
    """System prompt: how to classify a comment using LABEL_MAP. Single source of truth."""
    labels = ", ".join(LABEL_MAP.keys())
    rules = "\n".join(f"- {r}" for r in DECISION_RULES)
    return f"""You are a precise text classifier for comments in an online political \
discussion thread (r/PoliticalDiscussion). Classify each comment into EXACTLY ONE of \
these labels based on what kind of contribution it makes to the discussion:

{_label_block()}

Decision rules for ambiguous comments:
{rules}

Rules:
- Choose exactly one label. The labels are mutually exclusive.
- Judge what the comment IS DOING (argument vs. opinion vs. reaction vs. question), \
not whether you agree with it or whether its claims are true.
- Do not explain your reasoning. Respond with ONLY the label name, lowercase, exactly \
one of: {labels}."""


def classification_user_prompt(comment_text):
    """User prompt: wrap a single comment for the zero-shot baseline."""
    return f"Classify this comment:\n\n\"\"\"\n{comment_text}\n\"\"\"\n\nLabel:"


# ---------------------------------------------------------------------------
# Label stress-testing prompt (planning.md §7, AI Tool Plan).
# Paste this into an LLM BEFORE annotating to find weaknesses in the definitions.
# ---------------------------------------------------------------------------
STRESS_TEST_PROMPT = f"""I am designing a text classifier for comments in an online \
political discussion thread. Here are my labels and their definitions:

{_label_block()}

And here are my decision rules for ambiguous cases:
{chr(10).join('- ' + r for r in DECISION_RULES)}

Your task: write 8-10 short comments (1-3 sentences each, in the style of real Reddit \
political-discussion comments) that sit RIGHT ON THE BOUNDARY between two of these \
labels — cases where a reasonable annotator could argue for either label.

For each comment, output:
1. the comment text,
2. the two labels it is caught between,
3. one sentence on why it is genuinely ambiguous.

Do not write easy/clear-cut examples. The goal is to find where my definitions are too \
loose so I can tighten them before I annotate 200+ examples."""


if __name__ == "__main__":
    # Quick preview of all three prompts.
    print("=" * 70, "\nCLASSIFICATION SYSTEM PROMPT\n", "=" * 70, sep="")
    print(classification_system_prompt())
    print("\n" + "=" * 70, "\nCLASSIFICATION USER PROMPT (example)\n", "=" * 70, sep="")
    print(classification_user_prompt("Congress is broken because the filibuster blocks everything."))
    print("\n" + "=" * 70, "\nLABEL STRESS-TEST PROMPT\n", "=" * 70, sep="")
    print(STRESS_TEST_PROMPT)
