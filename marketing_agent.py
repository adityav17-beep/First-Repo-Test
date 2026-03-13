"""
Simple Marketing Agent powered by Claude.

Uses the Anthropic beta tool runner to give Claude four marketing tools:
  - draft_copy       : Write marketing copy for a given channel
  - analyze_audience : Build a target audience persona
  - plan_campaign    : Create a campaign brief
  - suggest_channels : Recommend the best marketing channels

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key_here
    python marketing_agent.py
"""

import json
import os
import anthropic
from anthropic import beta_tool

# ---------------------------------------------------------------------------
# Marketing tools
# ---------------------------------------------------------------------------

@beta_tool
def draft_copy(
    product: str,
    channel: str,
    tone: str = "professional",
    audience: str = "general consumers",
) -> str:
    """Draft marketing copy for a product.

    Args:
        product: The product or service being marketed.
        channel: The marketing channel (e.g. email, Instagram, Google Ad, LinkedIn).
        tone: The desired tone (e.g. professional, playful, urgent, inspirational).
        audience: A short description of the target audience.
    """
    # This tool's logic is handled by Claude itself — it uses the description
    # to know what to generate. We return a signal so Claude fills it in.
    return json.dumps({
        "instruction": "draft_copy",
        "product": product,
        "channel": channel,
        "tone": tone,
        "audience": audience,
    })


@beta_tool
def analyze_audience(
    product: str,
    industry: str,
    goals: str = "",
) -> str:
    """Build a target audience persona for a product.

    Args:
        product: The product or service being marketed.
        industry: The industry or vertical (e.g. SaaS, e-commerce, healthcare).
        goals: Optional marketing goals (e.g. brand awareness, lead generation).
    """
    return json.dumps({
        "instruction": "analyze_audience",
        "product": product,
        "industry": industry,
        "goals": goals,
    })


@beta_tool
def plan_campaign(
    product: str,
    objective: str,
    budget_range: str = "unspecified",
    duration_weeks: int = 4,
) -> str:
    """Create a campaign plan / brief.

    Args:
        product: The product or service being marketed.
        objective: The campaign objective (e.g. launch, retention, seasonal promotion).
        budget_range: Budget tier such as 'small (<$5k)', 'medium ($5k–$50k)', 'large (>$50k)'.
        duration_weeks: How many weeks the campaign should run.
    """
    return json.dumps({
        "instruction": "plan_campaign",
        "product": product,
        "objective": objective,
        "budget_range": budget_range,
        "duration_weeks": duration_weeks,
    })


@beta_tool
def suggest_channels(
    product: str,
    audience_age_range: str = "18-45",
    b2b_or_b2c: str = "B2C",
    budget_range: str = "medium ($5k–$50k)",
) -> str:
    """Recommend marketing channels for a product.

    Args:
        product: The product or service being marketed.
        audience_age_range: Age range of the target audience (e.g. '25-40').
        b2b_or_b2c: Whether this is a B2B or B2C product.
        budget_range: Budget tier to tailor channel recommendations.
    """
    return json.dumps({
        "instruction": "suggest_channels",
        "product": product,
        "audience_age_range": audience_age_range,
        "b2b_or_b2c": b2b_or_b2c,
        "budget_range": budget_range,
    })


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert marketing strategist and copywriter with deep experience
across B2B and B2C marketing. You help users with:

- Drafting compelling marketing copy for any channel
- Defining and analyzing target audiences
- Planning marketing campaigns from start to finish
- Recommending the most effective marketing channels

When a user asks a marketing question, use the available tools to structure your
thinking, then deliver a clear, actionable, and specific response. Always tailor
your advice to the user's product, audience, and goals.

Available tools:
  • draft_copy       – write channel-specific marketing copy
  • analyze_audience – define a target audience persona
  • plan_campaign    – build a campaign brief
  • suggest_channels – recommend marketing channels

Be concise but thorough. Format your responses with headers and bullet points
where appropriate so they are easy to scan."""


def run_agent(user_message: str, client: anthropic.Anthropic) -> str:
    """Run one turn of the marketing agent and return the final text response."""
    tools = [draft_copy, analyze_audience, plan_campaign, suggest_channels]
    messages = [{"role": "user", "content": user_message}]

    runner = client.beta.messages.tool_runner(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=messages,
    )

    last_text = ""
    for message in runner:
        for block in message.content:
            if hasattr(block, "type") and block.type == "text":
                last_text = block.text

    return last_text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Export it with:  export ANTHROPIC_API_KEY=your_key_here")
        return

    client = anthropic.Anthropic(api_key=api_key)

    print("=" * 60)
    print("  Marketing Agent  (powered by Claude)")
    print("=" * 60)
    print("Ask me anything about marketing — copy, campaigns,")
    print("audiences, channels, and more.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        print("\nAgent: ", end="", flush=True)
        try:
            response = run_agent(user_input, client)
            print(response)
        except anthropic.AuthenticationError:
            print("Authentication failed. Check your ANTHROPIC_API_KEY.")
            break
        except anthropic.RateLimitError:
            print("Rate limited. Please wait a moment and try again.")
        except anthropic.APIError as e:
            print(f"API error: {e}")
        print()


if __name__ == "__main__":
    main()
