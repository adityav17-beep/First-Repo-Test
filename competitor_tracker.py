#!/usr/bin/env python3
"""
Competitor Intelligence Tracker
Focus: Decision Intelligence & Agentic AI

Asks which competitors you want to track, visits their websites,
and produces a focused 1-page brief per competitor — saved as a
Markdown file and printed to the terminal.

Run it once a week to see what changed.

Usage:
    python competitor_tracker.py
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SNAPSHOT_DIR = Path("snapshots")
REPORTS_DIR = Path("reports")
COMPETITORS_FILE = Path("competitors.json")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SCRAPE_PROMPT = """You are a competitive intelligence analyst specialising in
decision intelligence, agentic AI, and enterprise automation.

Visit the URL below and parse the homepage only. Do not follow any links or navigate to other pages.

URL: {url}

Extract everything relevant to these three themes:
  1. Decision intelligence  (data-driven decisions, supply chain, planning, forecasting)
  2. Agentic tools          (AI agents, autonomous workflows, copilots, LLM-based automation)
  3. Enterprise AI solutions (platforms, APIs, industry-specific AI)

Return ONLY a valid JSON object — no markdown fences, no explanation:

{{
  "company_name": "...",
  "tagline": "main tagline or value proposition as written on the site",
  "positioning": "how they describe themselves in 2-3 sentences, using their own language",
  "products_and_solutions": [
    {{"name": "...", "description": "what it does, 1-2 sentences"}}
  ],
  "recent_launches": ["specific announcements, new products, or updates found"],
  "target_audience": "industries, roles, or company types they are targeting",
  "key_claims": ["the 2-4 strongest differentiation claims they make"],
  "pricing_or_business_model": "any pricing info or 'not publicly listed'",
  "notable_partnerships_or_customers": ["any featured partners or customers"]
}}

Quote actual text from the site wherever possible. Be specific."""


REPORT_PROMPT = """You are a competitive intelligence analyst.
Write a 1-page intelligence brief about {company_name} based on the data below.
The brief is for a business leader evaluating the competitive landscape in
decision intelligence and agentic AI.

SCRAPED DATA:
{data}

PREVIOUS SNAPSHOT ({prev_label}):
{prev_data}

---
Write the brief in this exact format (use Markdown):

# {company_name} — Competitive Intelligence Brief
**{date}  |  Focus: Decision Intelligence & Agentic AI**

---

## Positioning
How they describe themselves in this space. Use their actual language. 1 short paragraph.

## Key Products & Solutions
- **Product name**: what it does and why it matters in this space (one line each)
(list 3–6 most relevant)

## Recent Launches & Announcements
- (bullet list of the most recent things they are promoting — be specific)
If nothing notable found, say "No recent launches identified on the site."

## Who They Are Targeting
One sentence on industries/roles/company sizes they are going after.

## How They Differentiate
Their 2–4 strongest claims vs the market. Use their language where possible.

## What Changed Since Last Time
(Only include this section if a previous snapshot exists and something meaningful changed.
If this is the first run, omit the section entirely.)

## Strategic Takeaway
2–3 sentences: what does their current position tell you about where they are heading
in decision intelligence and agentic AI?

---
Keep the entire brief to approximately 400–500 words. Be direct and specific.
Avoid vague phrases like "they are focused on innovation". Quote real product names."""


# ---------------------------------------------------------------------------
# competitors.json helpers
# ---------------------------------------------------------------------------

def load_competitors() -> dict[str, str]:
    """Return {name: url} dict from competitors.json."""
    if not COMPETITORS_FILE.exists():
        default = {
            "Palantir": "https://www.palantir.com",
            "IBM": "https://www.ibm.com",
            "o9 Solutions": "https://o9solutions.com",
            "Microsoft": "https://www.microsoft.com",
            "Databricks": "https://www.databricks.com"
        }
        COMPETITORS_FILE.write_text(json.dumps(default, indent=2))
        print(f"Created {COMPETITORS_FILE} with 5 default competitors.\n")
        return default
    return json.loads(COMPETITORS_FILE.read_text())


def ask_which_competitors(competitors: dict[str, str]) -> dict[str, str]:
    """Print the list and let the user pick which ones to run."""
    names = list(competitors.keys())

    print("\nConfigured competitors:")
    for i, (name, url) in enumerate(competitors.items(), 1):
        print(f"  {i}. {name:<20} {url}")

    print("\nWhich competitors do you want to track?")
    print("  • Enter numbers separated by commas  (e.g.  1,3,5)")
    print("  • Or press Enter to track all of them")

    raw = input("\nYour choice: ").strip()

    if not raw:
        print(f"\nTracking all {len(names)} competitors.\n")
        return competitors

    selected = {}
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(names):
                name = names[idx]
                selected[name] = competitors[name]
            else:
                print(f"  (skipping '{part}' — out of range)")
        else:
            print(f"  (skipping '{part}' — not a number)")

    if not selected:
        print("No valid selection — tracking all.\n")
        return competitors

    print(f"\nTracking: {', '.join(selected.keys())}\n")
    return selected


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def snapshot_path(name: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", name)
    return SNAPSHOT_DIR / f"{safe}.json"


def load_previous_snapshot(name: str) -> "dict | None":
    path = snapshot_path(name)
    return json.loads(path.read_text()) if path.exists() else None


def save_snapshot(name: str, data: dict) -> None:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    data["scraped_at"] = datetime.now().isoformat()
    snapshot_path(name).write_text(json.dumps(data, indent=2))


def save_report(name: str, report: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    safe = re.sub(r"[^\w\-]", "_", name)
    path = REPORTS_DIR / f"{safe}_{date_str}.md"
    path.write_text(report)
    return path


# ---------------------------------------------------------------------------
# Agent calls
# ---------------------------------------------------------------------------

def scrape_competitor(client: anthropic.Anthropic, name: str, url: str) -> dict:
    """
    Claude visits the URL using the server-side web_fetch tool and returns
    structured JSON intelligence. Handles pause_turn automatically.
    """
    messages = [
        {"role": "user", "content": SCRAPE_PROMPT.format(url=url)}
    ]

    for _ in range(10):   # re-send up to 10 times if Claude pauses mid-fetch
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            tools=[{"type": "web_fetch_20260209", "name": "web_fetch"}],
            messages=messages,
        )
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        break

    text = next(
        (b.text for b in response.content if getattr(b, "type", "") == "text"),
        ""
    )

    # Strip markdown fences if Claude wrapped the JSON
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"company_name": name, "raw_response": text, "parse_error": True}


def generate_report(
    client: anthropic.Anthropic,
    name: str,
    current: dict,
    previous: "dict | None",
) -> str:
    """Ask Claude to write the 1-page brief from the scraped data."""
    prev_label = previous.get("scraped_at", "")[:10] if previous else "no previous snapshot"
    prev_data = json.dumps(previous, indent=2) if previous else "None — this is the first run."

    prompt = REPORT_PROMPT.format(
        company_name=name,
        data=json.dumps(current, indent=2),
        prev_label=prev_label,
        prev_data=prev_data,
        date=datetime.now().strftime("%B %d, %Y"),
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    return next(
        (b.text for b in response.content if getattr(b, "type", "") == "text"),
        "Report generation failed."
    )


# ---------------------------------------------------------------------------
# Per-competitor workflow
# ---------------------------------------------------------------------------

def process_competitor(client: anthropic.Anthropic, name: str, url: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"  {url}")
    print(f"{'─' * 60}")

    previous = load_previous_snapshot(name)

    print("  Step 1/2  Scraping website ...", end="", flush=True)
    current = scrape_competitor(client, name, url)
    save_snapshot(name, current)
    print(" done.")

    if current.get("parse_error"):
        print("  Warning: could not parse structured data from this site.")
        print(f"  Raw snapshot saved to {snapshot_path(name)}")
        return

    print("  Step 2/2  Writing 1-page brief ...", end="", flush=True)
    report = generate_report(client, name, current, previous)
    report_file = save_report(name, report)
    print(" done.")

    print(f"\n{'═' * 60}")
    print(report)
    print(f"{'═' * 60}")
    print(f"  Saved to: {report_file}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nError: ANTHROPIC_API_KEY is not set.")
        print("  Mac/Linux:  export ANTHROPIC_API_KEY=your_key_here")
        print("  Windows:    set ANTHROPIC_API_KEY=your_key_here\n")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("=" * 60)
    print("  Competitor Intelligence Tracker")
    print("  Focus: Decision Intelligence & Agentic AI")
    print("=" * 60)

    competitors = load_competitors()
    selected = ask_which_competitors(competitors)

    total = len(selected)
    for i, (name, url) in enumerate(selected.items(), 1):
        print(f"\n[{i}/{total}] Processing {name}…")
        try:
            process_competitor(client, name, url)
        except anthropic.AuthenticationError:
            print("\nAuthentication failed — check your ANTHROPIC_API_KEY.")
            sys.exit(1)
        except anthropic.RateLimitError:
            print(f"\nRate limited on {name}. Wait a moment and retry.")
        except anthropic.APIError as e:
            print(f"\nAPI error for {name}: {e}")
        except Exception as e:
            print(f"\nUnexpected error for {name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  All done. Reports saved to ./{REPORTS_DIR}/")
    print(f"  Run again next week to see what changed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
