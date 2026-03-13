#!/usr/bin/env python3
"""
Competitor Intelligence Tracker

Visits your competitors' websites, extracts structured intelligence
(positioning, products, key messages, pricing), saves a snapshot, and on
every subsequent run compares the new snapshot against the previous one to
surface what changed — new product launches, repositioning, messaging shifts.

Setup:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key_here

First run (saves baseline snapshots):
    python competitor_tracker.py

Track a single URL ad-hoc:
    python competitor_tracker.py --url https://competitor.com

Subsequent runs surface what changed since last time:
    python competitor_tracker.py
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SNAPSHOT_DIR = Path("snapshots")
COMPETITORS_FILE = Path("competitors.json")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """You are a competitive intelligence analyst.

Visit the URL provided, then return ONLY a valid JSON object — no markdown fences,
no explanation — with these exact fields:

{
  "company_name": "...",
  "tagline": "main tagline or value proposition as written on the site",
  "positioning_statement": "how they position themselves in the market (2-3 sentences drawn from the site copy)",
  "key_messages": ["message 1", "message 2", "..."],
  "products_and_solutions": [
    {"name": "product name", "description": "what it does"}
  ],
  "pricing_info": "any pricing details found, or 'not publicly listed'",
  "recent_launches": ["any new announcements, product launches, or updates found on the site"],
  "target_audience": "who they are targeting based on site copy",
  "differentiators": ["what they claim makes them different"]
}

Be specific — quote actual text from the site. Check the homepage, nav bar,
product/solutions pages, and any 'What's New' or blog sections you can reach."""


COMPARE_PROMPT = """You are a competitive intelligence analyst.

Compare these two snapshots of a competitor website and report what changed.

PREVIOUS SNAPSHOT  ({prev_date}):
{prev}

CURRENT SNAPSHOT  ({curr_date}):
{curr}

Write a concise intelligence brief with these sections:

## Positioning Changes
Did their core value proposition or tagline shift? Quote before/after if so.

## New Products or Solutions
Any additions to their lineup? Anything removed?

## Messaging Updates
New key messages, dropped messages, or tone/audience shifts?

## Recent Launches or Announcements
What are they actively promoting right now?

## Pricing Changes
Any changes to pricing visibility or strategy?

## Strategic Signal
In 2-3 sentences: what do these changes tell you about where they are heading?

If nothing meaningful changed, say so explicitly. Focus on what matters strategically."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_competitors() -> list[str]:
    """Load competitor URLs from config, creating a template if missing."""
    if not COMPETITORS_FILE.exists():
        template = {
            "competitors": [
                "https://www.hubspot.com",
                "https://www.salesforce.com"
            ],
            "_note": "Replace the example URLs with your actual competitors and re-run."
        }
        COMPETITORS_FILE.write_text(json.dumps(template, indent=2))
        print(f"Created {COMPETITORS_FILE} with example competitors.")
        print("Edit it to add your own competitors, then re-run.\n")
    data = json.loads(COMPETITORS_FILE.read_text())
    return data.get("competitors", [])


def snapshot_path(url: str) -> Path:
    safe = re.sub(r"[^\w\-.]", "_", url.replace("https://", "").replace("http://", ""))
    return SNAPSHOT_DIR / f"{safe}.json"


def load_previous_snapshot(url: str) -> dict | None:
    path = snapshot_path(url)
    return json.loads(path.read_text()) if path.exists() else None


def save_snapshot(url: str, data: dict) -> None:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    data["url"] = url
    data["scraped_at"] = datetime.now().isoformat()
    snapshot_path(url).write_text(json.dumps(data, indent=2))


def extract_json(text: str) -> dict:
    """Parse JSON from Claude's response, tolerating markdown fences."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back: grab the outermost {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No valid JSON found in response")

# ---------------------------------------------------------------------------
# Core agent calls
# ---------------------------------------------------------------------------


def scrape_competitor(client: anthropic.Anthropic, url: str) -> dict:
    """
    Ask Claude to fetch the competitor URL (via server-side web_fetch) and
    return structured intelligence as JSON.
    """
    messages = [
        {
            "role": "user",
            "content": (
                f"Visit this competitor's website and extract intelligence:\n\n"
                f"{url}\n\n{EXTRACT_PROMPT}"
            ),
        }
    ]

    # Claude may pause mid-task if the server-side web_fetch loop hits its
    # iteration limit. Re-send the conversation to resume.
    for _ in range(8):
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            tools=[{"type": "web_fetch_20260209", "name": "web_fetch"}],
            messages=messages,
        )

        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        # Reached end_turn — extract the text block with our JSON
        break

    text = next(
        (b.text for b in response.content if getattr(b, "type", "") == "text"),
        "",
    )

    try:
        return extract_json(text)
    except (ValueError, json.JSONDecodeError):
        return {"raw_response": text, "parse_error": True}


def compare_snapshots(
    client: anthropic.Anthropic,
    previous: dict,
    current: dict,
) -> str:
    """Ask Claude to diff two snapshots and produce a change brief."""
    prompt = COMPARE_PROMPT.format(
        prev_date=previous.get("scraped_at", "unknown")[:10],
        curr_date=current.get("scraped_at", datetime.now().isoformat())[:10],
        prev=json.dumps(previous, indent=2),
        curr=json.dumps(current, indent=2),
    )
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(
        (b.text for b in response.content if getattr(b, "type", "") == "text"),
        "No analysis generated.",
    )

# ---------------------------------------------------------------------------
# Per-competitor workflow
# ---------------------------------------------------------------------------


def track_competitor(client: anthropic.Anthropic, url: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {url}")
    print(f"{'─' * 60}")

    previous = load_previous_snapshot(url)

    print("  Scraping ...", end="", flush=True)
    current = scrape_competitor(client, url)
    save_snapshot(url, current)
    print(" done.")

    if current.get("parse_error"):
        print("  Warning: could not parse structured data from the page.")
        print(f"  Raw response saved to {snapshot_path(url)}")
        return

    if previous is None:
        print("  First run — baseline snapshot saved.\n")
        print("  Extracted intelligence:")
        for key, val in current.items():
            if key not in {"url", "scraped_at", "raw_response", "parse_error"}:
                display = json.dumps(val, ensure_ascii=False)
                print(f"    {key}: {display[:120]}{'…' if len(display) > 120 else ''}")
        print("\n  Run again later to detect what changed.")
        return

    print("  Comparing with previous snapshot ...", end="", flush=True)
    report = compare_snapshots(client, previous, current)
    print(" done.\n")
    print(report)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track competitor websites for positioning and product changes."
    )
    parser.add_argument(
        "--url",
        help="Track a single competitor URL instead of reading competitors.json",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("  export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("=" * 60)
    print("  Competitor Intelligence Tracker")
    print("=" * 60)

    urls = [args.url] if args.url else load_competitors()
    if not urls:
        print("No competitors configured. Add URLs to competitors.json and re-run.")
        sys.exit(0)

    print(f"Tracking {len(urls)} competitor(s)…")

    for url in urls:
        try:
            track_competitor(client, url)
        except anthropic.AuthenticationError:
            print("\nAuthentication failed — check your ANTHROPIC_API_KEY.")
            sys.exit(1)
        except anthropic.RateLimitError:
            print(f"\nRate limited while processing {url}. Wait a moment and retry.")
        except anthropic.APIError as e:
            print(f"\nAPI error for {url}: {e}")
        except Exception as e:
            print(f"\nUnexpected error for {url}: {e}")

    print(f"\n{'=' * 60}")
    print("Done. Snapshots saved to ./snapshots/")
    print("Schedule this script (e.g. daily cron) to get automatic change alerts.")


if __name__ == "__main__":
    main()
