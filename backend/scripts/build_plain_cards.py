"""Pre-generates plain-English executive cards for every corpus entry.

Output: app/data/plain_cards.json  {entry_id: {insight, steps, how_it_works, caveat, model}}
Used when the live explainer is off (no key) or fails, so the card is never jargon-heavy.

Run from backend/:  GEMINI_API_KEY=... python scripts/build_plain_cards.py [--only-missing]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import explainer  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "app" / "data"


async def main() -> None:
    corpus = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
    out_path = DATA / "plain_cards.json"
    cards = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    only_missing = "--only-missing" in sys.argv
    if explainer.provider() is None:
        sys.exit("Set GEMINI_API_KEY (or ANTHROPIC_API_KEY) first.")
    for i, doc in enumerate(corpus, 1):
        if only_missing and doc["id"] in cards:
            continue
        problem = (
            "A team has this engineering/business problem, which this research has been applied to: "
            + doc.get("abstract", doc["summary"])[:900]
        )
        card = None
        for attempt in range(4):
            card = await explainer.explain(problem, doc)
            if card:
                break
            await asyncio.sleep(10 * (attempt + 1))  # free-tier rate limits
        if card:
            cards[doc["id"]] = card
            out_path.write_text(json.dumps(cards, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{i}/{len(corpus)}] {doc['id']}: {'ok' if card else 'FAILED'}", flush=True)
        await asyncio.sleep(2.5)


if __name__ == "__main__":
    asyncio.run(main())
