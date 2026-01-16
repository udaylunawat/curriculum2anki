# AAIC → Anki Flashcard Generator

A production-grade pipeline that converts Applied AI Course (AAIC) markdown notes into high-quality Anki flashcards using free OpenRouter LLMs.

The system focuses on recall efficiency, strict formatting, and curriculum-aligned tagging.  
No fluff. No over-generation. No broken cards.

## What this does

- Converts AAIC markdown notes into Anki-importable CSVs
- Enforces strict flashcard structure and quality rules
- Uses only free-tier OpenRouter models
- Aligns every card with AAIC curriculum metadata

If a card is generated, it is reviewable in under 10 seconds.

## Directory structure

flashcards/
- README.md
- story.md
- curriculum_extracted.csv
- markdowns/            Input AAIC markdown files
- output/               Generated CSVs
- prompts/              Versioned prompt files
- validators/           Quality gates and hard checks
- telemetry/            Model health and failure stats
- docs/
  - AAIC_DECK_STANDARD.md
- main.py               Stable production pipeline
- main1.py … main8.py   Evolutionary stages
- test.py               Free model probing

## How to run

Install dependencies:

pip install -r requirements.txt

Set your OpenRouter API key:

export OPENROUTER_API_KEY=your_key

Run the stable pipeline:

python main.py

CSV files will be generated in the output/ directory.  
Import them directly into Anki.

## Design constraints

- Prefer under-generation over noisy cards
- Fail silently instead of polluting the deck
- Cards must stand alone
- Old decks are immutable
- Prompt changes require version bumps

gitingest \
 --exclude-pattern "archived/**" \
 --exclude-pattern "output/**" \
 --exclude-pattern "markdowns/**" \
 --exclude-pattern ".env" \
 --output flashcards_aaic.txt