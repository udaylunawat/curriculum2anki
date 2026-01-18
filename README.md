
# Curriculum2Anki

**A Knowledge Distillation Engine for Turning Long-Form Text into High-Signal Anki Decks**

Curriculum2Anki is a production-grade pipeline that converts **large, unstructured or semi-structured text documents** (courses, books, notes, PDFs converted to markdown) into **high-quality Anki flashcards** using LLMs.

It is designed around one principle:

> **If a card cannot be recalled confidently in under 10 seconds, it does not belong.**

This is not a toy flashcard generator.
It is a **knowledge distillation system** built to survive messy input, unreliable models, long documents (100k+ tokens), and real-world constraints like rate limits, interruptions, and cost ceilings.

---

## Why this exists

Most “AI flashcard” tools fail in predictable ways:

* Over-generate low-signal cards
* Lose curriculum or document structure
* Hallucinate relationships
* Break math rendering
* Cannot resume after failure
* Offer no observability or quality gates

Curriculum2Anki was built to solve those problems *systematically*.

In one overnight run, it generated **900+ ultra high-quality cards from a ~100k token document for ~$0.20**, using **Gemini 2.5 Flash (Thinking)** via OpenRouter.

---

## What this does

* Converts **long text documents** into Anki-importable CSV decks
* Works with:
    * Course notes
    * Technical books
    * Interview prep material
    * Research summaries
    * Any markdown-converted text
* Enforces **strict card quality rules**
* Aligns cards to **semantic structure**, not brittle numbering
* Uses **free / low-cost LLMs** with automatic fallback
* Supports **resume-after-interruption** at batch level
* Produces **deterministic, review-ready output**

If a card is generated, it is:
* Self-contained
* Strictly formatted
* Validated for LaTeX and structure
* Worth reviewing

**Silence is preferred over noise.**

## Sample Card
![alt text](sample.jpeg)

---

## Core design ideas

### 1. Knowledge distillation, not summarization
The system extracts **concepts**, then distills them into **atomic recall units**. Coverage is deliberately capped. Precision is prioritized.

### 2. Defensive parsing
Input documents are assumed to be inconsistent, poorly structured, and noisy.

The pipeline:
* Ignores formatting noise
* Uses robust header detection
* Falls back to semantic matching
* Refuses to guess when alignment is ambiguous

### 3. Curriculum / structure as source of truth
A lightweight CSV acts as the **authoritative schema** (Module, Chapter, Concept titles). Document structure is *mapped* to this schema, not trusted blindly.

This makes the system portable across courses, books, internal documentation, and knowledge bases.

### 4. Quality gates are non-negotiable
Cards are dropped if they fail checks such as:
* Front length > 12 words
* Missing ELI5 explanation
* Broken or unbalanced LaTeX
* Duplicate concepts
* Explanation without substance

Bad cards permanently damage decks. The system prefers to generate nothing rather than generate garbage.

### 5. Observability and resumability
* Batch-level progress tracking
* Per-chapter progress bars and ETA
* Model success/failure logging
* Resume state persisted to disk
* Append-only CSV output

This is built for **long-running jobs**, not demos.

---

## Directory structure

```text
flashcards/
├── README.md                  This document
├── STORY.md                   Design history and evolution
├── curriculum_extracted.csv   Document / curriculum schema
├── markdowns/                 Input text (markdown)
├── output/                    Generated decks + resume state
├── prompts/                   Versioned LLM prompts
├── validators/                Hard quality gates
├── telemetry/                 Model probing and health stats
├── docs/
│   └── DECK_STANDARD.md       Deck invariants and rules
├── main.py                    Stable production pipeline
├── main1.py … main8.py        Evolutionary stages
└── test.py                    Free-model probing harness

```

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt

```

### 2. Set API key

```bash
export OPENROUTER_API_KEY=your_key

```

### 3. Run on all documents

```bash
python main.py

```

### 4. Run on a single document

```bash
python main.py --file markdowns/your_doc.md

```

### Resume after interruption

Just rerun the same command. Progress is restored automatically.

> Generated CSVs appear in `output/` and can be imported directly into Anki.

---

## Models

* Designed to work with free and low-cost models
* Actively tested and rotated via `test.py`
* Current best results observed with: **Gemini 2.5 Flash (Thinking)**

Model unreliability is treated as a given, not an exception.

---

## What makes this production-grade

* Rate-limit aware
* Provider-aware fallback
* JSON repair logic
* Idempotent batch processing
* Deterministic output
* Resume safety
* Explicit failure modes
* No silent corruption

This codebase reflects real failure conditions, not ideal ones.

---

## How this generalizes beyond courses

Curriculum2Anki is intentionally not domain-specific. With minimal changes, it can power:

* Book-to-Anki pipelines
* Internal knowledge base distillation
* Interview prep generators
* Onboarding documentation agents
* Personal knowledge graphs

The core abstraction is: **Text → Concepts → Validated Recall Units**

---

## How this maps to real-world AI systems

This project touches multiple production-level concerns relevant to modern AI tooling:

* Information retrieval and matching
* Long-context handling
* Cost-aware LLM orchestration
* Agent-like task decomposition
* Reliability under partial failure
* Deterministic post-processing
* Human-in-the-loop quality enforcement

It is closer to a workflow automation engine using LLM agents than a simple NLP script.

---

## Philosophy

* Prefer under-generation to noise
* Prefer determinism to cleverness
* Prefer correctness to coverage
* Prefer systems that explain themselves

1. If a card is hard to review, delete it.
2. If alignment is unclear, stop.
3. If a model misbehaves, route around it.

---

## Status

* Actively evolving
* Built rapidly, but deliberately
* Designed to be extended into a SaaS or internal tool
* Optimized for correctness, cost, and recall efficiency

> **If you’re reading this as a reviewer or interviewer:**
> This repository was built to demonstrate systems thinking around AI agents and tooling, not just prompt engineering.
> The hard parts here are not the LLM calls.
> They are everything around them.

```
