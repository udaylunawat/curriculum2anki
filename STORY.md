# Story: How This Pipeline Evolved

This project did not begin as a “flashcard generator”.

It began as repeated failure.

AAIC content is long, inconsistent, and deceptively dense.  
Reading worked once. Revising did not.

Anki was the obvious answer.  
LLMs seemed like the shortcut.

They were not.

## The naive attempt

The first version simply fed markdown into an LLM and asked for flashcards.

The result was unusable:

- Front sides were verbose
- Explanations rambled
- Math rendering broke
- Cards overlapped heavily
- Review felt slow and frustrating

If recall takes more than 10 seconds, the card is wrong.  
Most cards failed that test.

## Markdown reality

AAIC markdown is not structured data.

Problems encountered:

- Missing or inconsistent headers
- Multiple lectures in one file
- Tables, images, and code blocks polluting context
- Broken numbering and partial chapters

Prompting harder did nothing.

The solution was defensive parsing:

- Header-based chunking with fallbacks
- Aggressive truncation
- Ignoring formatting noise
- Extracting existing questions when present
- Refusing to guess structure when ambiguous

This reduced hallucinations more than any prompt tweak.

## Free LLMs are unreliable by default

Using free OpenRouter models exposed real-world issues:

- Models disappearing without notice
- Undocumented rate limits
- Providers returning malformed JSON
- Outputs breaking mid-response
- Instruction-following drifting daily

The pipeline adapted by necessity:

- Probing all free models via test.py
- Tracking failures and success rates
- Caching only proven models
- Rotating providers
- Adding exponential backoff and cooldowns
- Repairing JSON via brace counting

Telemetry became mandatory.  
Trust became earned.

## Quality gates became non-negotiable

Bad cards permanently damage an Anki deck.

So the system learned to say no.

Cards are dropped if:

- Front text exceeds limits
- ELI5 explanation is missing
- LaTeX blocks are invalid
- Formula explanations are absent
- Duplicates are detected
- Length constraints are violated

Silence is better than noise.

No partial garbage output is allowed.

## Curriculum alignment changed everything

Early decks had good cards in the wrong order.

The fix was making the curriculum the source of truth:

- curriculum_extracted.csv drives tagging
- Lecture and module inference is mandatory
- Title matching uses token overlap thresholds
- Generation is refused if alignment is ambiguous

This transformed decks from short-term cramming tools into long-term assets.

## Why there are many main files

Each main*.py exists for a reason.

They document real lessons:

- main.py established the baseline
- early variants handled survival fixes
- later versions introduced telemetry-driven routing
- concept-first extraction reduced noise
- provider-aware rate limiting handled hidden quotas
- final versions aligned everything to curriculum boundaries

Deleting these would lie about the difficulty.

## The final rule

This system does not optimize for coverage.

It optimizes for recall.

If a card cannot be answered quickly and confidently, it does not belong.

That rule shaped every design decision.