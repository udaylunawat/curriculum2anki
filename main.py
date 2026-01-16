import os
import csv
import re
import json
import asyncio
import logging
import time
import random
import argparse
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from validators.csv_validator import validate_csv
from validators.latex_validator import check_latex

# ======================================================
# CONFIG
# ======================================================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

PROMPT_FILE = "prompts/anki_flashcard_prompt_13.txt"
PROMPT_VERSION = "13_6"
CURRICULUM_FILE = "curriculum_extracted.csv"
WORKING_MODELS_FILE = "telemetry/working_models.json"
OUTPUT_DIR = "output"

MAX_TOKENS = 8000
SAFE_CHAR_LIMIT = 12000

LECTURES_PER_BATCH = 3

GLOBAL_RPM = 20
PROVIDER_RPM = 6
RETRY_JITTER = (0.3, 3.0)

NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"{OUTPUT_DIR}/anki_cards_{NOW}.csv"
RESUME_FILE = f"{OUTPUT_CSV}.resume.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("run.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("anki")

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

def tlog(msg):
    tqdm.write(msg)

# ======================================================
# CLI
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument("--file", help="Markdown file to process (optional)")
args = parser.parse_args()

# ======================================================
# CLIENT
# ======================================================
client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=BASE_URL,
)

# ======================================================
# TEXT NORMALIZATION
# ======================================================
STOPWORDS = {
    "and", "or", "of", "for", "with", "to", "in",
    "intuition", "example", "examples", "introduction"
}

def tokenize(text: str):
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = []
    for w in words:
        if w in STOPWORDS:
            continue
        if w.endswith("s") and len(w) > 3:
            w = w[:-1]
        tokens.append(w)
    return set(tokens)

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

# ======================================================
# CURRICULUM
# ======================================================
def load_curriculum():
    rows = []
    with open(CURRICULUM_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "module": row["Module"].strip(),
                "chapter": row["Chapter"].strip(),
                "lecture_title": row["Lecture Title"].strip(),
                "tokens": tokenize(row["Lecture Title"]),
            })
    return rows

def resolve_lecture(title, curriculum):
    title_tokens = tokenize(title)

    best, best_score = None, 0
    for row in curriculum:
        score = len(title_tokens & row["tokens"])
        if score > best_score:
            best, best_score = row, score

    if best_score >= 2:
        return best
    if best_score == 1 and len(title_tokens) == 1 and len(best["tokens"]) == 1:
        return best
    return None

# ======================================================
# MARKDOWN PARSING
# ======================================================
LECTURE_RE = re.compile(
    r"""^\s*(\*\*)?\s*\d+\.\d+\s+(.+?)\s*(\*\*)?$""",
    re.IGNORECASE | re.VERBOSE,
)

def extract_lectures(md_text):
    blocks, current, buffer = [], None, []
    for line in md_text.splitlines():
        m = LECTURE_RE.match(line)
        if m:
            if current:
                blocks.append(current | {"content": "\n".join(buffer)})
            current, buffer = {"title": m.group(2).strip()}, []
        else:
            buffer.append(line)
    if current:
        blocks.append(current | {"content": "\n".join(buffer)})
    return blocks

# ======================================================
# MODELS
# ======================================================
def load_models():
    data = json.load(open(WORKING_MODELS_FILE))
    return [(p, ms) for p, ms in data.items() if ms]

# ======================================================
# JSON PARSER
# ======================================================
def extract_json(text):
    text = re.sub(r"```json|```", "", text, flags=re.I)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None

# ======================================================
# BATCHING
# ======================================================
def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

# ======================================================
# RESUME STATE
# ======================================================
def load_resume_state():
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "r") as f:
            return json.load(f)
    return {}

def save_resume_state(state):
    with open(RESUME_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ======================================================
# MAIN
# ======================================================
async def main():
    resume_state = load_resume_state()

    # ✅ FIX: respect --file
    if args.file:
        md_files = [args.file]
    else:
        md_files = [
            os.path.join("markdowns", f)
            for f in sorted(os.listdir("markdowns"))
            if f.lower().endswith(".md")
        ]

    curriculum = load_curriculum()
    providers = load_models()
    base_prompt = open(PROMPT_FILE, encoding="utf-8").read()

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)

        async def run_llm(prompt, chapter, batch_idx):
            for provider, models in providers:
                for model in models:
                    try:
                        resp = await client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            max_tokens=MAX_TOKENS,
                        )
                        data = extract_json(resp.choices[0].message.content)
                        if data and data.get("cards"):
                            tlog(f"[MODEL OK] {provider} | {model} | {chapter} | batch {batch_idx}")
                            return data["cards"]
                    except Exception as e:
                        tlog(f"[MODEL FAIL] {provider} | {model} | {type(e).__name__}")
                        await asyncio.sleep(random.uniform(*RETRY_JITTER))
            tlog(f"[LLM FAILURE] {chapter} | batch {batch_idx}")
            return []

        for md in md_files:
            lectures = extract_lectures(open(md, encoding="utf-8").read())

            resolved = []
            for b in lectures:
                meta = resolve_lecture(b["title"], curriculum)
                if not meta:
                    tlog(f"[UNRESOLVED] {b['title']}")
                    continue
                resolved.append((b, meta))

            by_chapter = defaultdict(list)
            for b, meta in resolved:
                by_chapter[meta["chapter"]].append((b, meta))

            for chapter, items in by_chapter.items():
                module = items[0][1]["module"]
                batches = list(chunked(items, LECTURES_PER_BATCH))

                last_done = resume_state.get(chapter, 0)
                total = len(batches)

                tlog(f"[CHAPTER START] {chapter} | {total} batches | resuming at batch {last_done + 1}")

                for idx, batch in enumerate(tqdm(batches, desc=chapter, unit="batch"), start=1):
                    if idx <= last_done:
                        continue

                    titles = [b["title"] for b, _ in batch]
                    tlog(f"[BATCH START] {chapter} | batch {idx} | {titles}")

                    combined = "\n\n".join(
                        f"### {b['title']}\n{b['content']}" for b, _ in batch
                    )

                    prompt = f"""
{base_prompt}

MODULE: {module}
CHAPTER: {chapter}

--- MARKDOWN START ---
{combined}
--- MARKDOWN END ---
"""
                    cards = await run_llm(prompt, chapter, idx)

                    for c in cards:
                        writer.writerow([
                            c["front"],
                            c["back"],
                            f"aaic::{slugify(module)}::{slugify(chapter)}"
                        ])
                        out.flush()

                    resume_state[chapter] = idx
                    save_resume_state(resume_state)

                    tlog(f"[BATCH DONE] {chapter} | batch {idx} | {len(cards)} cards")

                tlog(f"[CHAPTER DONE] {chapter}")

    validate_csv(OUTPUT_CSV)
    check_latex(OUTPUT_CSV)
    tlog(f"[DONE] Output → {OUTPUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())