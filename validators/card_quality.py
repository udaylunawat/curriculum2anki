# validators/card_quality.py
import re

MAX_FRONT_WORDS = 12
MAX_BACK_CHARS = 600

def is_valid_card(front: str, back: str) -> bool:
    # 1. Front length
    if len(front.split()) > MAX_FRONT_WORDS:
        return False

    # 2. ELI5 presence
    if "ELI5:" not in back:
        return False

    # 3. Formula without explanation
    if "[$ $" in back:
        parts = back.split("[/$ $]")
        if len(parts) < 2 or len(parts[1].strip()) < 20:
            return False

    # 4. Back length
    if len(back) > MAX_BACK_CHARS:
        return False

    # 5. Front repeated in back
    if front.lower() in back.lower():
        return False

    return True