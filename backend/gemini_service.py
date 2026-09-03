import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=api_key)

PROMPT = """
You are EcoLens, an AI waste identification and disposal assistant.

Analyze the uploaded image. It may contain ONE or SEVERAL distinct waste
items, possibly mixed together in a pile. Identify EACH distinct item you
can see.

Return ONLY valid JSON. Do not use markdown or code fences.

Use exactly this structure:

{
  "items": [
    {
      "item": "name of the item",
      "material": "main material",
      "bin": "wet, dry, recyclable, hazardous, or ewaste",
      "category": "waste category",
      "recyclable": true,
      "condition": "condition or contamination",
      "tip": "short practical disposal/preparation instruction",
      "confidence": 95
    }
  ]
}

You MUST actively consider all five bins below before deciding each item's bin.
Do not default to "wet" or "dry" just because an item isn't food or paper —
check it against "recyclable", "hazardous", and "ewaste" first.

Examples of each bin (use these as reference points, not a full list):
- wet: banana peel, leftover food, tea bags, vegetable scraps, eggshells
- dry: used tissues, chip wrappers with food residue, cigarette butts, dirty napkins
- recyclable: clean plastic bottles, glass jars, aluminum cans, cardboard, clean paper, clean metal tins
- hazardous: batteries (any kind), paint cans, pesticide containers, syringes/needles, broken glass, chemical bottles, expired medicine
- ewaste: phones, chargers, cables, remote controls, headphones, light bulbs, small appliances, circuit boards

Rules for each item:
- "bin" must be exactly one of: wet, dry, recyclable, hazardous, ewaste
- If the item is electronic or battery-powered in any way, it is "ewaste", never "dry" or "wet".
- If the item is a battery, chemical, sharp object, or medical waste, it is "hazardous", never "dry" or "wet".
- If the item is clean and made of plastic, glass, metal, or paper/cardboard, it is "recyclable".
- Only use "wet" or "dry" for genuinely non-recyclable, non-hazardous, non-electronic waste.
- confidence must be a number from 0 to 100
- Keep the tip short and practical
- Be environmentally responsible
- If uncertain, choose the safest reasonable category

Rules for listing items:
- List every distinct item you can identify separately. Don't merge different items into one entry.
- If the same type of item appears more than once (e.g. two bottles), list it once and mention the count in "item", e.g. "2 plastic bottles".
- If you genuinely can't distinguish separate items in a pile, return a single entry describing the whole pile.
"""

# Keyword safety net: if Gemini still mislabels an obviously electronic or
# hazardous item as wet/dry, this catches it and corrects the bin so a demo
# never shows something clearly wrong on stage.
EWASTE_KEYWORDS = [
    "phone", "charger", "cable", "battery pack", "remote", "headphone",
    "earphone", "laptop", "adapter", "circuit", "electronic", "led bulb",
    "bulb", "wire", "appliance", "computer", "mouse", "keyboard", "router",
]
HAZARDOUS_KEYWORDS = [
    "battery", "paint", "pesticide", "syringe", "needle", "chemical",
    "medicine", "medication", "broken glass", "bleach", "acid", "aerosol",
    "thermometer", "sharp",
]
RECYCLABLE_KEYWORDS = [
    "plastic bottle", "glass jar", "aluminum can", "tin can", "cardboard",
    "newspaper", "paper", "carton", "glass bottle", "metal can",
]


def _apply_keyword_override(entry: dict) -> dict:
    text = f"{entry.get('item', '')} {entry.get('material', '')}".lower()

    if any(kw in text for kw in EWASTE_KEYWORDS):
        entry["bin"] = "ewaste"
    elif any(kw in text for kw in HAZARDOUS_KEYWORDS):
        entry["bin"] = "hazardous"
    elif entry.get("bin") in ("wet", "dry") and any(kw in text for kw in RECYCLABLE_KEYWORDS):
        entry["bin"] = "recyclable"

    return entry


def _fill_defaults(entry: dict) -> dict:
    entry.setdefault("item", "Unknown item")
    entry.setdefault("material", "Unknown")
    entry.setdefault("bin", "dry")
    entry.setdefault("category", entry.get("bin", "dry"))
    entry.setdefault("recyclable", entry.get("bin") == "recyclable")
    entry.setdefault("condition", "Unknown")
    entry.setdefault("tip", "No specific guidance available.")
    entry.setdefault("confidence", 60)
    return entry


def analyze_waste(image_bytes: bytes, mime_type: str):
    """
    Sends an image to Gemini and returns:
    { "items": [ {item, material, bin, category, recyclable, condition, tip, confidence}, ... ] }
    """
    response = client.models.generate_content(
        model="gemini-3.7-flash",
        contents=[
            PROMPT,
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_bytes,
                }
            },
        ],
    )

    text = response.text.strip()

    # Remove accidental markdown code fences if Gemini adds them
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(text)

    items = parsed.get("items")
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError(f"No items found in Gemini response: {text}")

    items = [_apply_keyword_override(_fill_defaults(entry)) for entry in items]

    return {"items": items}