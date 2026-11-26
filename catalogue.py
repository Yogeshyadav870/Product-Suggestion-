"""Product catalogue and search engine for the AAHVANI Product Suggestion Dashboard."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CATALOGUE_PATH = Path(__file__).parent / "data" / "product_catalogue.json"


class ProductCatalogue:
    """Curated demo catalogue. All prices and listings are illustrative, not live offers."""

    ALIASES = {
        "smartphone": ("smartphone", "phone", "mobile", "iphone", "android"),
        "laptop": ("laptop", "notebook", "computer", "gaming laptop"),
        "headphones": ("headphones", "headphone", "earbuds", "earphones", "headset"),
        "tablet": ("tablet", "ipad"),
        "smartwatch": ("smartwatch", "smart watch", "fitness band", "watch"),
        "speaker": ("speaker", "bluetooth speaker", "soundbar"),
        "television": ("television", "smart tv", "led tv", "tv"),
        "footwear": ("footwear", "running shoes", "sports shoes", "sneakers", "shoes", "sandals"),
        "backpack": ("backpack", "laptop bag", "travel bag", "school bag", "rucksack", "bag"),
    }

    BRANDS = (
        "Samsung", "Apple", "Xiaomi", "Redmi", "OnePlus", "Motorola", "Nothing",
        "Realme", "Vivo", "Oppo", "Poco", "Google", "Asus", "Acer", "Lenovo",
        "HP", "Dell", "MSI", "JBL", "Sony", "boAt", "Noise", "Amazfit",
        "Nike", "Adidas", "Puma", "Wildcraft", "Skybags",
    )

    USE_CASES = {
        "gaming": ("gaming", "game", "games", "gamer", "gpu", "fps"),
        "coding": ("coding", "programming", "developer", "development", "software"),
        "study": ("study", "student", "college", "classes", "online class", "notes"),
        "work": ("work", "office", "professional", "productivity", "business"),
        "camera": ("camera", "photography", "photos", "photo", "selfie"),
        "battery": ("battery", "long battery", "backup", "battery life"),
        "performance": ("performance", "fast", "multitasking", "processor", "speed"),
        "music": ("music", "songs", "audio", "bass"),
        "calls": ("calls", "calling", "meetings", "microphone", "voice"),
        "fitness": ("fitness", "health", "workout", "running", "steps"),
        "content": ("editing", "video editing", "design", "creative", "content creation"),
        "movies": ("movies", "streaming", "watch videos", "entertainment"),
        "travel": ("travel", "portraits", "landscape"),
        "party": ("party", "parties", "gathering", "loud music"),
        "casual": ("casual", "daily wear", "everyday wear"),
        "sports": ("sports", "running", "walking", "training", "gym"),
        "comfort": ("comfort", "ergonomic", "back support", "posture"),
        "everyday": ("daily", "everyday", "general use", "all round", "all-round", "normal use"),
    }

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else CATALOGUE_PATH
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("Product catalogue must contain at least one item.")
        self.products = raw
        self.by_id = {item["id"]: item for item in raw}

    @classmethod
    def detect_category(cls, text: str) -> str:
        lower = (text or "").lower()
        aliases = sorted(
            ((alias, category) for category, values in cls.ALIASES.items() for alias in values),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        matches = []
        for alias, category in aliases:
            match = re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", lower)
            if match:
                matches.append((match.start(), -len(alias), category))
        return min(matches)[2] if matches else ""

    @staticmethod
    def parse_budget(text: str) -> int | None:
        raw = (text or "").lower().replace(",", "").replace("₹", " rs ")
        patterns = (
            r"\b(?:under|below|within|max(?:imum)?|budget(?:\s+of)?|upto|up\s+to|less\s+than|ke\s+andar|tak)\s*(?:rs\.?\s*)?([\d]+(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|thousand)?\b",
            r"\b(?:rs\.?\s*)([\d]+(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|thousand)?\b",
            r"\b([\d]+(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|thousand)\b",
        )
        multipliers = {
            "k": 1_000, "thousand": 1_000, "lakh": 100_000, "lac": 100_000,
            "cr": 10_000_000, "crore": 10_000_000,
        }
        for pattern in patterns:
            match = re.search(pattern, raw, re.I)
            if match:
                amount = float(match.group(1))
                unit = (match.group(2) or "").lower()
                value = int(round(amount * multipliers.get(unit, 1)))
                if value >= 1_000:
                    return value
        bare = re.search(r"\b(?:budget\s*(?:is|of)?\s*)?(\d{4,8})\b", raw)
        if bare:
            return int(bare.group(1))
        return None

    @classmethod
    def detect_brand(cls, text: str) -> str:
        lower = (text or "").lower()
        for brand in cls.BRANDS:
            if re.search(r"(?<!\w)" + re.escape(brand.lower()) + r"(?!\w)", lower):
                return brand
        if re.search(r"\b(any brand|no preference|any company|all brands|you decide|doesn't matter)\b", lower):
            return "Any"
        return ""

    @classmethod
    def detect_use_case(cls, text: str) -> str:
        lower = (text or "").lower()
        for use_case, markers in cls.USE_CASES.items():
            for marker in markers:
                if re.search(r"(?<!\w)" + re.escape(marker) + r"(?!\w)", lower):
                    return use_case
        if re.search(r"\b(no preference|anything|all round|all-round|general use)\b", lower):
            return "everyday"
        return ""

    def search(
        self,
        *,
        category: str,
        budget_max: int | None = None,
        brand: str = "",
        use_case: str = "",
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        candidates = [p for p in self.products if p["category"] == category]
        if budget_max is not None:
            candidates = [p for p in candidates if int(p["price_inr"]) <= int(budget_max)]
        if brand and brand.lower() != "any":
            candidates = [p for p in candidates if p["brand"].lower() == brand.lower()]

        def score(product: dict[str, Any]) -> tuple[int, int]:
            overlap = 4 if use_case and use_case in product.get("use_cases", []) else 0
            words = re.findall(r"[a-z0-9]+", (use_case or "").lower())
            text = " ".join([product["title"], product["brand"], *product.get("tags", [])]).lower()
            return overlap + sum(token in text for token in words), -int(product["price_inr"])

        return sorted(candidates, key=score, reverse=True)[:max(1, min(limit, 3))]

    def format_recommendations(self, products: list[dict[str, Any]]) -> str:
        if not products:
            return "I couldn't find a match in the current demo catalogue. Try increasing the budget or changing the brand."
        cards = []
        for index, product in enumerate(products, start=1):
            specs = " · ".join(product.get("specs", [])[:4])
            cards.append(
                f"### {product.get('icon', '🛍️')} Option {index} — {product['title']}\n"
                f"**{product['brand']} · ₹{int(product['price_inr']):,}**\n\n"
                f"{specs}\n\n*Why it fits:* {product.get('recommendation_reason', 'Matches your requirements.')}"
            )
        return (
            f"## I found {len(products)} matching options\n\n"
            + "\n\n---\n\n".join(cards)
            + "\n\nListings and prices are illustrative demo data, not live store offers."
        )
