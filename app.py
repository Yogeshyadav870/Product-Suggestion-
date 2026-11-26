"""Main application — AAHVANI AI Product Suggestion Dashboard.

A conversational product recommendation engine with Gemini AI and
a local demo catalogue fallback. Styled to match the AAHVANI AI theme.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
    GEMINI_IMPORT_ERROR = ""
except ImportError as _err:
    genai = None
    genai_types = None
    GEMINI_AVAILABLE = False
    GEMINI_IMPORT_ERROR = str(_err)

from catalogue import ProductCatalogue

_GEMINI_CLIENT = None
APP_NAME = "AAHVANI AI — Product Suggestion"
catalogue = ProductCatalogue()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def initial_state() -> dict:
    return {
        "session_id": str(uuid.uuid4()),
        "shopping": {
            "category": "",
            "budget_max": None,
            "brand": "",
            "use_case": "",
            "recommendations": [],
        },
        "gemini_previous_interaction_id": None,
    }


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _get_gemini_client():
    global _GEMINI_CLIENT
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from the local .env file.")
    if not GEMINI_AVAILABLE:
        raise RuntimeError(f"Gemini dependencies unavailable: {GEMINI_IMPORT_ERROR}")
    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                timeout=45_000,
                retry_options=genai_types.HttpRetryOptions(attempts=0),
            ),
        )
    return _GEMINI_CLIENT


def _gemini_model_name() -> str:
    configured = os.getenv("GEMINI_MODEL", "").strip()
    legacy = {"gemini-2.5-flash", "models/gemini-2.5-flash", "gemini-2.5-flash-preview"}
    if not configured or configured in legacy:
        return "gemini-3.8-flash"
    return configured.removeprefix("models/")


def _interaction_text(interaction) -> str:
    text = getattr(interaction, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    parts = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for block in getattr(step, "content", None) or []:
            if getattr(block, "type", None) == "text":
                t = getattr(block, "text", "")
                if t:
                    parts.append(t)
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Gemini-powered product suggestion
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """
You are AAHVANI AI, a friendly product suggestion assistant for Indian customers.
Use natural English, Hindi or Hinglish to match the customer's language. Be warm, conversational,
and ask only one relevant follow-up question at a time.

PRODUCT SUGGESTION FLOW
Help customers find the right product from the AAHVANI illustrative demo catalogue. Categories include
smartphones, laptops, headphones, tablets, smartwatches, speakers, televisions, footwear and backpacks.
Remember category, maximum budget, brand and primary use across all turns.
Never ask again for a detail the customer already supplied.
A customer may say phone, mobile or smartphone for the smartphone category.
For smartphones, ask what matters most: camera, gaming performance, battery life or balanced everyday use.
For laptops, ask about gaming, coding, study, office work or editing.
Ask one relevant category-specific question for other products.
If budget is missing, ask for it. If use-case is missing, ask about it.
If brand preference is missing, ask whether the customer has a preferred brand or wants all brands compared.
Treat "no preference", "any brand" or "all brands" as brand="Any".

CATALOGUE
Use search_products only after category, budget, brand preference and primary use are known.
Only recommend products returned by search_products. Preserve their exact product names, brands and INR prices.
Never invent specifications, availability, discounts or prices.
State that listings and prices are illustrative demo catalogue data, not live retailer stock.
The application updates the recommendation panel from this tool.

SELECTION
When the customer picks an option, acknowledge their choice and provide a brief summary.
""".strip()


def gemini_suggest(message: str, session: dict) -> str:
    """Run a stateful Gemini Interactions conversation with catalogue search tool."""
    session = session or initial_state()
    shopping = session.setdefault("shopping", initial_state()["shopping"])
    session["_tool_events"] = []

    # Parse facts from user message
    requested_category = catalogue.detect_category(message)
    if requested_category and requested_category != shopping.get("category"):
        shopping.update({
            "category": requested_category,
            "budget_max": None,
            "brand": "",
            "use_case": "",
            "recommendations": [],
        })

    parsed_budget = catalogue.parse_budget(message)
    if parsed_budget:
        shopping["budget_max"] = parsed_budget
    parsed_brand = catalogue.detect_brand(message)
    if parsed_brand:
        shopping["brand"] = parsed_brand
    parsed_use_case = catalogue.detect_use_case(message)
    if parsed_use_case:
        shopping["use_case"] = parsed_use_case

    # Tool: search_products
    def search_products(category: str, budget_max: int = 0, brand: str = "Any",
                        primary_use: str = "everyday") -> dict:
        detected = catalogue.detect_category(category) or str(category or "").strip().lower()
        selected_category = shopping.get("category") or detected
        budget = int(shopping.get("budget_max") or 0)

        if not selected_category:
            return {"ok": False, "needs_clarification": True, "message": "Ask which product category the customer wants."}
        if budget <= 0:
            return {"ok": False, "needs_clarification": True, "message": "Ask for the customer's maximum budget before searching."}

        chosen_brand = shopping.get("brand") or ""
        if not chosen_brand:
            return {"ok": False, "needs_clarification": True, "message": "Ask whether the customer has a preferred brand or wants all brands compared."}
        chosen_use = shopping.get("use_case") or ""
        if not chosen_use:
            return {"ok": False, "needs_clarification": True, "message": "Ask what matters most for this category before searching."}

        products = catalogue.search(
            category=selected_category, budget_max=budget,
            brand=chosen_brand, use_case=chosen_use, limit=3,
        )
        brand_note = ""
        if not products and chosen_brand.lower() != "any":
            products = catalogue.search(
                category=selected_category, budget_max=budget,
                use_case=chosen_use, limit=3,
            )
            if products:
                brand_note = f"No {chosen_brand} listing was found within the budget; alternative brands are shown."

        shopping.update({
            "category": selected_category,
            "budget_max": budget,
            "brand": chosen_brand,
            "use_case": chosen_use,
            "recommendations": products,
        })
        session["_tool_events"].append("search_products")
        return {
            "ok": bool(products),
            "count": len(products),
            "brand_note": brand_note,
            "products": [{
                "id": p["id"],
                "title": p["title"],
                "brand": p["brand"],
                "price_inr": int(p["price_inr"]),
                "specs": p.get("specs", []),
                "recommendation_reason": p.get("recommendation_reason", ""),
            } for p in products],
            "note": "Illustrative demo catalogue only; not live retailer stock or pricing.",
        }

    tools = [{
        "type": "function",
        "name": "search_products",
        "description": "Search the AAHVANI demo catalogue after the customer has supplied category, budget, brand preference and primary use.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "The requested category, such as smartphone, laptop or footwear."},
                "budget_max": {"type": "integer", "description": "Maximum budget in whole Indian rupees."},
                "brand": {"type": "string", "description": "Preferred brand, or Any if the customer wants all brands."},
                "primary_use": {"type": "string", "description": "Main use or priority, such as camera, battery, gaming, coding, study, work or everyday."},
            },
            "required": ["category"],
        },
    }]
    available_functions = {"search_products": search_products}

    try:
        client = _get_gemini_client()
        model = _gemini_model_name()
        request_options = {
            "model": model,
            "system_instruction": SYSTEM_INSTRUCTION,
            "tools": tools,
            "generation_config": {"temperature": 0.35},
            "store": True,
            "input": str(message or "").strip(),
        }
        previous_id = session.get("gemini_previous_interaction_id")
        if previous_id:
            request_options["previous_interaction_id"] = previous_id

        interaction = client.interactions.create(**request_options)

        # Execute tool calls
        for _ in range(6):
            function_calls = [
                step for step in (getattr(interaction, "steps", None) or [])
                if getattr(step, "type", None) == "function_call"
            ]
            if not function_calls:
                break

            function_results = []
            for step in function_calls:
                tool = available_functions.get(step.name)
                if tool is None:
                    result = {"ok": False, "error": f"Unsupported tool: {step.name}"}
                else:
                    try:
                        result = tool(**dict(step.arguments or {}))
                    except Exception as exc:
                        result = {"ok": False, "error": f"Tool failed: {type(exc).__name__}: {exc}"}

                result_step = {
                    "type": "function_result",
                    "name": step.name,
                    "call_id": step.id,
                    "result": result,
                }
                signature = getattr(step, "signature", None)
                if signature:
                    result_step["signature"] = signature
                function_results.append(result_step)

            interaction = client.interactions.create(
                model=model,
                previous_interaction_id=interaction.id,
                input=function_results,
                store=True,
            )

        session["gemini_previous_interaction_id"] = getattr(interaction, "id", None)
        answer = _interaction_text(interaction)
        if not answer:
            answer = "I understood your message, but I could not produce a text response. Please try once more."
        session["_tool_events"] = []
        return answer
    except Exception:
        logging.getLogger(__name__).exception("Gemini API request failed")
        raise


# ---------------------------------------------------------------------------
# Local fallback (no Gemini)
# ---------------------------------------------------------------------------

def local_suggest(message: str, session: dict) -> str:
    """Rule-based fallback when Gemini is not available."""
    shopping = session.setdefault("shopping", initial_state()["shopping"])
    text = " ".join((message or "").split()).strip()
    if not text:
        return "Tell me what you are looking to buy, and I will help you find the best option."

    category = catalogue.detect_category(text)
    if category and category != shopping.get("category"):
        shopping.update({
            "category": category, "budget_max": None, "brand": "", "use_case": "",
            "recommendations": [],
        })
    elif category:
        shopping["category"] = category

    budget = catalogue.parse_budget(text)
    if budget:
        shopping["budget_max"] = budget
    brand = catalogue.detect_brand(text)
    if brand:
        shopping["brand"] = brand
    use_case = catalogue.detect_use_case(text)
    if use_case:
        shopping["use_case"] = use_case

    if not shopping.get("category"):
        return (
            "What are you looking to buy? I can suggest smartphones, laptops, headphones, tablets, "
            "smartwatches, speakers, TVs, shoes or backpacks."
        )
    if not shopping.get("budget_max"):
        return f"What is your maximum budget for the {shopping['category']}?"
    if not shopping.get("use_case"):
        questions = {
            "smartphone": "What matters most — camera, gaming performance, battery life or balanced everyday use?",
            "laptop": "What will you mainly use it for — gaming, coding, studying, office work or editing?",
            "headphones": "Will you mainly use them for music, calls/meetings or gaming?",
            "tablet": "Will you mainly use it for studying, entertainment, drawing or work?",
            "smartwatch": "Is your priority fitness tracking, battery life or everyday smart features?",
            "speaker": "Will you mainly use it for music, parties, movies or a desktop setup?",
            "television": "Will you mainly use it for movies and streaming, gaming or everyday TV viewing?",
            "footwear": "Are you looking for running/sports shoes, walking shoes or casual everyday footwear?",
            "backpack": "Will you use it for college, office, travel or carrying a laptop?",
        }
        return questions.get(shopping["category"], "What is the main use you have in mind?")
    if not shopping.get("brand"):
        return "Do you have a preferred brand, or should I compare options from all brands?"

    # All requirements gathered — search
    products = catalogue.search(
        category=shopping["category"], budget_max=shopping["budget_max"],
        brand=shopping["brand"], use_case=shopping["use_case"], limit=3,
    )
    if not products and shopping["brand"] and shopping["brand"] != "Any":
        products = catalogue.search(
            category=shopping["category"], budget_max=shopping["budget_max"],
            use_case=shopping["use_case"], limit=3,
        )
    shopping["recommendations"] = products
    if not products:
        return (
            f"I couldn't find a {shopping['category']} in our demo catalogue under "
            f"₹{shopping['budget_max']:,}. Try increasing the budget or changing the category."
        )
    return catalogue.format_recommendations(products)


# ---------------------------------------------------------------------------
# Unified chat handler
# ---------------------------------------------------------------------------

def product_chat(message, history, session):
    session = session or initial_state()
    session.setdefault("session_id", str(uuid.uuid4()))

    if os.getenv("GEMINI_API_KEY", "").strip() and GEMINI_AVAILABLE:
        try:
            answer = gemini_suggest(message, session)
        except Exception as exc:
            answer = (
                "I'm having trouble reaching Gemini right now. "
                f"Please try sending your message again. Detail: {type(exc).__name__}: {exc}"
            )
    else:
        answer = local_suggest(message, session)

    history = list(history or [])
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ])
    shopping = session.get("shopping", {})
    recommendations = shopping.get("recommendations", [])
    recommendation_panel = (
        catalogue.format_recommendations(recommendations)
        if recommendations else
        "Your recommendations will appear here once AAHVANI understands what you need."
    )
    return history, session, recommendation_panel, ""


# ---------------------------------------------------------------------------
# Custom CSS — AAHVANI warm gold/cream theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
:root, :host, html, body, gradio-app, .gradio-container,
body.dark, .dark, [data-theme="dark"] {
    color-scheme: light !important;
    --page-bg: #dfcfbf !important;
    --card-bg: #fffdf9 !important;
    --inner-bg: #f6eee5 !important;
    --inner-bg-soft: #faf5ef !important;
    --ink: #211a14 !important;
    --muted: #74685c !important;
    --gold: #a97617 !important;
    --gold-deep: #87590f !important;
    --gold-soft: #e8d4ae !important;
    --line: #e5d6c7 !important;
    --body-background-fill: #dfcfbf !important;
    --body-text-color: #211a14 !important;
    --background-fill-primary: #fffdf9 !important;
    --background-fill-secondary: #f6eee5 !important;
    --block-background-fill: #fffdf9 !important;
    --block-border-color: #e5d6c7 !important;
    --block-label-text-color: #6e6052 !important;
    --input-background-fill: #f7f0e8 !important;
    --input-border-color: #e3d3c2 !important;
    --input-placeholder-color: #897b6d !important;
    --button-primary-background-fill: #332316 !important;
    --button-primary-background-fill-hover: #49321f !important;
    --button-primary-text-color: #fffdf9 !important;
    --button-secondary-background-fill: #f5ede3 !important;
    --button-secondary-text-color: #654719 !important;
    --border-color-primary: #e5d6c7 !important;
    --border-color-accent: #d8c6b3 !important;
}

html, body, body.dark {
    min-height: 100%;
    background: #dfcfbf !important;
    color: #211a14 !important;
}
body, body.dark, gradio-app {
    background:
        radial-gradient(ellipse at 12% 8%, rgba(255, 250, 243, .48), transparent 34%),
        radial-gradient(ellipse at 92% 92%, rgba(245, 222, 194, .42), transparent 32%),
        linear-gradient(135deg, #e2d1c0 0%, #ddcbb9 48%, #e6d4c2 100%) fixed !important;
}

.gradio-container, body.dark .gradio-container, gradio-app .gradio-container {
    position: relative !important;
    z-index: 1 !important;
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 28px 30px 54px !important;
    background: transparent !important;
    color: #211a14 !important;
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}

#hero {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 190px;
    padding: 30px 40px;
    margin: 0 0 24px;
    border: 1px solid rgba(255,255,255,.92);
    border-radius: 28px;
    background:
        radial-gradient(ellipse at 80% 100%, rgba(224, 199, 159, .18), transparent 38%),
        linear-gradient(112deg, #fbf8f3 0%, #f8f4ed 56%, #f5efe5 100%) !important;
    box-shadow: 0 14px 34px rgba(91, 65, 38, .10), inset 0 1px 0 #fff;
}
#hero:before {
    content: "";
    position: absolute;
    z-index: -1;
    width: 370px;
    height: 140px;
    right: 4%;
    bottom: -95px;
    border: 1px solid rgba(182, 135, 54, .15);
    border-radius: 50%;
    box-shadow: 0 -16px 0 -15px rgba(182, 135, 54, .12), 0 -32px 0 -31px rgba(182, 135, 54, .10);
    transform: rotate(-10deg);
}
.hero-copy { position: relative; z-index: 1; }
.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 13px;
    color: #936515;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.55px;
    text-transform: uppercase;
}
.hero-dot {
    width: 8px;
    height: 8px;
    display: inline-block;
    border-radius: 50%;
    background: #bf8d31;
    box-shadow: 0 0 0 4px rgba(191, 141, 49, .12);
}
.hero-tag {
    padding: 5px 11px;
    margin-left: 5px;
    border: 1px solid #e8ddd0;
    border-radius: 99px;
    color: #75685a;
    background: rgba(255,255,255,.62);
    font-size: 9px;
    letter-spacing: 1px;
}
#hero h1 {
    margin: 0;
    color: #211a14;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(38px, 4.1vw, 54px);
    font-weight: 700;
    line-height: 1.06;
    letter-spacing: .2px;
}
#hero h1 span { color: #987022; text-shadow: 0 2px 0 rgba(255,255,255,.35); }
#hero p { margin: 11px 0 0; color: #716456; font-size: 15px; line-height: 1.6; }
.hero-brandmark {
    position: relative;
    z-index: 1;
    display: block;
    width: 178px;
    height: 142px;
    flex: 0 0 178px;
    margin: 0 0 0 28px;
    overflow: visible;
}
.hero-brandmark svg { width: 100%; height: 100%; overflow: visible; }

.workspace-grid { gap: 20px !important; align-items: stretch !important; }
.surface-card, .panel {
    background: linear-gradient(145deg, #fffefa 0%, #fffdf9 100%) !important;
    color: #211a14 !important;
    border: 1px solid rgba(255,255,255,.96) !important;
    border-radius: 25px !important;
    padding: 21px !important;
    box-shadow: 0 13px 30px rgba(83, 59, 35, .10), inset 0 1px 0 #fff !important;
    backdrop-filter: blur(12px);
}
.section-head { display: flex; align-items: flex-start; gap: 12px; margin: 0 0 16px; }
.section-symbol {
    display: grid;
    place-items: center;
    width: 37px;
    height: 37px;
    flex: 0 0 37px;
    border-radius: 12px;
    color: #916313 !important;
    background: linear-gradient(145deg, #f7eddd, #f2e7d6) !important;
    border: 1px solid #ebddc9 !important;
    font-size: 17px;
}
.section-head h2 { margin: 0; color: #241c15 !important; font-size: 18px; font-weight: 780; letter-spacing: -.2px; }
.section-head p { margin: 4px 0 0; color: #776a5d !important; font-size: 12px; line-height: 1.5; }

#conversation-chat, #conversation-chat > *, #conversation-chat .wrap,
#conversation-chat .container, #conversation-chat .message-wrap,
#conversation-chat [data-testid="chatbot"] {
    background: linear-gradient(150deg, #f5ede4 0%, #f8f2eb 100%) !important;
    color: #30251b !important;
    border-color: #e8d9c9 !important;
}
#text-shopping-chat {
    overflow: hidden;
    border: 1px solid #e8d9c9 !important;
    border-radius: 18px !important;
    box-shadow: inset 0 1px 5px rgba(83, 59, 35, .035);
    background: #fffdf9 !important;
}
#text-shopping-chat .message { border-radius: 15px !important; }
#message-input, #message-input > *, #message-input .wrap,
#message-input textarea, textarea, input, select {
    border: 1px solid #e3d3c2 !important;
    border-radius: 13px !important;
    background: linear-gradient(145deg, #f7f0e8, #f9f4ee) !important;
    color: #30251b !important;
    box-shadow: inset 0 1px 2px rgba(83, 59, 35, .025) !important;
}
textarea::placeholder, input::placeholder { color: #897b6d !important; }
label, .label-wrap span { color: #6e6052 !important; font-weight: 700 !important; }
.gradio-container .prose h3 { color: #2a2119 !important; }
.gradio-container .prose, .gradio-container p { color: #74675a; }

.gradio-container button {
    min-height: 44px;
    border-radius: 13px !important;
    font-weight: 720 !important;
    transition: transform .16s ease, filter .16s ease, box-shadow .16s ease;
}
.gradio-container button:hover { filter: brightness(1.035); transform: translateY(-1px); }
.gradio-container button.primary, .gradio-container .primary {
    border: 1px solid rgba(255,255,255,.3) !important;
    background: linear-gradient(135deg, #332316 0%, #1e1711 100%) !important;
    color: #fffdf9 !important;
    box-shadow: 0 7px 17px rgba(58, 38, 20, .17) !important;
}
.gradio-container button.secondary {
    border: 1px solid #e1d1bf !important;
    background: linear-gradient(135deg, #fffdf9, #f5ede3) !important;
    color: #654719 !important;
    box-shadow: 0 4px 12px rgba(83, 59, 35, .07) !important;
}
.chat-compose-row { align-items: end !important; gap: 10px !important; }
#shopping-send { min-height: 56px !important; min-width: 92px !important; }
.status-note { color: #776a5d !important; font-size: 12px; line-height: 1.55; }
.right-rail { gap: 16px !important; }

#text-shopping-chat .user,
#text-shopping-chat .user *,
#text-shopping-chat [data-testid="user"],
#text-shopping-chat [data-testid="user"] * {
    background-color: #ead9c3 !important;
    color: #30251b !important;
    border-color: #d9c4a8 !important;
}
#text-shopping-chat .bot,
#text-shopping-chat .bot *,
#text-shopping-chat [data-testid="assistant"],
#text-shopping-chat [data-testid="assistant"] * {
    color: #30251b !important;
}
#text-shopping-chat {
    --color-accent-soft: #ead9c3 !important;
    --body-text-color: #30251b !important;
}
#shopping-recommendations {
    color: #30251b !important;
    line-height: 1.65 !important;
}
#shopping-recommendations h3 { color: #563d20 !important; margin-bottom: 6px !important; }
#shopping-recommendations hr { border-color: #e7d8c8 !important; }

/* Quick suggestion chips */
.quick-suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 10px 0;
}
.quick-suggestions button {
    padding: 8px 16px !important;
    border: 1px solid #d8c6b3 !important;
    border-radius: 20px !important;
    background: #f5ede3 !important;
    color: #654719 !important;
    font-size: 13px !important;
    cursor: pointer;
    min-height: 36px !important;
}
.quick-suggestions button:hover {
    background: #ead9c3 !important;
    border-color: #c4aa8b !important;
}

footer { opacity: .48 !important; }

@media (max-width: 850px) {
    .gradio-container { padding: 14px 12px 32px !important; }
    #hero { min-height: 150px; padding: 23px 21px; }
    #hero h1 { font-size: 36px; }
    .hero-brandmark { width: 88px; height: 80px; flex-basis: 88px; margin-left: 8px; }
    .surface-card, .panel { padding: 15px !important; }
}
@media (max-width: 520px) {
    .hero-brandmark { width: 62px; height: 62px; flex-basis: 62px; }
    #hero { padding: 20px 17px; }
    .hero-tag { display: none; }
    #hero p { font-size: 13px; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; scroll-behavior: auto !important; } }
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title=APP_NAME,
    theme=gr.themes.Soft(primary_hue="amber", secondary_hue="orange", neutral_hue="stone"),
    css=CUSTOM_CSS,
) as demo:
    session_state = gr.State(initial_state())

    # Hero banner
    gr.HTML("""
    <header id="hero">
        <div class="hero-copy">
            <div class="hero-kicker"><span class="hero-dot"></span> AI PRODUCT SUGGESTION WORKSPACE <span class="hero-tag">DEMO STORE</span></div>
            <h1>AAHVANI <span>AI</span></h1>
            <p>Your personal AI shopping assistant — describe what you need and get instant product suggestions through natural conversation.</p>
        </div>
        <div class="hero-brandmark" aria-label="AAHVANI AI headset and soundwave logo">
            <svg viewBox="0 0 220 160" role="img" aria-hidden="true">
                <defs>
                    <linearGradient id="aahvaniGold" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0" stop-color="#d4a447"/>
                        <stop offset=".48" stop-color="#a8781d"/>
                        <stop offset="1" stop-color="#765015"/>
                    </linearGradient>
                </defs>
                <path d="M43 83 C43 28 79 9 113 9 C149 9 177 34 177 83"
                      fill="none" stroke="url(#aahvaniGold)" stroke-width="12" stroke-linecap="round"/>
                <path d="M43 77 C36 77 31 83 31 91 L31 113 C31 122 37 128 46 128 L51 128 L51 77 Z"
                      fill="url(#aahvaniGold)"/>
                <path d="M177 77 L169 77 L169 128 L176 128 C185 128 191 122 191 113 L191 91 C191 83 185 77 177 77 Z"
                      fill="url(#aahvaniGold)"/>
                <g fill="url(#aahvaniGold)">
                    <rect x="73" y="81" width="8" height="20" rx="4"/>
                    <rect x="88" y="68" width="8" height="45" rx="4"/>
                    <rect x="103" y="52" width="8" height="70" rx="4"/>
                    <rect x="118" y="72" width="8" height="35" rx="4"/>
                    <rect x="133" y="82" width="8" height="18" rx="4"/>
                    <rect x="148" y="76" width="7" height="29" rx="3.5"/>
                </g>
                <path d="M181 117 C181 136 166 143 150 142 C143 142 138 139 133 135"
                      fill="none" stroke="url(#aahvaniGold)" stroke-width="6" stroke-linecap="round"/>
                <circle cx="131" cy="134" r="7" fill="url(#aahvaniGold)"/>
                <path d="M28 145 C76 157 139 151 190 116 C205 106 210 94 211 83"
                      fill="none" stroke="url(#aahvaniGold)" stroke-width="4" stroke-linecap="round"/>
            </svg>
        </div>
    </header>
    """)

    with gr.Row(elem_classes=["workspace-grid"]):
        # Left: Chat panel
        with gr.Column(scale=8, elem_classes=["surface-card", "conversation-panel"]):
            gr.HTML("""
            <div class="section-head">
                <div class="section-symbol">✦</div>
                <div>
                    <h2>Chat with AAHVANI</h2>
                    <p>Tell me what you want to buy — I'll ask the right questions and suggest the best products for you.</p>
                </div>
            </div>
            """)

            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "Hi, I'm AAH-vuh-nee AI. 👋\n\nWhat are you looking to buy today? Just tell me the product and your budget — for example:\n\n• *\"I want a phone under 25000\"*\n• *\"Suggest me a laptop for coding under 60k\"*\n• *\"Best headphones under 5000\"*"}],
                height=460,
                label="Text conversation",
                type="messages",
                elem_id="text-shopping-chat",
            )

            with gr.Row(elem_classes=["chat-compose-row"]):
                message = gr.Textbox(
                    placeholder="Try: I need a smartphone under ₹20,000...",
                    label="Message AAHVANI",
                    lines=2,
                    scale=7,
                    elem_id="shopping-message",
                )
                send = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    elem_id="shopping-send",
                )

            # Quick suggestion buttons
            gr.HTML("""
            <div class="quick-suggestions" id="quick-suggestions">
                <button onclick="document.querySelector('#shopping-message textarea').value = 'I want to buy a phone under 25000'; document.querySelector('#shopping-send').click();">📱 Phone under ₹25K</button>
                <button onclick="document.querySelector('#shopping-message textarea').value = 'Suggest a laptop for coding under 60000'; document.querySelector('#shopping-send').click();">💻 Laptop under ₹60K</button>
                <button onclick="document.querySelector('#shopping-message textarea').value = 'Best headphones under 5000'; document.querySelector('#shopping-send').click();">🎧 Headphones under ₹5K</button>
                <button onclick="document.querySelector('#shopping-message textarea').value = 'I need running shoes under 5000'; document.querySelector('#shopping-send').click();">👟 Shoes under ₹5K</button>
                <button onclick="document.querySelector('#shopping-message textarea').value = 'Suggest a smartwatch under 3000'; document.querySelector('#shopping-send').click();">⌚ Watch under ₹3K</button>
            </div>
            """)

            gemini_status = "✅ Gemini API connected" if (os.getenv("GEMINI_API_KEY", "").strip() and GEMINI_AVAILABLE) else "⚠️ Gemini unavailable — using local catalogue"
            gr.Markdown(
                f"{gemini_status}. Powered by the AAHVANI demo catalogue.",
                elem_classes=["status-note"],
            )

        # Right: Recommendations panel
        with gr.Column(scale=5, elem_classes=["right-rail"]):
            with gr.Group(elem_classes=["surface-card"], elem_id="recommendations-panel"):
                gr.HTML("""
                <div class="section-head">
                    <div class="section-symbol">✧</div>
                    <div><h2>Recommended for you</h2><p>Up to three matching products from the demo catalogue.</p></div>
                </div>
                """)
                recommendations_panel = gr.Markdown(
                    "Your recommendations will appear here once AAHVANI understands what you need.",
                    elem_id="shopping-recommendations",
                )

            with gr.Group(elem_classes=["surface-card"]):
                gr.HTML("""
                <div class="section-head">
                    <div class="section-symbol">ℹ</div>
                    <div><h2>How it works</h2><p>Simple 3-step process to get personalized suggestions.</p></div>
                </div>
                """)
                gr.Markdown("""
**1. Tell me what you need** — Product type and budget\n
**2. Answer follow-ups** — Brand preference and core feature priority\n
**3. Get suggestions** — Up to 3 curated options from the demo catalogue\n
---
*Categories available:* Smartphones, Laptops, Headphones, Tablets, Smartwatches, Speakers, TVs, Footwear, Backpacks

*All listings and prices are illustrative demo data.*
                """)

    # Wire events
    send.click(
        product_chat,
        [message, chatbot, session_state],
        [chatbot, session_state, recommendations_panel, message],
    )
    message.submit(
        product_chat,
        [message, chatbot, session_state],
        [chatbot, session_state, recommendations_panel, message],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("HOST", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7861")),
        share=False,
    )
