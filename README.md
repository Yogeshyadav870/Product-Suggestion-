# AAHVANI AI — Product Suggestion Dashboard

A conversational AI product recommendation dashboard powered by **Gemini AI** and a local demo catalogue. Built as part of the **AAHVANI AI** project ecosystem.

![AAHVANI AI](assets/aahvani_logo.svg)

## 🚀 Features

- **AI-Powered Chat** — Natural conversation with Gemini AI to understand your product needs
- **Smart Follow-ups** — Asks about category, budget, brand preference, and core feature priority
- **Instant Recommendations** — Up to 3 curated product suggestions from the demo catalogue
- **Local Fallback** — Works without Gemini using rule-based product matching
- **AAHVANI Theme** — Warm gold/cream design matching the AAHVANI AI brand

## 📦 Supported Categories

- 📱 Smartphones
- 💻 Laptops
- 🎧 Headphones
- 📋 Tablets
- ⌚ Smartwatches
- 🔊 Speakers
- 📺 Televisions
- 👟 Footwear
- 🎒 Backpacks

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Yogeshyadav870/AAHVANI-AI.git
cd AAHVANI-AI
```

### 2. Create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API key

Copy `.env.example` to `.env` and add your Gemini API key:

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 5. Run the dashboard

```bash
python app.py
```

Open your browser to `http://127.0.0.1:7860`

## 💬 Usage Examples

- *"I want to buy a phone under 25000"*
- *"Suggest a laptop for coding under 60k"*
- *"Best headphones under 5000 for music"*
- *"I need running shoes under 5000"*
- *"Suggest a smartwatch for fitness under 3000"*

## 🏗️ Architecture

```
Product Suggestion/
├── app.py                  # Main Gradio dashboard application
├── catalogue.py            # Product catalogue and search engine
├── data/
│   └── product_catalogue.json  # Demo product catalogue (24 products)
├── assets/
│   └── aahvani_logo.svg    # AAHVANI AI brand logo
├── requirements.txt        # Python dependencies
├── .env                    # API key configuration (not in git)
├── .env.example            # Template for .env
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🤖 How the AI Conversation Works

1. User says what they want (e.g., "phone under 25000")
2. AAHVANI asks follow-up questions one at a time:
   - What matters most? (camera, gaming, battery, everyday)
   - Preferred brand? (or compare all)
3. Once all details are gathered, it searches the catalogue
4. Up to 3 matching products are displayed in the recommendation panel

## 📝 Notes

- All product listings and prices are **illustrative demo data**, not live retailer stock
- The dashboard works with or without a Gemini API key (local fallback available)
- Part of the [AAHVANI AI](https://github.com/Yogeshyadav870/AAHVANI-AI) project

## 📄 License

MIT License — Part of the AAHVANI AI project by Yogesh Yadav.
