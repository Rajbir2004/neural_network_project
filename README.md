# Crew-style Trip Planner (Flask)
This repository is a **manual** Crew-style trip planner implemented in Flask.
It runs three simple agents server-side (planner, detailer, budgeter) and can use either:
- **mock** LLM mode (free, deterministic—default), **or**
- **gemini** mode (calls Google Gemini via `google-genai` SDK — requires GEMINI_API_KEY).

## How to run locally (mock mode — no API keys needed)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export LLM_MODE=mock
python app.py
```

## How to use Gemini (may incur costs / requires API key)
1. Get a Gemini API key from Google AI Studio / Google Cloud; set it as `GEMINI_API_KEY`.
2. Set `LLM_MODE=gemini` and optionally `GEMINI_MODEL` (e.g. gemini-2.5-flash).
3. Run the app as above.

**Important:** Google offers free tiers and credits in some cases, but you should check Google AI Studio / Vertex AI pricing before heavy usage. See Google docs:
- https://ai.google.dev/gemini-api/docs/quickstart
- https://cloud.google.com/vertex-ai/generative-ai/pricing
(Links in docs may change—follow official Google AI Studio instructions.)

## Deploying to Render
1. Push repository to GitHub.
2. Create a new Web Service on Render and connect the repo.
3. Set Build command: `pip install -r requirements.txt`
   Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 3`
4. Add environment variables in Render:
   - `LLM_MODE` = `mock` (or `gemini`)
   - `GEMINI_API_KEY` = your key (only if using gemini)
   - `GEMINI_MODEL` = optional (default `gemini-2.5-flash`)
5. Deploy.

## Notes
- The mock mode is free and deterministic — recommended if you want zero cost.
- Gemini access may be free for light usage in Google AI Studio, but heavy usage can incur charges. Always check pricing.
