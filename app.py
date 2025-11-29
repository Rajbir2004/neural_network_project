import os
import time
from flask import Flask, render_template, request, jsonify
from typing import Dict, Any, Optional
import requests

# Optional import for Gemini (google-genai). Only required if you enable Gemini.
try:
    import google.genai as genai
except Exception:
    genai = None

app = Flask(__name__, template_folder='templates')

LLM_MODE = os.environ.get("LLM_MODE", "gemini")  # "gemini" or "mock"
# CORRECT — read the env var named GEMINI_API_KEY
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

class LLMAdapter:
    def __init__(self, mode: str = "mock"):
        self.mode = mode
        if mode == "gemini" and genai is None:
            raise RuntimeError("google-genai SDK not available. Install google-genai or set LLM_MODE=mock.")
        if mode == "gemini" and not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY env var not set. Provide a valid Google API key or switch to mock mode.")
        if mode == "gemini":
            # Configure client if SDK available
            # The SDK picks up GEMINI_API_KEY from env normally; we'll set it explicitly when possible.
            os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY","")
            self.client = genai.Client() if genai is not None else None
        else:
            self.client = None

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if self.mode == "mock":
            
            if "outline" in prompt.lower():
                return ("Day 1: Arrival, city walk, local dinner.\n"
                        "Day 2: Museums and landmarks.\n"
                        "Day 3: Day trip to nearby nature spot.\n")
            if "detail day 1" in prompt.lower():
                return ("09:00 Arrive, check-in\n11:00 Brunch at Cafe A\n14:00 City museum (2 hrs)\n19:00 Dinner at local restaurant\n")
            if "budget" in prompt.lower():
                return ("Estimated total: 600 USD (flights 300, hotels 200, food & transport 100). Tips: book trains early.\n")
            # fallback
            return "Generated (mock) response for prompt:\n" + (prompt[:500] + ("..." if len(prompt) > 500 else ""))
        else:
            # Call Gemini via google-genai SDK (per Google GenAI docs).
            # If google.genai isn't available, try direct REST (not implemented here).
            if not self.client:
                raise RuntimeError("Gemini client not initialized.")
            resp = self.client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            # resp structure varies; try to extract text
            text = None
            try:
                text = getattr(resp, 'text', None)
                if not text:
                    # SDK sometimes offers candidates
                    candidates = getattr(resp, 'candidates', None)
                    if candidates:
                        text = "".join([c.content.parts[0].text for c in candidates if c.content.parts])
            except Exception:
                text = str(resp)
            return text or str(resp)


def planner_agent(inputs: Dict[str, Any], llm: LLMAdapter) -> Dict[str, Any]:
    prompt = f"""You are a trip planner assistant.
Create a concise day-by-day outline for a trip.
Destination: {inputs.get('destination')}
Dates: {inputs.get('start_date')} to {inputs.get('end_date')}
Interests: {inputs.get('interests')}
Budget: {inputs.get('budget')}
Output a short day-by-day outline labelled 'outline'."""
    outline = llm.generate("outline:\n" + prompt, max_tokens=400)
    return {"outline": outline}

def details_agent(inputs: Dict[str, Any], day_text: str, llm: LLMAdapter) -> Dict[str, Any]:
    prompt = f"""You are the detailer. Expand the following day plan into timed activities for readability.
Day plan: {day_text}
Destination: {inputs.get('destination')}
Output a bullet/time-stamped list."""
    details = llm.generate("detail day: " + prompt, max_tokens=400)
    return {"details": details}

def budget_agent(inputs: Dict[str, Any], llm: LLMAdapter) -> Dict[str, Any]:
    prompt = f"""You are a budget assistant. Provide a short estimated budget and 3 cost-saving tips.
Inputs: destination={inputs.get('destination')}, budget={inputs.get('budget')}
Output: Estimated total and tips."""
    budget = llm.generate("budget:\n" + prompt, max_tokens=200)
    return {"budget": budget}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plan', methods=['POST'])
def plan():
    data = request.json or request.form
    inputs = {
        'destination': data.get('destination',''),
        'start_date': data.get('start_date',''),
        'end_date': data.get('end_date',''),
        'interests': data.get('interests',''),
        'budget': data.get('budget',''),
    }

    try:
        llm = LLMAdapter(mode=LLM_MODE)
    except Exception as e:
        return jsonify({'error': 'llm_init_failed', 'detail': str(e)}), 500

    # Run planner agent
    planner_out = planner_agent(inputs, llm)
    outline_text = planner_out.get('outline','')

    # Simple parsing: split outline into lines and generate details for first 3 lines
    days = [line.strip() for line in outline_text.splitlines() if line.strip()]
    day_details = []
    for i, day in enumerate(days[:5]):  # limit to first 5 days for performance
        det = details_agent(inputs, day, llm)
        day_details.append({'day': day, 'details': det['details']})

    budget_res = budget_agent(inputs, llm)

    result = {
        'inputs': inputs,
        'outline': outline_text,
        'day_details': day_details,
        'budget': budget_res['budget'],
        'meta': {
            'llm_mode': LLM_MODE,
            'gemini_model': GEMINI_MODEL if LLM_MODE=='gemini' else None
        }
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
