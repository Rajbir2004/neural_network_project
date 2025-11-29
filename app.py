import os
import time
from flask import Flask, render_template, request, jsonify
from typing import Dict, Any, Optional

# Try to import a real CrewAI library. If not available, use a tiny local shim so
# the project still runs with the same semantics (agents -> tasks -> crew run).
try:
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew
    _CREWAI_AVAILABLE = True
except Exception:
    # Minimal shim: Agent wraps a callable, Task is thin, Crew runs tasks sequentially.
    _CREWAI_AVAILABLE = False

    class CrewAgent:
        def __init__(self, name: str, fn):
            self.name = name
            self.fn = fn

        def run(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    class CrewTask:
        def __init__(self, agent: CrewAgent, inputs: Dict[str, Any], **meta):
            self.agent = agent
            self.inputs = inputs
            self.meta = meta

        def run(self):
            return self.agent.run(self.inputs)

    class Crew:
        def __init__(self, tasks=None):
            self.tasks = tasks or []

        def add_task(self, task: CrewTask):
            self.tasks.append(task)

        def run(self):
            results = []
            for t in self.tasks:
                results.append(t.run())
            return results

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
            os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
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
            return "Generated (mock) response for prompt:\n" + (prompt[:500] + ("..." if len(prompt) > 500 else ""))
        else:
            if not self.client:
                raise RuntimeError("Gemini client not initialized.")
            resp = self.client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = None
            try:
                text = getattr(resp, 'text', None)
                if not text:
                    candidates = getattr(resp, 'candidates', None)
                    if candidates:
                        text = "".join([c.content.parts[0].text for c in candidates if c.content.parts])
            except Exception:
                text = str(resp)
            return text or str(resp)

# --- Define the original agents as normal functions (keeps same functionality) ---

def planner_fn(inputs: Dict[str, Any], llm: LLMAdapter) -> Dict[str, Any]:
    prompt = f"""You are a trip planner assistant.
Create a concise day-by-day outline for a trip.
Destination: {inputs.get('destination')}
Dates: {inputs.get('start_date')} to {inputs.get('end_date')}
Interests: {inputs.get('interests')}
Budget: {inputs.get('budget')}
Output a short day-by-day outline labelled 'outline'."""
    outline = llm.generate("outline:\n" + prompt, max_tokens=400)
    return {"outline": outline}


def details_fn(inputs: Dict[str, Any], day_text: str, llm: LLMAdapter) -> Dict[str, Any]:
    prompt = f"""You are the detailer. Expand the following day plan into timed activities for readability.
Day plan: {day_text}
Destination: {inputs.get('destination')}
Output a bullet/time-stamped list."""
    details = llm.generate("detail day: " + prompt, max_tokens=400)
    return {"details": details}


def budget_fn(inputs: Dict[str, Any], llm: LLMAdapter) -> Dict[str, Any]:
    prompt = f"""You are a budget assistant. Provide a short estimated budget and 3 cost-saving tips.
Inputs: destination={inputs.get('destination')}, budget={inputs.get('budget')}
Output: Estimated total and tips."""
    budget = llm.generate("budget:\n" + prompt, max_tokens=200)
    return {"budget": budget}

# --- Wrap the functions into CrewAgent objects ---

def make_agents(llm: LLMAdapter):
    planner_agent = CrewAgent("planner", lambda inputs: planner_fn(inputs, llm))
    details_agent = CrewAgent("details", lambda data: details_fn(data['inputs'], data['day_text'], llm))
    budget_agent = CrewAgent("budget", lambda inputs: budget_fn(inputs, llm))
    return planner_agent, details_agent, budget_agent

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

    # Create agents (either real CrewAI agents or the shim above)
    planner_agent, details_agent, budget_agent = make_agents(llm)

    # Create tasks and run via Crew object. We run planner first, then detail tasks, then budget.
    crew = Crew()

    # Task 1: planner
    task_planner = CrewTask(agent=planner_agent, inputs=inputs)
    crew.add_task(task_planner)

    # Execute planner immediately to obtain outline for subsequent detail tasks
    planner_res = task_planner.run() if hasattr(task_planner, 'run') else planner_agent.run(inputs)
    outline_text = planner_res.get('outline', '') if isinstance(planner_res, dict) else str(planner_res)

    # Simple parsing: split outline into lines and generate details for first 5 lines
    days = [line.strip() for line in outline_text.splitlines() if line.strip()]
    day_details = []
    detail_tasks = []

    for i, day in enumerate(days[:5]):
        # Build inputs for detail agent (we send a small dict so the agent signature is uniform)
        detail_inputs = {'inputs': inputs, 'day_text': day}
        t = CrewTask(agent=details_agent, inputs=detail_inputs, meta={'day_index': i})
        detail_tasks.append(t)
        crew.add_task(t)

    # Add budget task
    task_budget = CrewTask(agent=budget_agent, inputs=inputs)
    crew.add_task(task_budget)

    # Run detail + budget tasks. In this simple implementation we run sequentially and collect outputs.
    # If you have a real Crew implementation that supports parallelism and richer orchestration,
    # it will take over here.
    for t in detail_tasks:
        res = t.run()
        # res expected to be dict with 'details'
        day_details.append({'day': t.inputs['day_text'], 'details': res.get('details') if isinstance(res, dict) else res})

    budget_res = task_budget.run()
    budget_text = budget_res.get('budget') if isinstance(budget_res, dict) else budget_res

    result = {
        'inputs': inputs,
        'outline': outline_text,
        'day_details': day_details,
        'budget': budget_text,
        'meta': {
            'llm_mode': LLM_MODE,
            'gemini_model': GEMINI_MODEL if LLM_MODE=='gemini' else None,
            'crewai_present': _CREWAI_AVAILABLE
        }
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
