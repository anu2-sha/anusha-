import os
import re
import json
import joblib
import requests
import jsonschema
from jsonschema import validate

# ---------------------------------------------------------------------------
# 1. API Configuration & Connection Setup
# ---------------------------------------------------------------------------
# Retrieve the API key from the environment variable (Do not hardcode)
API_KEY = os.environ.get('LLM_API_KEY')
# Using OpenRouter as an example HTTP POST public LLM API provider
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "meta-llama/llama-3-8b-instruct:free" 

def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    """
    Constructs payload, handles HTTP POST request, checks status codes, 
    and handles potential network errors safely.
    """
    if not API_KEY:
        print("Error: LLM_API_KEY environment variable is not set.")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            print(f"API Error: Status Code {response.status_code} - {response.text}")
            return None
        
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    except Exception as e:
        print(f"Exception encountered during API Call: {e}")
        return None

# Simple sanity test verification
print("--- 1. Testing LLM Connection ---")
test_resp = call_llm("You are a literal echo.", "Reply with only the word: hello", temperature=0.0)
print(f"Sanity Test Response: {test_resp.strip() if test_resp else 'Failed to connect'}\n")


# ---------------------------------------------------------------------------
# 2. PII Guardrail Implementation
# ---------------------------------------------------------------------------
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


# ---------------------------------------------------------------------------
# 3. Schema Definitions & Mock Model Utilities (Track C Setup)
# ---------------------------------------------------------------------------
# Define Target JSON Schema with 5 mandatory scalar fields
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"]
}

FALLBACK_DICT = {
    "prediction_label": None,
    "confidence_level": None,
    "top_reason": None,
    "second_reason": None,
    "next_step": None
}

# Mock model setup for reproducibility if external file isn't present
class MockModel:
    def predict(self, array): return [1]
    def predict_proba(self, array): return [[0.15, 0.85]]

def encode_record(features_dict):
    # Simulated preprocessing (e.g., matching training columns format)
    return [[features_dict.get(k, 0) for k in sorted(features_dict.keys())]]

# Attempt to load actual Part 3 model file, use robust fallback if missing
try:
    best_model = joblib.load('best_model.pkl')
except Exception:
    best_model = MockModel()


# ---------------------------------------------------------------------------
# 4. Prompt Engineering Setup
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert ML Explainer. Your job is to generate a clean, structured JSON "
    "explanation given feature values, a predicted class, and a predicted probability.\n"
    "You must return ONLY a valid JSON object matching this schema strict format:\n"
    "{\n"
    "  \"prediction_label\": \"string\",\n"
    "  \"confidence_level\": \"low|medium|high\",\n"
    "  \"top_reason\": \"string\",\n"
    "  \"second_reason\": \"string\",\n"
    "  \"next_step\": \"string\"\n"
    "}\n"
    "Do not include markdown code block syntax (like ```json), introduction, or commentary."
)

USER_PROMPT_TEMPLATE = (
    "Feature values: {features}\n"
    "Predicted Class: {predicted_class}\n"
    "Predicted Probability: {predicted_proba}"
)


# ---------------------------------------------------------------------------
# 5. Pipeline Orchestrator with Parsers and Validations
# ---------------------------------------------------------------------------
def run_explanation_pipeline(features, temp=0.0):
    user_str = str(features)
    
    # Run Guardrail Check
    if has_pii(user_str):
        print("Input blocked: PII detected.")
        return None

    # Step 1: Compute Predictions
    encoded = encode_record(features)
    pred_class = int(best_model.predict(encoded)[0])
    pred_proba = float(best_model.predict_proba(encoded)[0][pred_class])

    # Step 2: Format User Prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        features=json.dumps(features),
        predicted_class=pred_class,
        predicted_proba=round(pred_proba, 4)
    )

    # Step 3: Run LLM Execution
    raw_response = call_llm(SYSTEM_PROMPT, user_prompt, temperature=temp)
    if not raw_response:
        return FALLBACK_DICT

    # Step 4: Parse & Validate
    cleaned_response = raw_response.strip()
    try:
        parsed_json = json.loads(cleaned_response)
        validate(instance=parsed_json, schema=EXPLANATION_SCHEMA)
        return parsed_json
    except json.JSONDecodeError:
        print("Validation Failed: Invalid JSON format returned.")
        return FALLBACK_DICT
    except jsonschema.ValidationError as e:
        print(f"Validation Failed: Schema rules violated. Error: {e.message}")
        return FALLBACK_DICT


# ---------------------------------------------------------------------------
# 6. Comprehensive Demonstration Runs
# ---------------------------------------------------------------------------
print("\n--- 2. Guardrail Demonstrations ---")
pii_input = {"age": 34, "income": 75000, "email": "user@example.com"}
clean_input = {"age": 34, "income": 75000}

print("Testing PII input:")
run_explanation_pipeline(pii_input)

print("\nTesting Clean input:")
res = run_explanation_pipeline(clean_input)
print(f"Outcome: {res}")


print("\n--- 3. Temperature Validation and Execution (Three Distinct Inputs) ---")
inputs = [
    {"credit_score": 790, "debt_to_income": 0.12, "history_years": 15},
    {"credit_score": 580, "debt_to_income": 0.54, "history_years": 2},
    {"credit_score": 670, "debt_to_income": 0.31, "history_years": 6}
]

for idx, item in enumerate(inputs, 1):
    print(f"\n>> Processing Test Input Vector #{idx}: {item}")
    out_t0 = run_explanation_pipeline(item, temp=0.0)
    out_t7 = run_explanation_pipeline(item, temp=0.7)
    print(f"  [Temp=0.0 Output]: {out_t0}")
    print(f"  [Temp=0.7 Output]: {out_t7}")
