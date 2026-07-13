Part 4 — LLM-Powered Feature: Tabular Batch Scoring DocumentationChosen Track: Track B (Tabular Record Batch Scoring)1. Prompt Design DecisionsVerbatim System PromptPlaintextYou are an automated tabular record risk assessment system. Your job is to analyze incoming user data records and output a strict, valid JSON assessment object according to the scoring rubric rules. Do not include markdown formatting, backticks, or any conversational prose. 

SCORING RUBRIC:
- risk_tier: 'high' if debt_to_income > 45 or credit_score < 600; 'medium' if debt_to_income is 30-45 or credit_score is 600-700; else 'low'.
- flag_for_review: true if risk_tier is 'high' or if missed_payments > 2; else false.
- confidence: 'high' if credit_history_years > 5; else 'medium' or 'low'.

WORKED EXAMPLE:
Input: {"credit_score": 580, "debt_to_income": 50, "missed_payments": 3, "credit_history_years": 8}
Output: {"risk_tier": "high", "flag_for_review": true, "primary_signal": "Low credit score coupled with high debt-to-income ratio.", "confidence": "high", "recommended_action": "Deny automated approval and route to senior underwriter review."}
User Prompt TemplatePlaintext{
  "credit_score": <int>,
  "debt_to_income": <int>,
  "missed_payments": <int>,
  "credit_history_years": <int>
}
Choice of Temperature = 0Setting temperature=0.0 ensures the model behaves deterministically. For high-stakes analytical pipelines mapping structural tabular keys to fixed-schema parameters, randomness can lead to syntax violations (e.g., unexpected trailing commas or unquoted keys) or logical drift away from the rubric definitions. Lowering the temperature guarantees predictable outputs aligned precisely with the mathematical logic defined in the instructions.2. Temperature A/B ComparisonPerformance Evaluation TableInputOutput at temp=0Output at temp=0.7Key DifferenceRecord 1 (Low Risk){"risk_tier": "low", "flag_for_review": false, "primary_signal": "Healthy indicators across all fields.", "confidence": "high", "recommended_action": "Approve application automatically."}{"risk_tier": "low", "flag_for_review": false, "primary_signal": "Excellent profile with zero flags", "confidence": "high", "recommended_action": "Standard issue approval workflow."}The structure remained stable, but the natural language strings (primary_signal, recommended_action) changed due to probabilistic generation.Record 2 (Med Risk){"risk_tier": "medium", "flag_for_review": false, "primary_signal": "Moderate debt-to-income balance.", "confidence": "low", "recommended_action": "Proceed with regular underwriting guidelines."}
