XAI Pipeline

Step 1:
Behavior dataset generation from trained IPPO policies.

Step 2:
Variance-based feature signal analysis using surrogate observations.
(Note: due to single dominant action distributions, SHAP fell back to variance analysis.)

Step 3:
Deterministic rule-based diagnosis engine for service classification.

Step 4:
Evidence-guided LLM verbalization.
LLM does not generate diagnoses.
LLM only interprets pre-computed evidence
(diagnosis, latency, confidence, feature signals).